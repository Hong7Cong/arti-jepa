"""rtMRI clip dataset implementing Arti-JEPA Part A (A.1-A.7).

Pipeline per sampled clip (order matters, per A.1 "normalize before resizing"):
  1. decode native frames with decord
  2. A.2 temporal: resample 83.28 -> target_fps by *linear temporal
     interpolation* on a uniform time grid -- realized here in the dataloader,
     never by re-encoding. Non-integer ratios are handled exactly. Two sampling
     modes:
       * "crop" -- one window per video (random start in train, centre in eval),
         optionally speed-perturbed. One dataset item == one video.
       * "tile" -- resample the *whole* video to target_fps and split it into
         consecutive, non-overlapping ``frames_per_clip`` chunks; each chunk is
         an independent dataset item (full temporal coverage, no speed perturb).
  3. A.1 intensity: percentile clip + per-clip z-score (rtMRI-specific, NOT
     ImageNet stats) -> then optional global train-set channel normalization.
  4. A.7 augmentation (train only): small translation, intensity jitter,
     additive Gaussian (Rician proxy), temporal speed perturbation. NEVER hflip
     / large rotation / colour jitter (they destroy phonetic meaning).
  5. A.3 spatial: bicubic resize 84->256/128, or reflect-pad 84->96.
  6. A.4 channels: replicate the single grayscale channel x3.

__getitem__ returns ``([clip], label, [clip_indices])`` -- the exact tuple the
V-JEPA ``MaskCollator`` and training loop expect (clip is ``[3, T, H, W]``).
"""

import csv
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from decord import VideoReader, cpu


@dataclass
class PreprocConfig:
    # -- A.2 temporal
    target_fps: float = 50.0
    frames_per_clip: int = 32           # even (tubelet T=2); 1.28 s @ 25 fps
    sampling: str = "crop"               # {"crop": one window/video, "tile": all
                                         #  target_fps frames split into chunks}
    random_temporal_crop: bool = True    # crop mode: random start (train) vs centre
    # -- A.1 intensity
    clip_percentiles: tuple = (1.0, 99.0)
    intensity_norm: str = "zscore"      # {"zscore", "minmax", "none"}
    grayscale_mean: float = 0.0          # global train stats (channel norm)
    grayscale_std: float = 1.0
    # -- A.3 spatial
    spatial_mode: str = "resize"        # {"resize", "pad"}
    spatial_size: int = 256              # 256 (primary) / 128 / 96(pad)
    # -- A.7 augmentation (train only)
    augment: bool = False
    translate_frac: float = 0.06         # max translation as frac of size
    intensity_jitter: float = 0.1        # +/- brightness & contrast
    gaussian_noise_std: float = 0.05
    speed_perturb: tuple = (0.9, 1.1)    # multiplies the resampling ratio
    # -- misc
    num_clips: int = 1
    tubelet_size: int = 2

    def __post_init__(self):
        assert self.frames_per_clip % self.tubelet_size == 0, (
            "frames_per_clip must be a multiple of tubelet_size (even-length clips)"
        )
        assert self.sampling in ("crop", "tile"), (
            f"unknown sampling {self.sampling!r} (expected 'crop' or 'tile')"
        )


def _to_gray(frames_u8: np.ndarray) -> torch.Tensor:
    """[K,H,W,3] uint8 RGB (grayscale-equal channels) -> [K,H,W] float32 in 0..255."""
    t = torch.from_numpy(frames_u8).float()
    return t.mean(dim=-1)


def _spatial(clip: torch.Tensor, mode: str, size: int) -> torch.Tensor:
    """[T,H,W] -> [T,size,size] via bicubic resize or reflect padding."""
    t = clip.unsqueeze(1)  # [T,1,H,W]
    if mode == "resize":
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode="bicubic", align_corners=False
        )
    elif mode == "pad":
        H = t.shape[-1]
        if size < H:
            raise ValueError(f"pad target {size} < input {H}; use mode='resize'")
        pad = size - H
        l = pad // 2
        t = torch.nn.functional.pad(t, (l, pad - l, l, pad - l), mode="reflect")
    else:
        raise ValueError(f"unknown spatial_mode {mode}")
    return t.squeeze(1)


def _augment(clip: torch.Tensor, cfg: PreprocConfig, rng: np.random.Generator):
    """Anatomically-safe augmentation on a [T,H,W] grayscale clip."""
    T, H, W = clip.shape
    # -- small translation (reflect-pad then crop a shifted window)
    if cfg.translate_frac > 0:
        m = max(1, int(round(cfg.translate_frac * H)))
        t = clip.unsqueeze(1)
        t = torch.nn.functional.pad(t, (m, m, m, m), mode="reflect")
        top = int(rng.integers(0, 2 * m + 1))
        left = int(rng.integers(0, 2 * m + 1))
        clip = t[:, 0, top:top + H, left:left + W]
    # -- intensity jitter (brightness shift + contrast scale)
    if cfg.intensity_jitter > 0:
        b = float(rng.uniform(-cfg.intensity_jitter, cfg.intensity_jitter))
        c = 1.0 + float(rng.uniform(-cfg.intensity_jitter, cfg.intensity_jitter))
        clip = (clip - clip.mean()) * c + clip.mean() + b
    # -- additive Gaussian noise (Rician proxy)
    if cfg.gaussian_noise_std > 0:
        clip = clip + torch.from_numpy(
            rng.normal(0.0, cfg.gaussian_noise_std, size=clip.shape).astype("float32")
        )
    return clip


def _intensity_norm(clip: torch.Tensor, cfg: PreprocConfig) -> torch.Tensor:
    """Percentile clip + per-clip standardization (A.1, rtMRI-specific)."""
    lo, hi = cfg.clip_percentiles
    flat = clip.flatten()
    p_lo = torch.quantile(flat, lo / 100.0)
    p_hi = torch.quantile(flat, hi / 100.0)
    clip = torch.clamp(clip, float(p_lo), float(p_hi))
    if cfg.intensity_norm == "zscore":
        std = clip.std()
        clip = (clip - clip.mean()) / (std + 1e-6)
    elif cfg.intensity_norm == "minmax":
        rng = (p_hi - p_lo)
        clip = (clip - p_lo) / (rng + 1e-6)
    return clip


class RTMRIVideoDataset(torch.utils.data.Dataset):
    """rtMRI clips with V-JEPA-compatible item structure."""

    def __init__(self, manifest, split, cfg: PreprocConfig, seed=0,
                 label_key="group_idx"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.seed = seed
        self.label_key = label_key
        with open(manifest) as f:
            rows = [r for r in csv.DictReader(f)
                    if (split is None or r.get("split") == split)]
        if not rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")
        self.rows = rows
        # tile mode: flatten (video, chunk) pairs into a single index up front
        # using the manifest's n_frames/fps (no need to open 2.3k videos here).
        self.index = self._build_tile_index() if cfg.sampling == "tile" else None

    def __len__(self):
        return len(self.index) if self.index is not None else len(self.rows)

    def _build_tile_index(self):
        """List of (row_idx, chunk_idx) for every complete chunk in every video.

        A video of ``n_src`` frames resampled to ``target_fps`` spans
        ``floor((n_src-1)/r)+1`` output frames (r = src_fps/target_fps); we keep
        the ``floor(n_out / frames_per_clip)`` complete, non-overlapping chunks
        and drop the short tail.
        """
        cfg = self.cfg
        index = []
        for ri, row in enumerate(self.rows):
            try:
                n_src = int(float(row["n_frames"]))
                src_fps = float(row["fps"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "sampling='tile' needs n_frames/fps columns in the manifest "
                    "(rebuild with `build_manifest --probe`)"
                )
            if n_src <= 1 or src_fps <= 0:
                warnings.warn(f"skipping {row['path']}: bad n_frames/fps")
                continue
            r = src_fps / cfg.target_fps
            n_out = int(math.floor((n_src - 1) / r)) + 1
            for c in range(n_out // cfg.frames_per_clip):
                index.append((ri, c))
        if not index:
            raise ValueError(f"tile index empty for split={self.split!r} "
                             f"(videos shorter than {cfg.frames_per_clip} frames "
                             f"@ {cfg.target_fps} fps?)")
        return index

    def _tile_indices(self, row, chunk):
        """Source-frame positions for chunk ``chunk`` of ``row`` on the
        target_fps grid: s_k = k * (src_fps/target_fps), k in [c*F, c*F+F)."""
        cfg = self.cfg
        r = float(row["fps"]) / cfg.target_fps
        k = chunk * cfg.frames_per_clip + np.arange(cfg.frames_per_clip, dtype=np.float64)
        return k * r

    def _sample_source_indices(self, n_src, src_fps, rng):
        cfg = self.cfg
        speed = float(rng.uniform(*cfg.speed_perturb)) if cfg.augment else 1.0
        r = (src_fps / cfg.target_fps) * speed   # source frames per output frame
        N = cfg.frames_per_clip * cfg.num_clips
        span = (N - 1) * r
        max_start = max(0.0, (n_src - 1) - span)
        if cfg.random_temporal_crop and cfg.augment and max_start > 0:
            start = float(rng.uniform(0.0, max_start))
        else:
            start = max_start / 2.0
        s = start + np.arange(N, dtype=np.float64) * r
        s = np.clip(s, 0.0, n_src - 1.0)
        return s

    def _load_clip(self, path, rng, s=None):
        cfg = self.cfg
        vr = VideoReader(path, num_threads=2, ctx=cpu(0))
        n_src = len(vr)
        if s is None:                                  # crop mode picks the window
            src_fps = float(vr.get_avg_fps())
            s = self._sample_source_indices(n_src, src_fps, rng)
        s = np.clip(s, 0.0, n_src - 1.0)               # tile positions may overrun

        f0 = np.floor(s).astype(np.int64)
        f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)

        need = np.unique(np.concatenate([f0, f1]))
        remap = {int(v): i for i, v in enumerate(need)}
        gray = _to_gray(vr.get_batch(need).asnumpy())  # [K,H,W]
        i0 = torch.tensor([remap[int(v)] for v in f0])
        i1 = torch.tensor([remap[int(v)] for v in f1])
        clip = (1.0 - frac) * gray[i0] + frac * gray[i1]   # [N,H,W] linear interp

        clip = _intensity_norm(clip, cfg)
        if cfg.augment:
            clip = _augment(clip, cfg, rng)
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)  # [N,S,S]
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        clip = clip.unsqueeze(0).repeat(3, 1, 1, 1)  # A.4 replicate -> [3,N,S,S]
        return clip, s

    def __getitem__(self, index):
        cfg = self.cfg
        if self.index is not None:                       # tile: index -> (video, chunk)
            row_idx, chunk = self.index[index]
            row = self.rows[row_idx]
        else:                                            # crop: index -> video
            row = self.rows[index]
        # Per-item RNG: deterministic in eval, varied in train (epoch via torch seed).
        base = self.seed + index
        if cfg.augment:
            base += int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(base)
        try:
            s_tile = self._tile_indices(row, chunk) if self.index is not None else None
            clip, s = self._load_clip(row["path"], rng, s=s_tile)
        except Exception as e:  # noqa: BLE001 -- mirror repo behaviour: resample
            warnings.warn(f"failed to load {row['path']}: {e}")
            return self.__getitem__(int(rng.integers(0, len(self))))

        N = cfg.frames_per_clip
        label = int(row.get(self.label_key, 0) or 0)
        if self.index is not None:                       # one chunk == one clip
            clips = [clip]
            clip_indices = [np.round(s).astype(np.int64)]
        else:
            clips = [clip[:, i * N:(i + 1) * N] for i in range(cfg.num_clips)]
            clip_indices = [
                np.round(s[i * N:(i + 1) * N]).astype(np.int64)
                for i in range(cfg.num_clips)
            ]
        return clips, label, clip_indices
