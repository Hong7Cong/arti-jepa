"""rtMRI clips + frame-aligned arti-6 for AC-JEPA (aucjepa_plans_new.md §0.5, §1.2).

``RTMRIArtiDataset`` extends ``RTMRIVideoDataset`` (tile sampling) but the frames and
the articulators come from the SAME cached session (``arti_cache.py``), so they are
frame-exact by construction -- the alignment risk of the plan (§1.1) is gone. Each
item is ``{clip [3,T,H,W], arti [T',6], valid [T']}``:

  * clip  -- MRI frames read from ``<stem>.image.npy`` (mmap), linearly resampled onto
    the ``target_fps`` grid and run through the standard rtMRI preprocessing
    (percentile clip + per-clip z-score -> bicubic resize -> grayscale x3).
  * arti  -- ``<stem>.arti.npy`` (the 6 constriction signals @100 Hz) average-pooled
    onto the encoder's ``T'`` temporal tokens (``arti_cond.pool_arti_to_tokens``) and
    per-dim z-scored with the cache's corpus stats.

Tile-only: each (session, chunk) is one item; the chunk's token windows align to the
full-session arti via ``clip_start_frame = chunk * frames_per_clip``.
"""

import json
import os

import numpy as np
import torch

from artijepa.arti_cond import normalize_arti, pool_arti_to_tokens
from artijepa.rtmri_dataset import (RTMRIVideoDataset, _intensity_norm, _spatial)


class RTMRIArtiDataset(RTMRIVideoDataset):
    def __init__(self, manifest, split, cfg, cache_dir=None, seed=0, normalize="zscore"):
        assert cfg.sampling == "tile", "RTMRIArtiDataset requires sampling='tile'"
        super().__init__(manifest, split, cfg, seed=seed)
        self.normalize = normalize
        # stats live next to the .npy cache; default to the manifest's directory's
        # sibling if not given (meta.json sits in arti_feats/<set>).
        meta_path = os.path.join(cache_dir, "meta.json") if cache_dir else None
        if meta_path is None or not os.path.exists(meta_path):
            # fall back: meta.json beside the first arti_npy
            meta_path = os.path.join(os.path.dirname(self.rows[0]["arti_npy"]), "meta.json")
        meta = json.load(open(meta_path))
        self.A = int(meta["dim"])
        self.a_mean = np.asarray(meta["mean"], np.float32)
        self.a_std = np.asarray(meta["std"], np.float32)
        self.nominal_rate = float(meta.get("arti_rate_hz", 100.0))
        self.n_tok = cfg.frames_per_clip // cfg.tubelet_size

        # keep only rows whose arti (and image) npy are present, then rebuild the
        # tile index against the filtered rows (parent's index holds row indices).
        kept = []
        for row in self.rows:
            ap = row.get("arti_npy")
            ip = row.get("image_npy")
            if ap and os.path.exists(ap) and ip and os.path.exists(ip):
                kept.append(row)
        if not kept:
            raise ValueError(f"No cached arti/image rows for split={split!r}")
        self.rows = kept
        self.index = self._build_tile_index()

    # -- video frames from the cached IMAGE .npy (mmap; linear temporal interp) ---
    def _load_clip_npy(self, image_npy, s):
        cfg = self.cfg
        frames = np.load(image_npy, mmap_mode="r")            # [T,H,W] uint8
        n_src = frames.shape[0]
        s = np.clip(s, 0.0, n_src - 1.0)
        f0 = np.floor(s).astype(np.int64)
        f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
        need = np.unique(np.concatenate([f0, f1]))
        remap = {int(v): i for i, v in enumerate(need)}
        gray = torch.from_numpy(np.asarray(frames[need], dtype=np.float32))   # [K,H,W]
        i0 = torch.tensor([remap[int(v)] for v in f0])
        i1 = torch.tensor([remap[int(v)] for v in f1])
        clip = (1.0 - frac) * gray[i0] + frac * gray[i1]      # [N,H,W] linear interp
        clip = _intensity_norm(clip, cfg)
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        return clip.unsqueeze(0).repeat(3, 1, 1, 1)           # [3,N,S,S]

    def _pooled_arti(self, row, chunk):
        cfg = self.cfg
        feats = np.load(row["arti_npy"]).astype(np.float32)   # [T,6]
        dur = float(row.get("duration_s") or 0.0) or None
        rate = (feats.shape[0] / dur) if dur else self.nominal_rate
        e, valid = pool_arti_to_tokens(
            feats, rate, self.n_tok, cfg.tubelet_size, cfg.target_fps,
            clip_start_frame=chunk * cfg.frames_per_clip)
        if self.normalize == "zscore":
            e = normalize_arti(e, self.a_mean, self.a_std)
        return e.astype(np.float32), valid

    def __getitem__(self, i):
        cfg = self.cfg
        row_idx, chunk = self.index[i]
        row = self.rows[row_idx]
        base = self.seed + i
        if cfg.augment:
            base += int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(base)
        try:
            s_tile = self._tile_indices(row, chunk)
            clip = self._load_clip_npy(row["image_npy"], s_tile)
            e, valid = self._pooled_arti(row, chunk)
        except Exception as ex:  # noqa: BLE001 -- mirror parent: resample on failure
            import warnings
            warnings.warn(f"failed item {i} ({row.get('stem')}): {ex}")
            return self.__getitem__(int(rng.integers(0, len(self))))
        return {
            "clip": clip,                                     # [3,T,H,W]
            "arti": torch.from_numpy(e),                      # [T',6]
            "valid": torch.from_numpy(valid),                 # [T'] bool
        }


def collate(batch):
    """Stack dict items -> (clips [B,3,T,H,W], arti [B,T',6], valid [B,T'])."""
    clips = torch.stack([b["clip"] for b in batch], 0)
    arti = torch.stack([b["arti"] for b in batch], 0)
    valid = torch.stack([b["valid"] for b in batch], 0)
    return clips, arti, valid
