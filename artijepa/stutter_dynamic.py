"""Dynamic-length rtMRI window dataloader for the binary stutter task.

Unlike ``stutter_binary`` (fixed ``frames_per_clip`` -> one ``[3,F,S,S]`` clip), this
samples each event window at a **target FPS** (or the native ~99 fps) so the number
of frames is **proportional to the event's duration**, then tiles them into fixed
``window``-frame sub-clips that are **in-distribution** for the 32f-pretrained
encoder. One event -> ``[K, 3, window, S, S]`` with ``K`` varying per event.

Downstream (``eval_stutter_binary_dynamic``) each window is mean-pooled over its S'
spatial tokens to one vector per frame, and the K windows are concatenated into a
**variable-length temporal sequence** ``[L=K*T', D]`` consumed by a sequence model
(masked attentive pool / packed LSTM). Keeping the pooled-spatial step means the
feature cache stays tiny (a few hundred KB per clip) regardless of duration.

This module does **not** modify the fixed-length ``stutter_binary`` path; it reuses
its preprocessing (``_preproc``) and OpenCV (pal8-safe) decode approach.
"""

import cv2
import numpy as np
import torch

from artijepa import stutter as S
from artijepa import stutter_binary as SB
from artijepa.rtmri_dataset import _intensity_norm, _spatial, _to_gray

NATIVE_FPS = 99.0


def n_windows(dur, sample_fps, window, native_fps=NATIVE_FPS):
    """# of ``window``-frame sub-clips for a ``dur``-second event at ``sample_fps``.

    ``sample_fps=None``/``"native"`` uses the video's native rate. The frame budget
    ``round(dur*fps)`` is floored at one window and rounded up to a whole window.
    """
    fps_s = native_fps if sample_fps in (None, "native") else float(sample_fps)
    nf = max(window, int(round(dur * fps_s)))
    return (nf + window - 1) // window


def _load_windows_cv2(path, t0, t1, cfg, sample_fps, window, native_fps=NATIVE_FPS):
    """Decode ``[t0,t1]`` (s) into ``K`` tiled ``window``-frame clips -> [K,3,window,S,S].

    Samples ``K*window`` frames uniformly across the (padded) window -- so the density
    is ~``sample_fps`` -- with linear interpolation between bracketing native frames,
    then the standard Arti-JEPA preprocessing (percentile clip -> z-score/minmax ->
    resize -> grayscale x3). Reshaping the frame axis into ``[K, window]`` yields K
    temporally-contiguous 32f sub-clips. OpenCV decode (decord reads these pal8 avis
    as black frames -- see docs/STUTTERING.md).
    """
    dur = max(t1 - t0, 1e-3)
    fps_s = native_fps if sample_fps in (None, "native") else float(sample_fps)
    K = max(1, (max(window, int(round(dur * fps_s))) + window - 1) // window)
    nframes = K * window

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = float(cap.get(cv2.CAP_PROP_FPS)) or float(cfg.target_fps)
    times = np.linspace(t0, t1, nframes, dtype=np.float64)
    s = np.clip(times * vfps, 0.0, n_src - 1.0)
    f0 = np.floor(s).astype(np.int64)
    f1 = np.minimum(f0 + 1, n_src - 1)
    frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
    need = np.unique(np.concatenate([f0, f1]))
    lo, hi = int(need[0]), int(need[-1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    want = set(int(v) for v in need)
    got = {}
    idx = lo
    while idx <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            got[idx] = fr
        idx += 1
    cap.release()
    if not got:
        raise RuntimeError(f"cv2 read no frames from {path} [{lo},{hi}]")
    last = got[min(got)]
    stack = np.stack([got.get(int(v), last) for v in need], 0)     # [M,H,W,3]

    gray = _to_gray(stack)                                          # [M,H,W]
    remap = {int(v): i for i, v in enumerate(need)}
    i0 = torch.tensor([remap[int(v)] for v in f0])
    i1 = torch.tensor([remap[int(v)] for v in f1])
    clip = (1.0 - frac) * gray[i0] + frac * gray[i1]               # [nframes,H,W]
    clip = _intensity_norm(clip, cfg)
    clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)      # [nframes,Sz,Sz]
    clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
    Sz = cfg.spatial_size
    clip = clip.reshape(K, window, Sz, Sz)                         # [K,window,Sz,Sz]
    clip = clip.unsqueeze(1).repeat(1, 3, 1, 1, 1)                 # [K,3,window,Sz,Sz]
    return clip, K


class DynamicWindowDataset(torch.utils.data.Dataset):
    """One event -> ``K`` tiled ``window``-frame clips (K ~ duration * sample_fps).

    ``seq_len[i] = K_i * (window // tubelet)`` is the eventual temporal-sequence length
    after spatial pooling; it is computed from the row durations at init (no decode) so
    the ragged feature cache can be pre-sized.
    """

    def __init__(self, rows, cfg, sample_fps, window=32, event_pad_s=0.0,
                 classes=("fluent", "disfluent")):
        assert window % cfg.tubelet_size == 0, "window must be divisible by tubelet_size"
        self.cfg = cfg
        self.sample_fps = sample_fps
        self.window = int(window)
        self.pad = float(event_pad_s)
        self.classes = list(classes)
        self.Tp = self.window // cfg.tubelet_size
        self.rows, self.labels, self.n_win, self.seq_len = [], [], [], []
        for r in rows:
            y = S.row_label(r, "binary", self.classes)
            if y is None:
                continue
            t0, t1 = self._window(r)
            k = n_windows(t1 - t0, sample_fps, self.window)
            self.rows.append(r); self.labels.append(y)
            self.n_win.append(k); self.seq_len.append(k * self.Tp)
        if not self.rows:
            raise ValueError("no binary rows for dynamic dataset")
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def _window(self, r):
        t0 = max(0.0, float(r["xmin"]) - self.pad)
        t1 = max(t0 + 1e-3, float(r["xmax"]) + self.pad)
        return t0, t1

    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        return np.bincount(self.labels, minlength=len(self.classes))

    def __getitem__(self, i):
        r = self.rows[i]
        t0, t1 = self._window(r)
        clips, k = _load_windows_cv2(r["path"], t0, t1, self.cfg,
                                     self.sample_fps, self.window)
        return {"clips": clips, "n_win": int(k), "label": int(self.labels[i]),
                "seg": int(r["seg_id"]), "speaker": r["speaker"]}


def collate_windows(batch):
    """Extraction collate: keep the batch as a list (K varies per clip)."""
    return batch


def make_dataset(rows, sample_fps, window=32, spatial_size=256, spatial_mode="resize",
                 intensity_norm="zscore", grayscale_stats=SB.GRAYSCALE_STATS,
                 tubelet_size=2, event_pad_s=0.0, classes=("fluent", "disfluent")):
    """DynamicWindowDataset with the standard binary preprocessing (reuses SB._preproc)."""
    cfg = SB._preproc(window, spatial_size, spatial_mode, intensity_norm,
                      grayscale_stats, tubelet_size)
    return DynamicWindowDataset(rows, cfg, sample_fps, window=window,
                                event_pad_s=event_pad_s, classes=classes)
