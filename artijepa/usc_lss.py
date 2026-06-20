"""usc_lss OOD speaker dataset for phoneme prediction (Arti-JEPA eval Task 2).

`/scratch1/hongn/usc_lss`: ONE held-out speaker (usc_s1), 684 utterances, with
**gold ARPABET phonemes + timestamps**. This is out-of-distribution vs the
75-speaker training corpus on three axes -- different speaker, **104x104** video
(vs 84x84), and **99 fps** (vs 83.28) -- so it is a clean generalization test.

The dataloader is the standard Arti-JEPA pipeline (decord linear-interp temporal
resample -> per-clip z-score -> bicubic resize -> grayscale x3), but it also
returns **per temporal-token phoneme labels** time-aligned in seconds (so the 99
fps is irrelevant once resampled to target_fps -- see `phonemes.py`).

Each utterance is tiled into consecutive `frames_per_clip`-frame windows on the
target_fps grid (the short tail is kept and padded; padded tokens are labeled
IGNORE_INDEX), so every annotated frame is covered. Per-utterance predictions are
reassembled in (utt, tile) order for the sequence PER.
"""

import csv
import math
import os

import numpy as np
import torch
from decord import VideoReader, cpu

from artijepa import phonemes as P
from artijepa.rtmri_dataset import _intensity_norm, _spatial, _to_gray, PreprocConfig


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_manifest(root="/scratch1/hongn/usc_lss", out_csv=None, verbose=True):
    """Probe each utterance's video and write a manifest CSV with split labels."""
    out_csv = out_csv or os.path.join(root, "phoneme_manifest.csv")
    split_of = {}
    for split in ("train", "val", "test"):
        sp = os.path.join(root, f"{split}.txt")
        if os.path.exists(sp):
            for line in open(sp):
                uid = line.strip()
                if uid:
                    split_of[uid] = split
    rows = []
    for uid, split in split_of.items():
        vid = os.path.join(root, "video", f"{uid}.avi")
        pj = os.path.join(root, "phonemes", f"{uid}.json")
        wav = os.path.join(root, "audio", f"{uid}.wav")
        if not (os.path.exists(vid) and os.path.exists(pj)):
            if verbose:
                print(f"[usc_lss] skip {uid}: missing video/phonemes")
            continue
        vr = VideoReader(vid, num_threads=1, ctx=cpu(0))
        n = len(vr); fps = float(vr.get_avg_fps())
        rows.append({
            "utt_id": uid, "path": vid, "phoneme_json": pj,
            "audio": wav if os.path.exists(wav) else "",
            "n_frames": n, "fps": fps, "duration_s": n / fps, "split": split,
        })
    rows.sort(key=lambda r: r["utt_id"])
    cols = ["utt_id", "path", "phoneme_json", "audio", "n_frames", "fps",
            "duration_s", "split"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    if verbose:
        from collections import Counter
        print(f"[usc_lss] wrote {len(rows)} utterances -> {out_csv}  "
              f"{dict(Counter(r['split'] for r in rows))}")
    return out_csv


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class USCLSSPhonemeDataset(torch.utils.data.Dataset):
    """Tiled rtMRI clips + per-temporal-token phoneme labels for one OOD speaker."""

    def __init__(self, manifest, split, cfg: PreprocConfig, drop_sil_for_per=True):
        super().__init__()
        assert cfg.tubelet_size and cfg.frames_per_clip % cfg.tubelet_size == 0
        self.cfg = cfg
        self.split = split
        self.drop_sil_for_per = drop_sil_for_per
        with open(manifest) as f:
            self.rows = [r for r in csv.DictReader(f)
                         if (split is None or r.get("split") == split)]
        if not self.rows:
            raise ValueError(f"No usc_lss rows for split={split!r}")
        self.segments = [P.load_gold_segments(r["phoneme_json"]) for r in self.rows]
        self.n_tok = cfg.frames_per_clip // cfg.tubelet_size
        self.index = self._build_tile_index()       # (row, chunk, n_chunks)
        self.num_classes = P.NUM_PHONEMES
        self.collapse_drop = {P.SIL_IDX}             # CTC-collapse: drop silence

    def _n_out(self, row):
        r = float(row["fps"]) / self.cfg.target_fps
        return int(math.floor((int(float(row["n_frames"])) - 1) / r)) + 1

    def _build_tile_index(self):
        F = self.cfg.frames_per_clip
        idx = []
        for ri, row in enumerate(self.rows):
            n_out = self._n_out(row)
            n_chunks = max(1, int(math.ceil(n_out / F)))   # keep+pad the tail
            for c in range(n_chunks):
                idx.append((ri, c, n_chunks))
        return idx

    def __len__(self):
        return len(self.index)

    def _load_clip(self, path, s):
        """Linear temporal interpolation at source positions s (float array)."""
        vr = VideoReader(path, num_threads=2, ctx=cpu(0))
        n_src = len(vr)
        s = np.clip(s, 0.0, n_src - 1.0)
        f0 = np.floor(s).astype(np.int64)
        f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
        need = np.unique(np.concatenate([f0, f1]))
        remap = {int(v): i for i, v in enumerate(need)}
        gray = _to_gray(vr.get_batch(need).asnumpy())          # [K,H,W]
        i0 = torch.tensor([remap[int(v)] for v in f0])
        i1 = torch.tensor([remap[int(v)] for v in f1])
        clip = (1.0 - frac) * gray[i0] + frac * gray[i1]        # [F,H,W]
        cfg = self.cfg
        clip = _intensity_norm(clip, cfg)
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        return clip.unsqueeze(0).repeat(3, 1, 1, 1)             # [3,F,S,S]

    def __getitem__(self, i):
        cfg = self.cfg
        ri, chunk, n_chunks = self.index[i]
        row = self.rows[ri]
        r = float(row["fps"]) / cfg.target_fps
        F = cfg.frames_per_clip
        out_frames = chunk * F + np.arange(F, dtype=np.float64)  # target-grid frames
        s = out_frames * r                                       # -> source positions
        clip = self._load_clip(row["path"], s)
        labels = P.segments_to_token_labels(
            self.segments[ri], self.n_tok, tubelet=cfg.tubelet_size,
            target_fps=cfg.target_fps, clip_start_frame=chunk * F)
        return {
            "clip": clip,                                   # [3,F,S,S]
            "labels": torch.from_numpy(labels),             # [n_tok] (IGNORE outside)
            "utt": ri, "chunk": chunk, "n_chunks": n_chunks,
        }

    def reference_sequences(self):
        """{utt_row_idx: gold phoneme-index sequence} for PER."""
        return {ri: P.reference_sequence(self.segments[ri], drop_sil=self.drop_sil_for_per)
                for ri in range(len(self.rows))}


def collate(batch):
    clips = torch.stack([b["clip"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    meta = [(b["utt"], b["chunk"], b["n_chunks"]) for b in batch]
    return clips, labels, meta
