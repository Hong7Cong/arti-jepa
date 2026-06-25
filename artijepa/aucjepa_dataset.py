"""rtMRI clips + aligned cached audio embeddings for AC-JEPA-audio (plans §5).

``RTMRIAudioDataset`` extends ``RTMRIVideoDataset`` (tile sampling) so each item is
``{clip [3,T,H,W], audio [T',A], valid [T']}``: the video is loaded by the parent's
machinery; the audio is the offline WavLM cache (``build_audio_features.py``)
pooled onto the encoder's ``T'`` temporal tokens (``audio_cond.pool_audio_to_tokens``)
and per-dim z-scored with the cache's corpus stats.

Tile-only: each (video, chunk) is one item; the chunk's token windows align to the
full-video audio via ``clip_start_frame = chunk * frames_per_clip`` -- the exact
alignment used by ``audio_phoneme.PseudoPhonemeDataset`` and the phoneme eval.
"""

import csv
import json
import os

import numpy as np
import torch

from artijepa.audio_cond import normalize_audio, pool_audio_to_tokens
from artijepa.rtmri_dataset import RTMRIVideoDataset


class RTMRIAudioDataset(RTMRIVideoDataset):
    def __init__(self, manifest, split, cfg, audio_dir, seed=0, normalize="zscore"):
        assert cfg.sampling == "tile", "RTMRIAudioDataset requires sampling='tile'"
        super().__init__(manifest, split, cfg, seed=seed)
        self.audio_dir = audio_dir
        self.normalize = normalize
        meta = json.load(open(os.path.join(audio_dir, "meta.json")))
        self.A = int(meta["dim"])
        self.a_mean = np.asarray(meta["mean"], np.float32)
        self.a_std = np.asarray(meta["std"], np.float32)
        self.nominal_rate = float(meta.get("audio_rate_hz", meta["sr"] / 320.0))
        self.n_tok = cfg.frames_per_clip // cfg.tubelet_size

        # keep only rows whose audio is cached, then rebuild the tile index against
        # the filtered rows (the parent's index holds indices into self.rows).
        kept, self.audio_paths, self.durations = [], [], []
        for row in self.rows:
            p = os.path.join(audio_dir, os.path.splitext(os.path.basename(row["path"]))[0] + ".npy")
            if os.path.exists(p):
                kept.append(row)
                self.audio_paths.append(p)
                self.durations.append(float(row.get("duration_s") or 0.0) or None)
        if not kept:
            raise ValueError(f"No cached audio in {audio_dir} for split={split!r}")
        self.rows = kept
        self.index = self._build_tile_index()

    def _audio_rate(self, row_idx, T_audio):
        dur = self.durations[row_idx]
        return (T_audio / dur) if dur else self.nominal_rate

    def _pooled_audio(self, row_idx, chunk):
        cfg = self.cfg
        feats = np.load(self.audio_paths[row_idx])                # [T_audio, A] fp16
        rate = self._audio_rate(row_idx, feats.shape[0])
        e, valid = pool_audio_to_tokens(
            feats, rate, self.n_tok, cfg.tubelet_size, cfg.target_fps,
            clip_start_frame=chunk * cfg.frames_per_clip)
        if self.normalize == "zscore":
            e = normalize_audio(e, self.a_mean, self.a_std)
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
            clip, _ = self._load_clip(row["path"], rng, s=s_tile)
            e, valid = self._pooled_audio(row_idx, chunk)
        except Exception as ex:  # noqa: BLE001 -- mirror parent: resample on failure
            import warnings
            warnings.warn(f"failed item {i} ({row['path']}): {ex}")
            return self.__getitem__(int(rng.integers(0, len(self))))
        return {
            "clip": clip,                                          # [3,T,H,W]
            "audio": torch.from_numpy(e),                          # [T',A]
            "valid": torch.from_numpy(valid),                      # [T'] bool
        }


def collate(batch):
    """Stack dict items -> (clips [B,3,T,H,W], audio [B,T',A], valid [B,T'])."""
    clips = torch.stack([b["clip"] for b in batch], 0)
    audio = torch.stack([b["audio"] for b in batch], 0)
    valid = torch.stack([b["valid"] for b in batch], 0)
    return clips, audio, valid
