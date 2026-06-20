"""Pseudo-phoneme labels from the paired audio (Arti-JEPA eval Task 1).

The 75-speaker corpus ships no phoneme annotations, but every mp4 carries an
audio stream (aac @ 22.05 kHz). We derive **pseudo** phoneme labels by running an
audio phoneme-recognition model (a wav2vec2 / WavLM CTC head, e.g. the eSpeak
phoneme model) on that audio, then ask: *can the frozen JEPA video features
predict the audio-derived phoneme stream?* (metrics = Cohen's kappa + PER, in the
audio model's own phone vocabulary; this is a self-consistent transfer probe).

Pipeline (run `build_pseudo_labels` once -> caches per-utterance id streams):
  1. ffmpeg: mp4 audio -> 16 kHz mono float32 (no torchaudio/soundfile needed).
  2. wav2vec2 CTC: 16 kHz wav -> per-~20 ms phoneme-id logits -> argmax stream.
  3. cache the id stream + its frame rate; the dataset aligns it to JEPA tokens
     in SECONDS (so any video fps works -- see `phonemes.py`).

The pseudo label-gen is a separate batch step (model inference over 2,371 clips);
the eval (`eval_phoneme.py --config eval_phoneme_pseudo.yaml`) then reuses the
cached streams just like the gold usc_lss path.
"""

import csv
import json
import math
import os
import shutil
import subprocess

import numpy as np
import torch

from artijepa.rtmri_dataset import _intensity_norm, _spatial, _to_gray, PreprocConfig

# default audio phoneme recogniser (eSpeak IPA phonemes via wav2vec2-large CTC)
DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_LABEL_DIR = "/scratch1/hongn/artijepa/pseudo_phonemes"


def _ffmpeg():
    return shutil.which("ffmpeg") or "/scratch1/hongn/conda/envs/sam_3d_body/bin/ffmpeg"


def extract_audio(path, sr=16000):
    """Decode any media's audio to mono float32 @ sr via ffmpeg (raw f32le pipe)."""
    cmd = [_ffmpeg(), "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# --------------------------------------------------------------------------- #
# CTC pseudo-label generation
# --------------------------------------------------------------------------- #
def load_ctc(model_name=DEFAULT_MODEL, device="cpu"):
    try:
        from transformers import AutoModelForCTC, AutoProcessor
        AutoProcessor  # force the (lazy) processing_auto import chain to resolve now
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not import a transformers CTC model in this env. The `artijepa` "
            "env pins torch 2.6 but ships transformers 5.x, which needs torch>=2.7 "
            f"(import error: {type(e).__name__}: {e}). build_pseudo_labels() is a "
            "DECOUPLED batch step -- run it in an env with a compatible stack "
            "(e.g. `pip install 'transformers<5' torchaudio` alongside torch 2.6, "
            "or any torch>=2.7 env); it only writes .npy label streams + vocab.json, "
            "which the artijepa eval then reads without transformers."
        ) from e
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name).to(device).eval()
    vocab = proc.tokenizer.get_vocab()                       # {phone: id}
    id2phon = {i: p for p, i in vocab.items()}
    blank = getattr(model.config, "pad_token_id", 0) or 0
    return proc, model, id2phon, blank


@torch.no_grad()
def frame_ids(wav, proc, model, device="cpu", sr=16000):
    """Per-frame argmax phoneme ids + the stream's frame rate (Hz)."""
    iv = proc(wav, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    logits = model(iv).logits[0]                             # [Tf, V]
    ids = logits.argmax(-1).cpu().numpy().astype(np.int64)
    fps = len(ids) / (len(wav) / sr)                         # ~49 Hz for wav2vec2
    return ids, float(fps)


def build_pseudo_labels(manifest, out_dir=DEFAULT_LABEL_DIR, model_name=DEFAULT_MODEL,
                        device=None, limit=None):
    """Run the CTC model over every clip in a 75-speaker manifest; cache id streams.

    Writes ``<out_dir>/<stem>.npy`` (int64 frame ids) + a shared ``vocab.json``
    (id2phon, blank id, model). Re-running skips already-cached clips.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    proc, model, id2phon, blank = load_ctc(model_name, device)
    json.dump({"model": model_name, "blank": blank,
               "id2phon": {str(i): p for i, p in id2phon.items()}},
              open(os.path.join(out_dir, "vocab.json"), "w"), indent=1)
    rows = list(csv.DictReader(open(manifest)))
    if limit:
        rows = rows[:limit]
    fps_log = {}
    for n, r in enumerate(rows):
        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        outp = os.path.join(out_dir, stem + ".npy")
        if os.path.exists(outp):
            continue
        try:
            wav = extract_audio(r["path"])
            ids, fps = frame_ids(wav, proc, model, device)
            np.save(outp, ids)
            fps_log[stem] = fps
        except Exception as e:                               # noqa: BLE001
            print(f"[pseudo] FAIL {stem}: {e}")
        if n % 50 == 0:
            print(f"[pseudo] {n}/{len(rows)} (last fps {fps_log.get(stem,'?')})")
    json.dump(fps_log, open(os.path.join(out_dir, "frame_fps.json"), "w"))
    print(f"[pseudo] done -> {out_dir}")


# --------------------------------------------------------------------------- #
# dataset (mirrors usc_lss.USCLSSPhonemeDataset, pseudo labels)
# --------------------------------------------------------------------------- #
class PseudoPhonemeDataset(torch.utils.data.Dataset):
    """75-speaker clips + per-token PSEUDO phoneme labels (audio-model CTC)."""

    def __init__(self, manifest, split, cfg: PreprocConfig,
                 label_dir=DEFAULT_LABEL_DIR):
        super().__init__()
        self.cfg = cfg
        self.label_dir = label_dir
        with open(manifest) as f:
            rows = [r for r in csv.DictReader(f)
                    if (split is None or r.get("split") == split)]
        vj = json.load(open(os.path.join(label_dir, "vocab.json")))
        self.blank = int(vj["blank"])
        self.id2phon = {int(k): v for k, v in vj["id2phon"].items()}
        self.num_classes = max(self.id2phon) + 1
        self.collapse_drop = {self.blank}
        ffps = json.load(open(os.path.join(label_dir, "frame_fps.json")))
        self.rows, self.ids, self.lab_fps = [], [], []
        for r in rows:
            stem = os.path.splitext(os.path.basename(r["path"]))[0]
            p = os.path.join(label_dir, stem + ".npy")
            if not os.path.exists(p):
                continue
            self.rows.append(r)
            self.ids.append(np.load(p))
            self.lab_fps.append(float(ffps.get(stem, 49.0)))
        if not self.rows:
            raise ValueError(f"No cached pseudo labels for split={split!r} in {label_dir}")
        self.n_tok = cfg.frames_per_clip // cfg.tubelet_size
        self.index = self._tile_index()

    def _n_out(self, row):
        r = float(row["fps"]) / self.cfg.target_fps
        return int(math.floor((int(float(row["n_frames"])) - 1) / r)) + 1

    def _tile_index(self):
        F = self.cfg.frames_per_clip
        idx = []
        for ri, row in enumerate(self.rows):
            n_chunks = max(1, int(math.ceil(self._n_out(row) / F)))
            idx += [(ri, c, n_chunks) for c in range(n_chunks)]
        return idx

    def __len__(self):
        return len(self.index)

    def _token_labels(self, ri, chunk):
        """Pseudo phoneme id per temporal token (seconds-aligned to audio stream)."""
        cfg = self.cfg
        ids, afps = self.ids[ri], self.lab_fps[ri]
        from artijepa.phonemes import token_center_times, IGNORE_INDEX
        t = token_center_times(self.n_tok, cfg.tubelet_size, cfg.target_fps,
                               clip_start_frame=chunk * cfg.frames_per_clip)
        fi = np.round(t * afps).astype(np.int64)
        out = np.full(self.n_tok, IGNORE_INDEX, dtype=np.int64)
        ok = (fi >= 0) & (fi < len(ids))
        out[ok] = ids[fi[ok]]
        return out

    def _load_clip(self, path, s):
        from decord import VideoReader, cpu
        vr = VideoReader(path, num_threads=2, ctx=cpu(0))
        n_src = len(vr); s = np.clip(s, 0.0, n_src - 1.0)
        f0 = np.floor(s).astype(np.int64); f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
        need = np.unique(np.concatenate([f0, f1]))
        remap = {int(v): i for i, v in enumerate(need)}
        gray = _to_gray(vr.get_batch(need).asnumpy())
        i0 = torch.tensor([remap[int(v)] for v in f0])
        i1 = torch.tensor([remap[int(v)] for v in f1])
        clip = (1.0 - frac) * gray[i0] + frac * gray[i1]
        cfg = self.cfg
        clip = _intensity_norm(clip, cfg)
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        return clip.unsqueeze(0).repeat(3, 1, 1, 1)

    def __getitem__(self, i):
        cfg = self.cfg
        ri, chunk, n_chunks = self.index[i]
        row = self.rows[ri]
        r = float(row["fps"]) / cfg.target_fps
        F = cfg.frames_per_clip
        s = (chunk * F + np.arange(F, dtype=np.float64)) * r
        return {
            "clip": self._load_clip(row["path"], s),
            "labels": torch.from_numpy(self._token_labels(ri, chunk)),
            "utt": ri, "chunk": chunk, "n_chunks": n_chunks,
        }

    def reference_sequences(self):
        """CTC-collapsed pseudo sequence per utterance (drop blank)."""
        from artijepa.phonemes import collapse_sequence
        return {ri: collapse_sequence(self.ids[ri], drop=tuple(self.collapse_drop))
                for ri in range(len(self.rows))}


from artijepa.usc_lss import collate  # noqa: E402  (shared dict collate)
