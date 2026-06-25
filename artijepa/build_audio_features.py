"""Offline audio-feature cache for AC-JEPA-audio (plans_aucjepa.md §5).

For each clip we decode its paired audio (``extract_audio`` -> 16 kHz mono) and
run a frozen HF speech encoder (WavLM / Wav2Vec2), caching the per-frame last
(or chosen) hidden state ``[T_audio, A]`` to ``.npy`` (fp16). A shared
``meta.json`` records the model, layer, dim ``A``, the audio frame rate, and the
**per-dim corpus z-score stats** (mean/std over the TRAIN split) that
``audio_cond.normalize_audio`` uses before forming the state/action tokens.

This is a **decoupled batch step** (mirrors ``audio_phoneme.build_pseudo_labels``)
because the ``artijepa`` env pins torch 2.6 while its transformers (5.x) needs
torch>=2.7 and hangs on import. Run it in a transformers-compatible env --
``his-extract`` (torch 2.6+cu124 + transformers 4.56.2) drives the P100/V100 AND
loads WavLM:

    conda activate /scratch1/hongn/conda/envs/his-extract
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.build_audio_features \
        --manifest /scratch1/hongn/artijepa/manifest_alltrain.csv \
        --out /scratch1/hongn/artijepa/audio_feats/wavlm_base_plus \
        --model microsoft/wavlm-base-plus --layer -1 --limit 20      # smoke

Training (``aucjepa_train.py``) reads only the ``.npy`` + ``meta.json`` (no
transformers import), so the cache is env-decoupled and deterministic.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from artijepa.audio_phoneme import extract_audio

DEFAULT_MODEL = "microsoft/wavlm-base-plus"
DEFAULT_OUT = "/scratch1/hongn/artijepa/audio_feats/wavlm_base_plus"
SR = 16000


def load_audio_model(model_name=DEFAULT_MODEL, device="cpu"):
    """Frozen HF speech encoder + matching feature extractor (eval, no grad)."""
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not import transformers in this env. build_audio_features() is a "
            "DECOUPLED batch step -- run it in a transformers-compatible env (e.g. "
            "`his-extract`: torch 2.6 + transformers 4.56). The `artijepa` env's "
            f"transformers (5.x) needs torch>=2.7 and hangs ({type(e).__name__}: {e})."
        ) from e
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return fe, model


@torch.no_grad()
def audio_hidden(wav, fe, model, layer=-1, device="cpu"):
    """Per-frame hidden state of ``layer`` for one waveform -> ([T_audio, A], rate_hz)."""
    iv = fe(wav, sampling_rate=SR, return_tensors="pt")
    inp = {k: v.to(device) for k, v in iv.items()}
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[layer][0]                     # [T_audio, A]
    rate = h.shape[0] / (len(wav) / SR)                 # ~49.95 Hz for 16 kHz/320x
    return h.float().cpu().numpy(), float(rate)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_audio_features(manifest, out_dir=DEFAULT_OUT, model_name=DEFAULT_MODEL,
                         layer=-1, device=None, limit=None, stats_split="train"):
    """Cache per-clip audio hidden states + write meta.json (with corpus z-score stats).

    Pass 1 extracts features (skipping already-cached clips); pass 2 streams the
    ``stats_split`` clips to accumulate per-dim mean/std. Re-running is safe and
    recomputes stats from whatever is cached.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    fe, model = load_audio_model(model_name, device)
    rows = list(csv.DictReader(open(manifest)))
    if limit:
        rows = rows[:limit]

    # -- pass 1: extract + cache
    rates, dim, n_done, n_fail = [], None, 0, 0
    for n, r in enumerate(rows):
        outp = os.path.join(out_dir, _stem(r["path"]) + ".npy")
        if os.path.exists(outp):
            if dim is None:
                dim = int(np.load(outp, mmap_mode="r").shape[1])
            continue
        try:
            wav = extract_audio(r["path"], sr=SR)
            feats, rate = audio_hidden(wav, fe, model, layer, device)
            np.save(outp, feats.astype(np.float16))
            rates.append(rate)
            dim = feats.shape[1]
            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[audio] FAIL {_stem(r['path'])}: {type(e).__name__}: {e}")
        if n % 50 == 0:
            print(f"[audio] {n}/{len(rows)} (new {n_done}, fail {n_fail}, "
                  f"last rate {rates[-1] if rates else '?'})")

    # -- pass 2: per-dim z-score stats over the stats_split (streamed, float64)
    has_split = rows and ("split" in rows[0])
    stat_rows = [r for r in rows if (not has_split or r.get("split") == stats_split)]
    s = np.zeros(dim, np.float64); ss = np.zeros(dim, np.float64); cnt = 0
    sample_rates = []
    for r in stat_rows:
        p = os.path.join(out_dir, _stem(r["path"]) + ".npy")
        if not os.path.exists(p):
            continue
        f = np.load(p).astype(np.float64)               # [T_audio, A]
        s += f.sum(0); ss += (f * f).sum(0); cnt += f.shape[0]
        # derive exact per-clip rate from duration when available
        dur = r.get("duration_s")
        if dur:
            sample_rates.append(f.shape[0] / float(dur))
    mean = (s / max(1, cnt))
    var = np.maximum(ss / max(1, cnt) - mean ** 2, 1e-8)
    std = np.sqrt(var)

    rate_hz = float(np.median(sample_rates)) if sample_rates else \
        (float(np.median(rates)) if rates else SR / 320.0)
    meta = {
        "model": model_name, "layer": layer, "dim": int(dim), "sr": SR,
        "audio_rate_hz": rate_hz,         # nominal; dataset derives exact rate per clip
        "stats_split": stats_split, "stats_n_frames": int(cnt),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[audio] done -> {out_dir}  (dim={dim}, rate~{rate_hz:.3f} Hz, "
          f"stats over {cnt} frames from {len(stat_rows)} {stats_split} clips)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/scratch1/hongn/artijepa/manifest_alltrain.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats-split", default="train")
    args = ap.parse_args()
    build_audio_features(args.manifest, args.out, args.model, args.layer,
                         args.device, args.limit, args.stats_split)


if __name__ == "__main__":
    main()
