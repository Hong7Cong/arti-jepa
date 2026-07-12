"""Frozen T-SSL / V-JEPA binary FLUENT-vs-DISFLUENT classification (Task 8b).

Trains + evaluates a small probe on top of a **frozen** rtMRI encoder (the 256px
T-SSL V-JEPA2 checkpoint, loaded exactly as in ``examples/demo.ipynb``) for the
binary "regular speech vs disfluency" task.

Why a separate script from ``eval_disfluency.py``: that eval builds its clips with
``stutter.DisfluencySegmentDataset``, which decodes the stuttering ``.avi`` with
**decord** -- and every one of those files is ``rawvideo``/``pix_fmt=pal8``, which
decord silently reads as **all-zero (black) frames** (see ``docs/STUTTERING.md`` §
"Decoder bug"). Any feature/eval decord produces on this corpus is therefore invalid.
This script instead uses ``stutter_binary`` (§7 of the doc): OpenCV decoding
(pal8-safe) **and** fluent negatives whose duration distribution is matched to the
positives, so clip length is not a give-away feature.

Pipeline (mirrors ``eval_disfluency`` frozen mode, reusing its probe code):
  1. Build binary rows (pos = disfluency events, neg = duration-matched fluent
     windows) with ``stutter_binary.build_rows``.
  2. Extract one pooled feature per clip **once** with the frozen encoder and cache
     it; leave-one-speaker-out (LOSO) folds reuse the single cache.
  3. Train an attentive/mean/mlp probe (inverse-freq class weighting), model-select
     on a stratified val split by macro-F1, report on the held-out speaker(s).

Geometry note: the checkpoint was pretrained at **256px / 32 frames / tubelet 2 /
patch 16**. We feed clips at that exact geometry (``frames_per_clip`` uniformly
sampled across each event window at the video's native ~99 fps) so the encoder runs
in-distribution -- the ~4096-token forward from ``demo.ipynb``, not the 200-frame /
25k-token loader default (which would be far OOD and ~memory-prohibitive on ViT-L).

Metrics: macro-F1 (primary), balanced accuracy, accuracy, per-class P/R/F1, and the
confusion matrix -- per held-out speaker and pooled over folds.

Run:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_stutter_binary \
        --config dev_artiJEPA/configs/eval_stutter_binary.yaml
    # override the checkpoint / split from the CLI:
    ... --checkpoint /data1/hongn/arti-jepa/tssl_vitl_256_combined/ckpt_100.pt --tag tssl256
    ... --split loso            # (default) leave-one-speaker-out over the 7 PWS
    ... --split fixed --test-speaker PWS10 --val-speaker PWS7
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml

from artijepa import stutter as S
from artijepa import stutter_binary as SB
from artijepa.checkpoint import filtered_load
from artijepa.model import build_models
# Reuse the exact probe / training machinery from the disfluency-type eval so the
# two evals stay byte-identical where they overlap (probe, class weights, folds).
from artijepa.eval_disfluency import train_probe, _val_split

BINARY_CLASSES = ["fluent", "disfluent"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# resource capping (1 GPU, bounded CPU cores)
# --------------------------------------------------------------------------- #
# GPU: the whole script only ever uses `cuda:0`; export CUDA_VISIBLE_DEVICES=0 in
# the launcher to make that a hard guarantee on a multi-GPU box.
# CPU: total cores ~= num_workers (video-decode processes, 1 thread each) + the
# main process's `cpu_threads`. We pin every library's internal thread pool so a
# worker can't silently fan out to all cores (OpenCV decode + numpy/torch resize).
def _pin_threads(n):
    """Cap this process's OpenCV / numpy / torch intra-op threads to ``n``."""
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        import cv2
        cv2.setNumThreads(1)                 # decode is sequential per clip anyway
    except Exception:
        pass


def _worker_init(_wid):
    """DataLoader worker: 1 OpenCV + 1 torch thread so N workers ~= N cores."""
    _pin_threads(1)


# --------------------------------------------------------------------------- #
# frozen encoder (T-SSL / V-JEPA2 256px checkpoint) -- loaded like demo.ipynb
# --------------------------------------------------------------------------- #
def load_frozen_encoder(cfg, device):
    """Build the ViT-L encoder at the checkpoint's geometry and load EMA weights.

    Mirrors ``examples/demo.ipynb``: build at 256px / 32f / tubelet 2 / patch 16,
    then ``filtered_load`` the ``target_encoder`` (EMA) state -- its keys are already
    ``backbone.*`` so they map onto the ``MultiSeqWrapper`` directly.
    """
    d, ec = cfg["data"], cfg["encoder"]
    encoder, _ = build_models(
        device=device, model_name=ec.get("model_name", "vit_large"),
        spatial_size=d["spatial_size"], frames_per_clip=d["frames_per_clip"],
        patch_size=d.get("patch_size", 16), tubelet_size=d.get("tubelet_size", 2),
        num_mask_tokens=1, use_activation_checkpointing=False)
    ckpt = torch.load(ec["checkpoint"], map_location="cpu", weights_only=False)
    key = ec.get("key", "target_encoder")
    if key not in ckpt:
        key = next(k for k in ("target_encoder", "encoder", "ema_encoder") if k in ckpt)
    n, miss, skip = filtered_load(encoder, ckpt[key])
    print(f"[bin-eval] encoder<-{os.path.basename(ec['checkpoint'])}:{key} "
          f"loaded {n} miss {len(miss)} skip {len(skip)} (epoch {ckpt.get('epoch','?')})")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


# --------------------------------------------------------------------------- #
# rows + per-clip feature extraction (cached; LOSO folds share one cache)
# --------------------------------------------------------------------------- #
def build_rows(cfg):
    d = cfg["data"]
    rows, stats = SB.build_rows(
        root=d.get("root", SB.ROOT), tiers=tuple(d.get("tiers", ["disfluency"])),
        neg_per_pos=d.get("neg_per_pos", 1.0), seed=d.get("build_seed", 0),
        min_dur=d.get("min_dur", 0.20), max_dur=d.get("max_dur", 8.0),
        merge_gap=d.get("merge_gap", 0.25))
    return rows, stats


def _tag(cfg):
    """A content hash so different geometries / row-builds get distinct caches."""
    d, ec = cfg["data"], cfg["encoder"]
    hd = {"ckpt": ec["checkpoint"], "key": ec.get("key", "target_encoder"),
          "sz": d["spatial_size"], "fpc": d["frames_per_clip"],
          "tub": d.get("tubelet_size", 2), "pad": d.get("event_pad_s", 0.0),
          "pool_spatial": d.get("pool_spatial", False),
          "neg_per_pos": d.get("neg_per_pos", 1.0), "build_seed": d.get("build_seed", 0),
          "tiers": d.get("tiers", ["disfluency"]), "min_dur": d.get("min_dur", 0.20),
          "max_dur": d.get("max_dur", 8.0), "merge_gap": d.get("merge_gap", 0.25)}
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or os.path.basename(os.path.dirname(ec["checkpoint"]))
    return f"{tag}_{h}"


@torch.no_grad()
def extract(encoder, cfg, rows, device, dtype):
    """One pooled feature per clip -> (feats [N,L,D] or [N,D], labels, speakers).

    ``pool_spatial=False`` keeps the temporal-major token grid ``[T'*S', D]`` for the
    attentive probe; ``True`` mean-pools every token to ``[D]`` for the mean/mlp probe.
    Cached under ``meta.cache_dir/<tag>`` so LOSO reuses the single extraction.
    """
    d = cfg["data"]
    name = _tag(cfg)
    cdir = os.path.join(cfg["meta"]["cache_dir"], name); os.makedirs(cdir, exist_ok=True)
    fp, yp, kp = (os.path.join(cdir, f"all.{x}.npy") for x in ("feats", "label", "spk"))
    if all(os.path.exists(x) for x in (fp, yp, kp)):
        print(f"[bin-eval] cache hit <- {cdir}")
        return np.load(fp, mmap_mode="r"), np.load(yp), list(np.load(kp))

    # Build the dataset via SB, but the loader by hand so we can pin per-worker
    # threads (SB.make_loader has no worker_init_fn hook) -> bounded CPU usage.
    ds = SB.make_dataset(
        rows, num_frames=d["frames_per_clip"], spatial_size=d["spatial_size"],
        spatial_mode=d.get("spatial_mode", "resize"),
        intensity_norm=d.get("intensity_norm", "zscore"),
        grayscale_stats=d.get("grayscale_stats", SB.GRAYSCALE_STATS),
        tubelet_size=d.get("tubelet_size", 2), event_pad_s=d.get("event_pad_s", 0.0))
    nw = d.get("num_workers", 6)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=d.get("batch_size", 8), shuffle=False, num_workers=nw,
        collate_fn=S.collate, worker_init_fn=_worker_init if nw > 0 else None,
        pin_memory=(device.type == "cuda"))
    print(f"[bin-eval] dataset={len(ds)} class_counts(fluent,disfluent)="
          f"{ds.class_counts().tolist()} (num_workers={nw})")

    tub = d.get("tubelet_size", 2)
    Tp = d["frames_per_clip"] // tub
    pool_spatial = d.get("pool_spatial", False)
    feats = None; labels = []; spks = []; pos = 0; t0 = time.time(); N = len(ds)
    for bi, (clips, y, meta) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            tok = encoder.backbone(clips)                    # [B, T'*S', D]
        B, Ntot, D = tok.shape
        tok = tok.float().reshape(B, Tp, Ntot // Tp, D)      # [B,T',S',D]
        v = tok.mean((1, 2)) if pool_spatial else tok.reshape(B, -1, D)
        v = v.cpu().numpy().astype(np.float16)
        if feats is None:
            feats = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16,
                                              shape=(N,) + v.shape[1:])
        feats[pos:pos + B] = v
        labels += y.tolist(); spks += [m[1] for m in meta]; pos += B
        if bi % 20 == 0:
            print(f"[bin-eval]  extract {pos}/{N} ({time.time()-t0:.0f}s)")
    feats.flush()
    labels = np.asarray(labels, dtype=np.int64)
    np.save(yp, labels); np.save(kp, np.asarray(spks))
    print(f"[bin-eval] extracted {feats.shape} in {time.time()-t0:.0f}s -> {cdir}")
    return np.load(fp, mmap_mode="r"), labels, spks


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    _pin_threads(cfg["data"].get("cpu_threads", 2))     # bound main-process cores
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "bfloat16").lower(), torch.float32)
    classes = BINARY_CLASSES
    # attentive probe pools the raw token grid; mean/mlp use the pre-pooled vector.
    # mean/mlp consume a pre-pooled [D] vector; attentive & attentive_lstm keep the
    # token grid (attentive_lstm pools the S' spatial tokens itself, per frame).
    cfg["data"]["pool_spatial"] = cfg["probe"].get("type", "attentive") in ("mean", "mlp")
    print(f"[bin-eval] frozen binary | classes={classes} "
          f"probe={cfg['probe'].get('type','attentive')} "
          f"pool_spatial={cfg['data']['pool_spatial']} device={device}")

    gs = cfg["data"].get("grayscale_stats")
    if gs and os.path.exists(gs):
        print(f"[bin-eval] grayscale stats <- {gs}")
    else:
        print(f"[bin-eval] grayscale stats {gs!r} absent -> global channel-norm "
              f"defaults to mean=0/std=1 (per-clip z-score still applied)")

    rows, row_stats = build_rows(cfg)
    encoder = load_frozen_encoder(cfg, device)
    feats, y, speakers = extract(encoder, cfg, rows, device, dtype)
    del encoder; torch.cuda.empty_cache()

    y = np.asarray(y, dtype=np.int64)
    spk = np.asarray(speakers)
    keep = np.arange(len(y), dtype=np.int64)          # every clip has a binary label
    uniq_spk = sorted(set(spk.tolist()))
    print(f"[bin-eval] {len(keep)} clips; per-class(fluent,disfluent)="
          f"{np.bincount(y, minlength=2).tolist()}; speakers {uniq_spk}")

    split_mode = cfg["data"].get("split_mode", "loso")
    val_frac = cfg["data"].get("val_frac", 0.15)
    folds, all_true, all_pred, all_spk = [], [], [], []
    yfull = y                                          # feats rows align 1:1 with y

    if split_mode == "loso":
        for test_spk in uniq_spk:
            te = keep[spk == test_spk]
            tr_all = keep[spk != test_spk]
            if len(np.unique(yfull[tr_all])) < 2 or len(te) == 0:
                print(f"[bin-eval] skip fold {test_spk}: degenerate")
                continue
            tr, va = _val_split(tr_all, yfull, len(classes), val_frac, rng)
            tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, test_spk,
                                        device, classes)
            folds.append({"speaker": test_spk, "n_test": int(len(te)),
                          "val_f1": round(vf1, 4), **tm})
            all_true += yfull[te].tolist(); all_pred += pred.tolist()
            all_spk += [test_spk] * len(te)
    elif split_mode == "fixed":
        test_spk = cfg["data"]["test_speaker"]; val_spk = cfg["data"].get("val_speaker")
        te = keep[spk == test_spk]
        if val_spk:
            va = keep[spk == val_spk]
            tr = keep[(spk != test_spk) & (spk != val_spk)]
        else:
            tr, va = _val_split(keep[spk != test_spk], yfull, len(classes), val_frac, rng)
        if len(te) == 0 or len(tr) == 0:
            raise SystemExit(f"[bin-eval] fixed split degenerate (test={test_spk!r} val={val_spk!r})")
        print(f"[bin-eval] fixed split: train={len(tr)} "
              f"(speakers {sorted({speakers[int(i)] for i in tr})}) "
              f"val={len(va)} ({val_spk}) test={len(te)} ({test_spk})")
        tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, test_spk, device, classes)
        folds.append({"speaker": test_spk, "val_speaker": val_spk,
                      "n_test": int(len(te)), "val_f1": round(vf1, 4), **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()
        all_spk += [test_spk] * len(te)
    else:  # random stratified 60/20/20
        perm = keep.copy(); rng.shuffle(perm)
        n = len(perm); te = perm[: int(0.2 * n)]; rest = perm[int(0.2 * n):]
        tr, va = _val_split(rest, yfull, len(classes), val_frac, rng)
        tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, "random",
                                    device, classes)
        folds.append({"speaker": "random", "n_test": int(len(te)),
                      "val_f1": round(vf1, 4), **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()

    if not folds:
        raise SystemExit("[bin-eval] no usable folds")
    pooled = S.classification_metrics(np.asarray(all_true), np.asarray(all_pred),
                                      len(classes), classes)
    out = {"encoder": cfg["encoder"]["checkpoint"], "type": "vjepa", "mode": "frozen",
           "task": "binary", "classes": classes,
           "probe": cfg["probe"].get("type", "attentive"),
           "spatial_size": cfg["data"]["spatial_size"],
           "frames_per_clip": cfg["data"]["frames_per_clip"], "seed": seed,
           "split_mode": split_mode,
           "row_stats": {k: row_stats[k] for k in ("n_pos", "n_neg", "pos_dur", "neg_dur")},
           "pooled": pooled,
           "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
           "balanced_acc_mean": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
           "accuracy_mean": round(float(np.mean([f["accuracy"] for f in folds])), 4),
           "folds": folds}
    _report(cfg, out)
    return out


def _report(cfg, out):
    print("\n===== STUTTER BINARY (fluent vs disfluent) RESULT =====")
    printable = {k: v for k, v in out.items() if k not in ("folds",)}
    print(json.dumps(printable, indent=2))
    print("[bin-eval] per-fold macro-F1:",
          {f["speaker"]: f["macro_f1"] for f in out["folds"]})
    tag = _tag(cfg)
    rp = os.path.join(cfg["meta"]["out"],
                      f"stutter_binary_{tag}_{out['probe']}_{out['split_mode']}_s{out['seed']}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[bin-eval] wrote {rp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="T-SSL/V-JEPA checkpoint (.pt)")
    ap.add_argument("--key", default=None, help="state-dict key (default target_encoder)")
    ap.add_argument("--probe", default=None,
                    choices=["attentive", "attentive_lstm", "mean", "mlp"])
    ap.add_argument("--lstm-hidden", type=int, default=None, help="attentive_lstm hidden size")
    ap.add_argument("--lstm-layers", type=int, default=None)
    ap.add_argument("--lstm-chunk", type=int, default=None,
                    help="attentive_lstm: temporal chunk for the per-frame spatial pool")
    ap.add_argument("--lstm-checkpoint", action="store_true",
                    help="attentive_lstm: gradient-checkpoint the chunked spatial pool")
    ap.add_argument("--split", default=None, choices=["loso", "fixed", "random"])
    ap.add_argument("--test-speaker", default=None)
    ap.add_argument("--val-speaker", default=None)
    ap.add_argument("--frames", type=int, default=None, help="frames_per_clip (default 32)")
    ap.add_argument("--neg-per-pos", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="video-decode workers (each ~1 core); default 6")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="main-process intra-op threads; default 2")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg["encoder"]["checkpoint"] = args.checkpoint
    if args.key is not None:
        cfg["encoder"]["key"] = args.key
    if args.probe is not None:
        cfg["probe"]["type"] = args.probe
    if args.lstm_hidden is not None:
        cfg["probe"]["lstm_hidden"] = args.lstm_hidden
    if args.lstm_layers is not None:
        cfg["probe"]["lstm_layers"] = args.lstm_layers
    if args.lstm_chunk is not None:
        cfg["probe"]["chunk"] = args.lstm_chunk
    if args.lstm_checkpoint:
        cfg["probe"]["checkpoint"] = True
    if args.split is not None:
        cfg["data"]["split_mode"] = args.split
    if args.test_speaker is not None:
        cfg["data"]["test_speaker"] = args.test_speaker
    if args.val_speaker is not None:
        cfg["data"]["val_speaker"] = args.val_speaker
    if args.frames is not None:
        cfg["data"]["frames_per_clip"] = args.frames
    if args.neg_per_pos is not None:
        cfg["data"]["neg_per_pos"] = args.neg_per_pos
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.cpu_threads is not None:
        cfg["data"]["cpu_threads"] = args.cpu_threads
    if args.tag is not None:
        cfg["meta"]["tag"] = args.tag
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    run(cfg)


if __name__ == "__main__":
    main()
