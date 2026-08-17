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
    # parallel key -- FINAL epoch-215 checkpoint under its own tag (own feature cache; the
    # config default stays ckpt_100/tag tssl256, so existing caches + results are untouched):
    ... --checkpoint /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt --tag tssl256_215
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
from artijepa.checkpoint import clean_backbone_key, filtered_load, resolve_checkpoint
from artijepa.model import build_models
# Reuse the exact probe / training machinery from the disfluency-type eval so the
# two evals stay byte-identical where they overlap (probe, class weights, folds).
from artijepa.eval_disfluency import train_probe, _val_split, pool_mode_for_probe

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
# frozen encoder -- multi-encoder dispatch (docs/STUTTERING.md §12 phase 1)
# --------------------------------------------------------------------------- #
# Every encoder object exposes ``.backbone(clip)`` returning the FULL temporal-major
# token grid ``[B, T'*S', D]`` (``pool_spatial=False``); ``extract`` then reduces that
# grid itself via ``pool_mode`` (none/spatial/all). This mirrors the multi-encoder
# ``load_frozen_encoder`` in ``eval_disfluency.py`` so the two evals stay identical on
# encoder handling -- but on the OpenCV ``stutter_binary`` loader (pal8-safe), not the
# decord loader that reads this corpus as all-black frames (see the doc's decoder bug).
#
# Supported ``encoder.type``:
#   vjepa (default) -- Arti-JEPA / V-JEPA2 ViT-L. ``spec``/``checkpoint``:
#       * "pretrained" (or a bare model_name) -> FAIR V-JEPA2 weights via
#         resolve_checkpoint, loaded onto ``encoder.backbone`` with clean_backbone_key.
#       * a local .pt path -> the T-SSL fine-tune, loaded onto the full ``encoder``
#         (its keys are already ``backbone.*``), like examples/demo.ipynb.
#   videomae        -- MCG-NJU VideoMAE-L (HF) or a local rt-MRI repo ``.pth`` via
#                      ``videomae_checkpoint`` (224px/16f/tubelet2, minmax norm).
#   image_baseline  -- a timm 2-D encoder (``model``: vitl|dinov2|clip|siglip|resnet)
#                      applied per frame, tubelet-pooled over time (minmax norm).
def load_frozen_encoder(cfg, device):
    d, ec = cfg["data"], cfg["encoder"]
    etype = ec.get("type", "vjepa")

    if etype == "videomae":
        from artijepa.videomae_baseline import VideoMAEEncoder, DEFAULT_VIDEOMAE
        enc = VideoMAEEncoder(ec.get("videomae_name", DEFAULT_VIDEOMAE),
                              frame_batch=ec.get("frame_batch", 8),
                              pool_spatial=False, grid_cap=ec.get("grid_cap", 16),
                              checkpoint=ec.get("videomae_checkpoint")).to(device).eval()
        # feed clips at VideoMAE's native geometry (it resamples T->16 internally,
        # but set frames/tubelet so extract's Tp = num_frames//tubelet is correct)
        d["spatial_size"] = enc.backbone.input_size
        d["frames_per_clip"] = enc.backbone.num_frames
        d["tubelet_size"] = enc.backbone.tubelet
        d["intensity_norm"] = "minmax"; d.pop("grayscale_stats", None)
        ec["spec"] = enc.backbone.name
        print(f"[bin-eval] videomae {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"frames={enc.backbone.num_frames} input={enc.backbone.input_size}")
        return enc

    if etype == "image_baseline":
        from artijepa.baselines import BaselineEncoder
        enc = BaselineEncoder(ec["model"], tubelet_size=d.get("tubelet_size", 2),
                              frame_batch=ec.get("frame_batch", 64),
                              pool_spatial=False, grid_cap=ec.get("grid_cap", 16)).to(device).eval()
        d["spatial_size"] = enc.backbone.input_size
        d["intensity_norm"] = "minmax"; d.pop("grayscale_stats", None)
        ec["spec"] = enc.backbone.name
        print(f"[bin-eval] image-baseline {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"input={enc.backbone.input_size}")
        return enc

    # --- V-JEPA / Arti-JEPA T-SSL ViT-L (build at the config geometry) ---
    encoder, _ = build_models(
        device=device, model_name=ec.get("model_name", "vit_large"),
        spatial_size=d["spatial_size"], frames_per_clip=d["frames_per_clip"],
        patch_size=d.get("patch_size", 16), tubelet_size=d.get("tubelet_size", 2),
        num_mask_tokens=1, use_activation_checkpointing=False)
    spec = ec.get("spec", ec.get("checkpoint"))
    if spec in (None, "pretrained"):
        ckpt = torch.load(resolve_checkpoint(ec.get("model_name", "vit_large"),
                                             ec.get("checkpoint")),
                          map_location="cpu", weights_only=False)
        key = ec.get("key", "target_encoder")
        if key not in ckpt:
            key = next(k for k in ("target_encoder", "encoder", "ema_encoder") if k in ckpt)
        n, miss, skip = filtered_load(encoder.backbone, clean_backbone_key(ckpt[key]))
        ec["spec"] = "pretrained"
        print(f"[bin-eval] encoder<-pretrained:{key} loaded {n} miss {len(miss)} skip {len(skip)}")
    else:
        ckpt = torch.load(spec, map_location="cpu", weights_only=False)
        key = ec.get("key", "target_encoder")
        if key == "auto":
            key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        if key not in ckpt:
            key = next(k for k in ("target_encoder", "encoder", "ema_encoder") if k in ckpt)
        n, miss, skip = filtered_load(encoder, ckpt[key])
        print(f"[bin-eval] encoder<-{os.path.basename(str(spec))}:{key} "
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


def _enc_id(ec):
    """A stable identity string for whichever encoder is configured (checkpoint path,
    pretrained tag, VideoMAE name/ckpt, or image-baseline alias)."""
    etype = ec.get("type", "vjepa")
    if etype == "videomae":
        return f"videomae:{ec.get('videomae_checkpoint') or ec.get('videomae_name') or 'large'}"
    if etype == "image_baseline":
        return f"image_baseline:{ec.get('model')}"
    return f"vjepa:{ec.get('spec') or ec.get('checkpoint') or 'pretrained'}"


def _tag(cfg):
    """A content hash so different encoders / geometries / row-builds get distinct caches."""
    d, ec = cfg["data"], cfg["encoder"]
    pm = d.get("pool_mode", "none")
    etype = ec.get("type", "vjepa")
    hd = {"enc": _enc_id(ec), "etype": etype, "key": ec.get("key", "target_encoder"),
          "sz": d["spatial_size"], "fpc": d["frames_per_clip"],
          "tub": d.get("tubelet_size", 2), "pad": d.get("event_pad_s", 0.0),
          "grid_cap": ec.get("grid_cap", 16),
          "pool_spatial": pm == "all",     # kept for hash-compat with earlier caches
          "neg_per_pos": d.get("neg_per_pos", 1.0), "build_seed": d.get("build_seed", 0),
          "tiers": d.get("tiers", ["disfluency"]), "min_dur": d.get("min_dur", 0.20),
          "max_dur": d.get("max_dur", 8.0), "merge_gap": d.get("merge_gap", 0.25)}
    if pm == "spatial":                    # new mode -> distinct cache; leaves none/all hashes intact
        hd["pool_mode"] = "spatial"
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag")
    if not tag:
        ckpt = ec.get("checkpoint")
        tag = os.path.basename(os.path.dirname(ckpt)) if ckpt else etype
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
    pool_mode = d.get("pool_mode", "none")
    feats = None; labels = []; spks = []; pos = 0; t0 = time.time(); N = len(ds)
    for bi, (clips, y, meta) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            tok = encoder.backbone(clips)                    # [B, T'*S', D]
        B, Ntot, D = tok.shape
        tok = tok.float().reshape(B, Tp, Ntot // Tp, D)      # [B,T',S',D]
        if pool_mode == "all":
            v = tok.mean((1, 2))                             # [B, D]        (mean/mlp)
        elif pool_mode == "spatial":
            v = tok.mean(2)                                  # [B, T', D]    (pooled_attentive)
        else:
            v = tok.reshape(B, -1, D)                        # [B, T'*S', D] (attentive[_lstm])
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
    # How the [T',S',D] grid is reduced at extraction (drives the cache shape):
    #   none    -> [T'*S', D]  attentive / attentive_lstm
    #   spatial -> [T', D]      pooled_attentive (mean spatial -> attend time; tiny cache)
    #   all     -> [D]          mean / mlp
    ptype = cfg["probe"].get("type", "attentive")
    cfg["data"]["pool_mode"] = pool_mode_for_probe(ptype)
    print(f"[bin-eval] frozen binary | classes={classes} probe={ptype} "
          f"pool_mode={cfg['data']['pool_mode']} device={device}")

    rows, row_stats = build_rows(cfg)
    encoder = load_frozen_encoder(cfg, device)      # may mutate geometry + intensity_norm
    # report the ACTUAL normalization the loader will use (baselines/videomae switch
    # to minmax and drop grayscale_stats inside load_frozen_encoder)
    norm = cfg["data"].get("intensity_norm", "zscore")
    gs = cfg["data"].get("grayscale_stats")
    if norm == "zscore" and gs and os.path.exists(gs):
        print(f"[bin-eval] intensity_norm=zscore; grayscale stats <- {gs}")
    elif norm == "zscore":
        print(f"[bin-eval] intensity_norm=zscore; grayscale stats {gs!r} absent -> "
              f"global channel-norm defaults to mean=0/std=1 (per-clip z-score still applied)")
    else:
        print(f"[bin-eval] intensity_norm={norm} (encoder applies its own mean/std)")

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

    # ``meta.save_probe`` (--save-probe) swaps in eval_stutter_type.train_probe_ckpt:
    # the SAME optimisation (AdamW, warmup+cosine, best-val model selection) that also
    # writes the best-val state_dict + everything needed to rebuild the probe to
    # ``meta.out/probes/<feature-tag>_<probe>_binary_<split>_s<seed>/fold_<spk>.pt``.
    # Imported lazily: eval_stutter_type imports THIS module (row builder / extract),
    # so a top-level import would be circular.
    save_probe = bool(meta.get("save_probe"))
    ckpt_dir = None
    if save_probe:
        from artijepa.eval_stutter_type import train_probe_ckpt
        cfg["data"].setdefault("task", "binary")   # recorded in the .pt metadata only
        ckpt_dir = os.path.join(meta["out"], "probes",
                                f"{_tag(cfg)}_{ptype}_binary_{split_mode}_s{seed}")
        print(f"[bin-eval] probe weights -> {ckpt_dir}/fold_<speaker>.pt")

    def _train(tr, va, te, name):
        """-> (test_metrics, test_pred, best_val_f1, extra fold fields)."""
        if save_probe:
            tm, pred, vf1, cp = train_probe_ckpt(cfg, feats, yfull, tr, va, te, name,
                                                 device, classes, ckpt_dir)
            return tm, pred, vf1, {"probe_ckpt": cp}
        tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, name, device, classes)
        return tm, pred, vf1, {}

    if split_mode == "loso":
        for test_spk in uniq_spk:
            te = keep[spk == test_spk]
            tr_all = keep[spk != test_spk]
            if len(np.unique(yfull[tr_all])) < 2 or len(te) == 0:
                print(f"[bin-eval] skip fold {test_spk}: degenerate")
                continue
            tr, va = _val_split(tr_all, yfull, len(classes), val_frac, rng)
            tm, pred, vf1, extra = _train(tr, va, te, test_spk)
            folds.append({"speaker": test_spk, "n_test": int(len(te)),
                          "val_f1": round(vf1, 4), **extra, **tm})
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
        tm, pred, vf1, extra = _train(tr, va, te, test_spk)
        folds.append({"speaker": test_spk, "val_speaker": val_spk,
                      "n_test": int(len(te)), "val_f1": round(vf1, 4), **extra, **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()
        all_spk += [test_spk] * len(te)
    else:  # random stratified 60/20/20
        perm = keep.copy(); rng.shuffle(perm)
        n = len(perm); te = perm[: int(0.2 * n)]; rest = perm[int(0.2 * n):]
        tr, va = _val_split(rest, yfull, len(classes), val_frac, rng)
        tm, pred, vf1, extra = _train(tr, va, te, "random")
        folds.append({"speaker": "random", "n_test": int(len(te)),
                      "val_f1": round(vf1, 4), **extra, **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()

    if not folds:
        raise SystemExit("[bin-eval] no usable folds")
    pooled = S.classification_metrics(np.asarray(all_true), np.asarray(all_pred),
                                      len(classes), classes)
    ec = cfg["encoder"]
    out = {"encoder": _enc_id(ec), "type": ec.get("type", "vjepa"), "mode": "frozen",
           "tag": cfg["meta"].get("tag"), "task": "binary", "classes": classes,
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
    # --- encoder selection (docs/STUTTERING.md §12 multi-encoder sweep) ---
    ap.add_argument("--enc-type", default=None,
                    choices=["vjepa", "videomae", "image_baseline"],
                    help="encoder family (default from config; 'vjepa' = Arti-JEPA/V-JEPA2 ViT-L)")
    ap.add_argument("--model", default=None,
                    help="image_baseline alias: vitl | dinov2 | clip | siglip | resnet")
    ap.add_argument("--spec", default=None,
                    help="vjepa: 'pretrained' (FAIR V-JEPA2) or a local .pt path (T-SSL)")
    ap.add_argument("--videomae-name", default=None, help="HF VideoMAE repo id")
    ap.add_argument("--videomae-checkpoint", default=None,
                    help="local rt-MRI VideoMAE repo .pth (e.g. checkpoint-214.pth)")
    ap.add_argument("--grid-cap", type=int, default=None,
                    help="cap the spatial grid side for videomae/image_baseline (default 16)")
    ap.add_argument("--checkpoint", default=None, help="T-SSL/V-JEPA checkpoint (.pt)")
    ap.add_argument("--key", default=None, help="state-dict key (default target_encoder)")
    ap.add_argument("--probe", default=None,
                    choices=["attentive", "pooled_attentive", "attentive_lstm", "mean", "mlp"])
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
    ap.add_argument("--save-probe", action="store_true",
                    help="save the best-val probe weights per LOSO fold under "
                         "meta.out/probes/<tag>_<probe>_binary_<split>_s<seed>/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.enc_type is not None:
        cfg["encoder"]["type"] = args.enc_type
    if args.model is not None:
        cfg["encoder"]["type"] = "image_baseline"; cfg["encoder"]["model"] = args.model
    if args.spec is not None:
        cfg["encoder"]["type"] = "vjepa"; cfg["encoder"]["spec"] = args.spec
    if args.videomae_name is not None:
        cfg["encoder"]["type"] = "videomae"; cfg["encoder"]["videomae_name"] = args.videomae_name
    if args.videomae_checkpoint is not None:
        cfg["encoder"]["type"] = "videomae"
        cfg["encoder"]["videomae_checkpoint"] = args.videomae_checkpoint
    if args.grid_cap is not None:
        cfg["encoder"]["grid_cap"] = args.grid_cap
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
    if args.save_probe:
        cfg["meta"]["save_probe"] = True
    run(cfg)


if __name__ == "__main__":
    main()
