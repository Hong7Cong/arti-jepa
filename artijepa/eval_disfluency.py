"""Frozen-encoder (or fine-tuned) STUTTERING DISFLUENCY-TYPE classification eval
(Arti-JEPA downstream Task 8).

Segment-level analogue of `eval_phoneme.py`: instead of a per-token phoneme label,
each labeled disfluency interval `[xmin, xmax]` (seconds) is one example, pooled to
a single clip vector and classified into a disfluency type. Reuses the seconds-based
alignment and the frozen-encoder -> attentive-pool path; the canonical setup is the
**attentive probe at 256px**, matching the phoneme eval.

Two modes (`meta.mode`):
  * `frozen`   (default) -- freeze any encoder (V-JEPA / T-SSL, `image_baseline`,
    or `videomae`), extract per-segment features **once** (cached), then train a
    small attentive probe. Leave-one-speaker-out (LOSO) folds reuse the one cache.
  * `finetune` -- fine-tune the VideoMAE encoder + head end-to-end on raw clips
    (no feature cache), for the frozen-vs-finetune comparison.

Metrics: macro-F1 (primary; severe class imbalance), balanced accuracy, accuracy,
per-class P/R/F1, confusion matrix -- per held-out speaker and pooled over folds.

Run:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_disfluency \
        --config dev_artiJEPA/configs/eval_disfluency.yaml
    # VideoMAE frozen baseline / fine-tune:
    ... --model videomae --tag videomae_frozen
    ... --mode finetune --model videomae --tag videomae_ft
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from artijepa import stutter as S
from artijepa.checkpoint import clean_backbone_key, filtered_load, resolve_checkpoint
from artijepa.model import build_models
from artijepa.rtmri_dataset import PreprocConfig


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# encoder + dataset
# --------------------------------------------------------------------------- #
def load_frozen_encoder(cfg, device):
    d, ec = cfg["data"], cfg["encoder"]
    etype = ec.get("type", "vjepa")
    pool_spatial = d.get("pool_spatial", False)      # segment attentive keeps the grid
    if etype == "videomae":
        from artijepa.videomae_baseline import VideoMAEEncoder, DEFAULT_VIDEOMAE
        enc = VideoMAEEncoder(ec.get("videomae_name", DEFAULT_VIDEOMAE),
                              frame_batch=ec.get("frame_batch", 8),
                              pool_spatial=pool_spatial,
                              grid_cap=ec.get("grid_cap", 16)).to(device).eval()
        d["spatial_size"] = enc.backbone.input_size
        d["frames_per_clip"] = enc.backbone.num_frames
        d["tubelet_size"] = enc.backbone.tubelet
        d["intensity_norm"] = "minmax"; d.pop("grayscale_stats", None)
        ec["spec"] = enc.backbone.name
        print(f"[dis-eval] videomae {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"frames={enc.backbone.num_frames} input={enc.backbone.input_size}")
        return enc
    if etype == "image_baseline":
        from artijepa.baselines import BaselineEncoder
        enc = BaselineEncoder(ec["model"], tubelet_size=d.get("tubelet_size", 2),
                              frame_batch=ec.get("frame_batch", 64),
                              pool_spatial=pool_spatial,
                              grid_cap=ec.get("grid_cap", 16)).to(device).eval()
        d["spatial_size"] = enc.backbone.input_size
        d["intensity_norm"] = "minmax"; d.pop("grayscale_stats", None)
        ec["spec"] = enc.backbone.name
        print(f"[dis-eval] image-baseline {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"input={enc.backbone.input_size}")
        return enc
    # V-JEPA / Arti-JEPA T-SSL encoder
    encoder, _ = build_models(
        device=device, model_name=ec.get("model_name", "vit_large"),
        spatial_size=d["spatial_size"], frames_per_clip=d["frames_per_clip"],
        patch_size=d.get("patch_size", 16), tubelet_size=d.get("tubelet_size", 2),
        num_mask_tokens=1, use_activation_checkpointing=False)
    spec = ec.get("spec", "pretrained")
    if spec in (None, "pretrained"):
        ckpt = torch.load(resolve_checkpoint(ec.get("model_name", "vit_large"),
                                             ec.get("checkpoint")),
                          map_location="cpu", weights_only=False)
        key = ec.get("key", "target_encoder")
        if key not in ckpt:
            key = next(k for k in ("target_encoder", "encoder", "ema_encoder") if k in ckpt)
        n, miss, skip = filtered_load(encoder.backbone, clean_backbone_key(ckpt[key]))
        print(f"[dis-eval] encoder<-pretrained:{key} loaded {n} miss {len(miss)} skip {len(skip)}")
    else:
        ckpt = torch.load(spec, map_location="cpu", weights_only=False)
        key = ec.get("key", "auto")
        if key == "auto":
            key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        n, miss, skip = filtered_load(encoder, ckpt[key])
        print(f"[dis-eval] encoder<-{os.path.basename(spec)}:{key} loaded {n} "
              f"miss {len(miss)} skip {len(skip)} (epoch {ckpt.get('epoch','?')})")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _preproc(cfg):
    d = cfg["data"]
    gmean, gstd = 0.0, 1.0
    sp = d.get("grayscale_stats")
    if sp and os.path.exists(sp):
        st = json.load(open(sp)); gmean, gstd = st["mean"], st["std"]
    return PreprocConfig(
        target_fps=d.get("target_fps", 50.0), frames_per_clip=d["frames_per_clip"],
        sampling="tile", spatial_mode=d.get("spatial_mode", "resize"),
        spatial_size=d["spatial_size"], intensity_norm=d.get("intensity_norm", "zscore"),
        grayscale_mean=gmean, grayscale_std=gstd, augment=False,
        random_temporal_crop=False, tubelet_size=d.get("tubelet_size", 2))


def all_segment_dataset(cfg):
    """A DisfluencySegmentDataset over EVERY manifest row (task='binary' keeps all,
    incl. fluent negatives) -- the task-independent order for the feature cache."""
    d = cfg["data"]
    rows = S.read_manifest(d["manifest"], tiers=d.get("tiers"))
    classes, _ = S.label_space("binary")
    ds = S.DisfluencySegmentDataset(rows, _preproc(cfg), task="binary",
                                    classes=classes, event_pad_s=d.get("event_pad_s", 0.15))
    return ds


# --------------------------------------------------------------------------- #
# per-segment feature extraction (cached, task-independent)
# --------------------------------------------------------------------------- #
def _tag(cfg):
    d, ec = cfg["data"], cfg["encoder"]
    hd = {"spec": ec.get("spec", "pretrained"), "key": ec.get("key", "auto"),
          "sz": d["spatial_size"], "fpc": d["frames_per_clip"],
          "fps": d.get("target_fps", 50.0), "manifest": d["manifest"],
          "pad": d.get("event_pad_s", 0.15), "pool_spatial": d.get("pool_spatial", False),
          "grid_cap": ec.get("grid_cap", 16), "tiers": d.get("tiers")}
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or ec.get("spec", "pretrained")
    return f"{os.path.basename(str(tag))}_{h}"


@torch.no_grad()
def extract(encoder, cfg, device, dtype):
    """-> feats f16 [N,L,D] (attentive) or [N,D] (mean), seg i64 [N], speakers list.

    Pools each segment's tokens to a fixed representation: `pool_spatial=False`
    keeps the temporal-major grid `[T'*S', D]` for the attentive probe; `True`
    mean-pools every token to `[D]` for the mean/linear probes.
    """
    name = _tag(cfg)
    cdir = os.path.join(cfg["meta"]["cache_dir"], name); os.makedirs(cdir, exist_ok=True)
    fp, sp, kp = (os.path.join(cdir, f"all.{x}.npy") for x in ("feats", "seg", "spk"))
    if all(os.path.exists(x) for x in (fp, sp, kp)):
        print(f"[dis-eval] cache hit <- {cdir}")
        return np.load(fp, mmap_mode="r"), np.load(sp), list(np.load(kp))

    ds = all_segment_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["data"].get("batch_size", 8), shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4), collate_fn=S.collate)
    tub = cfg["data"].get("tubelet_size", 2)
    Tp = cfg["data"]["frames_per_clip"] // tub
    pool_spatial = cfg["data"].get("pool_spatial", False)
    feats = None; segs = []; spks = []; pos = 0; t0 = time.time(); N = len(ds)
    for bi, (clips, _lab, meta) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            tok = encoder.backbone(clips)                    # [B, L or T', D]
        B, Ntot, D = tok.shape
        tok = tok.float().reshape(B, Tp, Ntot // Tp, D)      # [B,T',S',D]
        if pool_spatial:
            v = tok.mean((1, 2))                             # [B,D] -- pool all tokens
        else:
            v = tok.reshape(B, -1, D)                        # [B, T'*S', D]
        v = v.cpu().numpy().astype(np.float16)
        if feats is None:
            feats = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16,
                                              shape=(N,) + v.shape[1:])
        feats[pos:pos + B] = v
        segs += [m[0] for m in meta]; spks += [m[1] for m in meta]; pos += B
        if bi % 20 == 0:
            print(f"[dis-eval]  extract {pos}/{N} ({time.time()-t0:.0f}s)")
    feats.flush()
    segs = np.asarray(segs, dtype=np.int64)
    np.save(sp, segs); np.save(kp, np.asarray(spks))
    print(f"[dis-eval] extracted {feats.shape} in {time.time()-t0:.0f}s -> {cdir}")
    return np.load(fp, mmap_mode="r"), segs, spks


# --------------------------------------------------------------------------- #
# segment probe
# --------------------------------------------------------------------------- #
class SegmentProbe(nn.Module):
    """Pool a segment's tokens -> disfluency class logits.

      attentive      -- V-JEPA AttentivePooler over the whole [T'*S',D] token set -> linear.
      attentive_lstm -- AttentivePooler over the S' SPATIAL tokens *per frame*
                        (factorized: one shared pooler applied to each of the T'
                        temporal steps) -> a [T',D] sequence -> LSTM over time ->
                        linear. This decouples the quadratic spatial attention
                        (now over S'=256, not T'*S') from the temporal model (an
                        O(T') LSTM, not O((T'*S')^2)), and the per-frame spatial
                        pool is run in temporal *chunks* with optional gradient
                        checkpointing so peak activation memory is O(chunk*S')
                        instead of O(T'*S') -- the VRAM win at large T'.
      mean           -- LayerNorm+linear on the pre-pooled [D] vector.
      mlp            -- 2-layer MLP on the pre-pooled [D] vector.
    """

    def __init__(self, dim, num_classes, kind="attentive", hidden=512, heads=8,
                 dropout=0.1, t_steps=None, lstm_hidden=256, lstm_layers=1,
                 bidirectional=True, temporal_pool="mean", chunk=0, checkpoint=False):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "attentive":
            from src.models.attentive_pooler import AttentivePooler
            self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                          num_heads=heads, mlp_ratio=4.0, depth=1)
            self.head = nn.Linear(dim, num_classes)
        elif kind == "attentive_lstm":
            from src.models.attentive_pooler import AttentivePooler
            assert t_steps, "attentive_lstm needs t_steps (# temporal tokens T')"
            self.t_steps = int(t_steps)
            self.chunk = int(chunk) or self.t_steps    # temporal chunk for the spatial pool
            self.use_ckpt = bool(checkpoint)
            self.temporal_pool = temporal_pool
            self.spatial_pool = AttentivePooler(num_queries=1, embed_dim=dim,
                                                num_heads=heads, mlp_ratio=4.0, depth=1)
            self.lstm = nn.LSTM(dim, lstm_hidden, num_layers=lstm_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            out_dim = lstm_hidden * (2 if bidirectional else 1)
            self.head = nn.Linear(out_dim, num_classes)
        elif kind == "mean":
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, num_classes))
        else:
            raise ValueError(kind)

    def _spatial_pool_chunked(self, x):
        """[B,T',S',D] -> [B,T',D]: attentive-pool the S' spatial tokens per frame.

        Frames are pooled independently, so we sweep the T' axis in ``chunk``-sized
        blocks; under training we optionally gradient-checkpoint each block so its
        activations are recomputed in backward (peak ~ O(B*chunk*S') not O(B*T'*S')).
        """
        B, T, S, D = x.shape
        outs = []
        for c0 in range(0, T, self.chunk):
            xc = x[:, c0:c0 + self.chunk].reshape(-1, S, D)      # [B*tc, S', D]
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                q = checkpoint(lambda t: self.spatial_pool(t).squeeze(1), xc,
                               use_reentrant=False)
            else:
                q = self.spatial_pool(xc).squeeze(1)             # [B*tc, D]
            outs.append(q.reshape(B, -1, D))
        return torch.cat(outs, dim=1)                            # [B, T', D]

    def forward(self, x):
        if self.kind == "attentive":                         # x: [B, L, D]
            q = self.pooler(x).squeeze(1)                    # [B, D]
            return self.head(self.drop(q))
        if self.kind == "attentive_lstm":                    # x: [B, T'*S', D]
            B, L, D = x.shape
            T = self.t_steps; S = L // T
            seq = self._spatial_pool_chunked(x.reshape(B, T, S, D))   # [B, T', D]
            out, _ = self.lstm(self.drop(seq))               # [B, T', H*dir]
            z = out.mean(1) if self.temporal_pool == "mean" else out[:, -1]
            return self.head(self.drop(z))
        return self.net(self.drop(x))                        # x: [B, D]


class _FeatDS(torch.utils.data.Dataset):
    def __init__(self, feats, idx, y):
        self.feats, self.idx, self.y = feats, idx, y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.array(self.feats[j], dtype=np.float32)),
                int(self.y[i]))


@torch.no_grad()
def _predict(clf, feats, idx, device, bs=256):
    clf.eval(); out = []
    if feats.ndim == 3:
        bs = min(bs, 64)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        x = torch.from_numpy(np.array(feats[j], dtype=np.float32)).to(device)
        out.append(clf(x).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _class_weights(y, num_classes, device):
    cnt = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = np.where(cnt > 0, len(y) / (num_classes * np.maximum(cnt, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_probe(cfg, feats, y, tr, va, te, spk_te, device, classes):
    """Train SegmentProbe on tr, model-select on va (macro-F1), predict te.
    Returns (test_metrics_dict, test_pred, best_val_f1)."""
    pc = cfg["probe"]; num_classes = len(classes)
    dim = feats.shape[-1]
    t_steps = cfg["data"]["frames_per_clip"] // cfg["data"].get("tubelet_size", 2)
    clf = SegmentProbe(dim, num_classes, kind=pc.get("type", "attentive"),
                       hidden=pc.get("hidden", 512), heads=pc.get("heads", 8),
                       dropout=pc.get("dropout", 0.1), t_steps=t_steps,
                       lstm_hidden=pc.get("lstm_hidden", 256),
                       lstm_layers=pc.get("lstm_layers", 1),
                       bidirectional=pc.get("bidirectional", True),
                       temporal_pool=pc.get("temporal_pool", "mean"),
                       chunk=pc.get("chunk", 0),
                       checkpoint=pc.get("checkpoint", False)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    w = _class_weights(y[tr], num_classes, device) if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    loader = torch.utils.data.DataLoader(
        _FeatDS(feats, tr, y[tr]), batch_size=pc.get("batch_size", 64), shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4), drop_last=False)
    best = {"val_f1": -1.0, "test": None, "pred": None}
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0
        for x, yy in loader:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad()
            loss = lossf(clf(x), yy)
            loss.backward(); opt.step(); run += float(loss); nb += 1
        vp = _predict(clf, feats, va, device)
        vm = S.classification_metrics(y[va], vp, num_classes, classes)
        if vm["macro_f1"] > best["val_f1"]:
            tp = _predict(clf, feats, te, device)
            tm = S.classification_metrics(y[te], tp, num_classes, classes)
            best = {"val_f1": vm["macro_f1"], "test": tm, "pred": tp,
                    "epoch": ep + 1, "train_loss": run / max(1, nb)}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[probe {spk_te} e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} "
                  f"val macroF1={vm['macro_f1']:.3f} best={best['val_f1']:.3f}")
    return best["test"], best["pred"], best["val_f1"]


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def _val_split(tr_idx, y, num_classes, frac, rng):
    """Carve a stratified val subset (frac) out of tr_idx."""
    va = []
    for c in range(num_classes):
        cls = tr_idx[y[tr_idx] == c]
        rng.shuffle(cls)
        va.extend(cls[: max(1, int(round(len(cls) * frac)))].tolist())
    va = np.asarray(sorted(va), dtype=np.int64)
    tr = np.asarray(sorted(set(tr_idx.tolist()) - set(va.tolist())), dtype=np.int64)
    return tr, va


def run_frozen(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "bfloat16").lower(), torch.float32)
    task = cfg["data"].get("task", "type5")
    classes, _ = S.label_space(task)
    # mean/mlp probes consume a pre-pooled [D] vector; attentive & attentive_lstm
    # both need the token grid kept (the latter pools spatial itself, per frame).
    cfg["data"]["pool_spatial"] = cfg["probe"].get("type", "attentive") in ("mean", "mlp")
    print(f"[dis-eval] frozen | task={task} classes={classes} probe={cfg['probe'].get('type','attentive')} "
          f"pool_spatial={cfg['data']['pool_spatial']} device={device}")

    encoder = load_frozen_encoder(cfg, device)
    feats, seg_ids, speakers = extract(encoder, cfg, device, dtype)
    del encoder; torch.cuda.empty_cache()

    # map cached segments -> task labels (drop rows the task excludes)
    rows_by_seg = {int(r["seg_id"]): r for r in
                   S.read_manifest(cfg["data"]["manifest"], tiers=cfg["data"].get("tiers"))}
    keep, y = [], []
    for i, sid in enumerate(seg_ids):
        lab = S.row_label(rows_by_seg[int(sid)], task, classes)
        if lab is not None:
            keep.append(i); y.append(lab)
    keep = np.asarray(keep, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    spk = np.asarray([speakers[i] for i in keep])
    uniq_spk = sorted(set(spk.tolist()))
    print(f"[dis-eval] kept {len(keep)} segments; per-class {np.bincount(y, minlength=len(classes)).tolist()}; "
          f"speakers {uniq_spk}")

    split_mode = cfg["data"].get("split_mode", "loso")
    val_frac = cfg["data"].get("val_frac", 0.15)
    folds, all_true, all_pred, all_spk = [], [], [], []
    if split_mode == "loso":
        for test_spk in uniq_spk:
            te = keep[spk == test_spk]
            tr_all = keep[spk != test_spk]
            y_te = y[spk == test_spk]; y_tr_all = y[spk != test_spk]
            if len(np.unique(y_tr_all)) < 2 or len(te) == 0:
                print(f"[dis-eval] skip fold {test_spk}: degenerate")
                continue
            # local index space aligned to feats rows
            tr_all_idx = tr_all
            # build a y lookup aligned to feats rows
            yfull = np.full(len(feats), -1, dtype=np.int64); yfull[keep] = y
            tr, va = _val_split(tr_all_idx, yfull, len(classes), val_frac, rng)
            tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, test_spk,
                                        device, classes)
            folds.append({"speaker": test_spk, "val_f1": round(vf1, 4), **tm})
            all_true += yfull[te].tolist(); all_pred += pred.tolist()
            all_spk += [test_spk] * len(te)
    elif split_mode == "fixed":
        # ONE fixed, subject-disjoint split shared across encoders: held-out
        # test_speaker (+ val_speaker); train = the remaining speakers. This is
        # what guarantees every model sees the identical train/val/test partition.
        yfull = np.full(len(feats), -1, dtype=np.int64); yfull[keep] = y
        test_spk = cfg["data"]["test_speaker"]; val_spk = cfg["data"].get("val_speaker")
        te = keep[spk == test_spk]
        if val_spk:
            va = keep[spk == val_spk]
            tr = keep[(spk != test_spk) & (spk != val_spk)]
        else:
            tr, va = _val_split(keep[spk != test_spk], yfull, len(classes), val_frac, rng)
        if len(te) == 0 or len(tr) == 0:
            raise SystemExit(f"[dis-eval] fixed split degenerate (test={test_spk!r} val={val_spk!r})")
        print(f"[dis-eval] fixed split: train={len(tr)} (speakers "
              f"{sorted({speakers[int(i)] for i in tr})}) "
              f"val={len(va)} ({val_spk}) test={len(te)} ({test_spk})")
        tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, test_spk, device, classes)
        folds.append({"speaker": test_spk, "val_speaker": val_spk,
                      "val_f1": round(vf1, 4), **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()
        all_spk += [test_spk] * len(te)
    else:  # random stratified split
        yfull = np.full(len(feats), -1, dtype=np.int64); yfull[keep] = y
        perm = keep.copy(); rng.shuffle(perm)
        n = len(perm); te = perm[: int(0.2 * n)]; rest = perm[int(0.2 * n):]
        tr, va = _val_split(rest, yfull, len(classes), val_frac, rng)
        tm, pred, vf1 = train_probe(cfg, feats, yfull, tr, va, te, "random",
                                    device, classes)
        folds.append({"speaker": "random", "val_f1": round(vf1, 4), **tm})
        all_true += yfull[te].tolist(); all_pred += pred.tolist()

    pooled = S.classification_metrics(np.asarray(all_true), np.asarray(all_pred),
                                      len(classes), classes)
    out = {"encoder": cfg["encoder"].get("spec", "pretrained"),
           "type": cfg["encoder"].get("type", "vjepa"), "mode": "frozen",
           "task": task, "classes": classes, "probe": cfg["probe"].get("type", "attentive"),
           "spatial_size": cfg["data"]["spatial_size"], "seed": seed,
           "split_mode": split_mode,
           "pooled": pooled,
           "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
           "balanced_acc_mean": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
           "folds": folds}
    _report(cfg, out)
    return out


# --------------------------------------------------------------------------- #
# finetune (VideoMAE / raw clips, no cache)
# --------------------------------------------------------------------------- #
def _make_split_rows(cfg, task, classes, rng):
    """(train_rows, val_rows, test_rows) of manifest dicts for the finetune loop."""
    d = cfg["data"]
    rows = [r for r in S.read_manifest(d["manifest"], tiers=d.get("tiers"))
            if S.row_label(r, task, classes) is not None]
    if d.get("split_mode", "loso") == "loso":
        test_spk = d.get("test_speaker") or sorted({r["speaker"] for r in rows})[-1]
        te = [r for r in rows if r["speaker"] == test_spk]
        rest = [r for r in rows if r["speaker"] != test_spk]
    else:
        idx = np.arange(len(rows)); rng.shuffle(idx)
        cut = int(0.2 * len(rows))
        te = [rows[i] for i in idx[:cut]]; rest = [rows[i] for i in idx[cut:]]
    rng.shuffle(rest)
    vcut = max(1, int(cfg["data"].get("val_frac", 0.15) * len(rest)))
    return rest[vcut:], rest[:vcut], te


def run_finetune(cfg):
    from artijepa.videomae_baseline import VideoMAEClassifier, DEFAULT_VIDEOMAE
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task = cfg["data"].get("task", "type5"); classes, _ = S.label_space(task)
    ec, pc = cfg["encoder"], cfg["probe"]
    # VideoMAE fixes the input geometry
    cfg["data"]["spatial_size"] = 224; cfg["data"]["frames_per_clip"] = 16
    cfg["data"]["tubelet_size"] = 2; cfg["data"]["intensity_norm"] = "minmax"
    cfg["data"].pop("grayscale_stats", None)
    tr_rows, va_rows, te_rows = _make_split_rows(cfg, task, classes, rng)
    pcfg = _preproc(cfg)
    def ds(rows):
        return S.DisfluencySegmentDataset(rows, pcfg, task, classes,
                                          event_pad_s=cfg["data"].get("event_pad_s", 0.15))
    tr_ds, va_ds, te_ds = ds(tr_rows), ds(va_rows), ds(te_rows)
    print(f"[dis-eval] finetune | task={task} classes={classes} "
          f"train/val/test={len(tr_ds)}/{len(va_ds)}/{len(te_ds)}")

    model = VideoMAEClassifier(len(classes), ec.get("videomae_name", DEFAULT_VIDEOMAE),
                               pool=pc.get("type", "attentive"), heads=pc.get("heads", 8),
                               freeze_encoder=ec.get("freeze_encoder", False),
                               dropout=pc.get("dropout", 0.1)).to(device)
    enc_lr = pc.get("encoder_lr", 1e-5); head_lr = pc.get("lr", 1e-3)
    enc_params = [p for p in model.backbone.model.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.model.")]
    groups = ([{"params": enc_params, "lr": enc_lr}] if enc_params else []) + \
             [{"params": head_params, "lr": head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=pc.get("wd", 0.01))
    w = _class_weights(tr_ds.labels, len(classes), device) \
        if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "bfloat16").lower(), torch.float32)

    def loader(dd, sh):
        return torch.utils.data.DataLoader(dd, batch_size=pc.get("batch_size", 8),
                                           shuffle=sh, num_workers=cfg["data"].get("num_workers", 4),
                                           collate_fn=S.collate)
    tl = loader(tr_ds, True)

    @torch.no_grad()
    def evaluate(dd):
        model.eval(); yt, yp = [], []
        for clips, yy, _m in loader(dd, False):
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32 and device.type == "cuda")):
                logit = model(clips.to(device))
            yp += logit.argmax(-1).cpu().tolist(); yt += yy.tolist()
        return S.classification_metrics(np.asarray(yt), np.asarray(yp), len(classes), classes)

    epochs, warmup, base = pc.get("epochs", 15), pc.get("warmup", 2), head_lr
    best = {"val_f1": -1.0, "test": None}
    for ep in range(epochs):
        scale = (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        opt.param_groups[-1]["lr"] = base * scale
        if enc_params:
            opt.param_groups[0]["lr"] = enc_lr * scale
        model.train(); run = nb = 0
        for clips, yy, _m in tl:
            clips, yy = clips.to(device), yy.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32 and device.type == "cuda")):
                loss = lossf(model(clips), yy)
            loss.backward(); opt.step(); run += float(loss); nb += 1
        vm = evaluate(va_ds)
        if vm["macro_f1"] > best["val_f1"]:
            best = {"val_f1": vm["macro_f1"], "test": evaluate(te_ds), "epoch": ep + 1}
        print(f"[ft e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} "
              f"val macroF1={vm['macro_f1']:.3f} best={best['val_f1']:.3f}")

    out = {"encoder": ec.get("videomae_name", DEFAULT_VIDEOMAE), "type": "videomae",
           "mode": "finetune", "freeze_encoder": ec.get("freeze_encoder", False),
           "task": task, "classes": classes, "probe": pc.get("type", "attentive"),
           "seed": seed, "split_mode": cfg["data"].get("split_mode", "loso"),
           "test_speaker": cfg["data"].get("test_speaker"),
           "val_f1": best["val_f1"], "test": best["test"]}
    _report(cfg, out)
    return out


# --------------------------------------------------------------------------- #
def _report(cfg, out):
    print("\n===== DISFLUENCY RESULT =====")
    printable = {k: v for k, v in out.items() if k not in ("folds",)}
    print(json.dumps(printable, indent=2))
    tag = _tag(cfg)
    rp = os.path.join(cfg["meta"]["out"],
                      f"disfluency_{out['task']}_{tag}_{out['probe']}_{out['mode']}_"
                      f"s{out['seed']}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[dis-eval] wrote {rp}")


def run(cfg):
    if cfg["meta"].get("mode", "frozen") == "finetune":
        return run_finetune(cfg)
    return run_frozen(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default=None, help="T-SSL/V-JEPA checkpoint path (encoder.spec)")
    ap.add_argument("--model", default=None,
                    help="videomae | an image-baseline (clip|siglip|dinov2|vitl|resnet)")
    ap.add_argument("--mode", default=None, choices=["frozen", "finetune"])
    ap.add_argument("--task", default=None, choices=["type5", "type3", "binary"])
    ap.add_argument("--probe", default=None,
                    choices=["attentive", "attentive_lstm", "mean", "mlp"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default=None, choices=["loso", "fixed", "random"])
    ap.add_argument("--test-speaker", default=None)
    ap.add_argument("--val-speaker", default=None)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="finetune mode: keep VideoMAE frozen (probe-only)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    if args.mode is not None:
        cfg["meta"]["mode"] = args.mode
    if args.model == "videomae":
        cfg["encoder"]["type"] = "videomae"
    elif args.model is not None:
        cfg["encoder"]["type"] = "image_baseline"; cfg["encoder"]["model"] = args.model
    if args.encoder is not None:
        cfg["encoder"]["spec"] = args.encoder
    if args.freeze_encoder:
        cfg["encoder"]["freeze_encoder"] = True
    if args.task is not None:
        cfg["data"]["task"] = args.task
    if args.probe is not None:
        cfg["probe"]["type"] = args.probe
    if args.split is not None:
        cfg["data"]["split_mode"] = args.split
    if args.test_speaker is not None:
        cfg["data"]["test_speaker"] = args.test_speaker
    if args.val_speaker is not None:
        cfg["data"]["val_speaker"] = args.val_speaker
    if args.tag is not None:
        cfg["meta"]["tag"] = args.tag
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch
    run(cfg)


if __name__ == "__main__":
    main()
