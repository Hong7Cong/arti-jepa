"""Frozen-encoder PHONEME-prediction eval (Arti-JEPA downstream task, Plan T0/C).

Replaces the old stimulus-group probe. Freeze a V-JEPA2 / Arti-JEPA encoder,
extract **per temporal-token** features, train a small probe to predict the
phoneme at each token, and report frame-level **Cohen's kappa** + sequence
**Phoneme Error Rate (PER)** on held-out data. Point it at different encoders for
the headline "with vs without T-SSL" lift.

Two datasets (set `data.kind`):
  * `usc_lss`  -- Task 2, GOLD phonemes, one OOD speaker (104x104 @ 99 fps). Self
    contained, no audio model. (`artijepa.usc_lss`)
  * `pseudo`   -- Task 1, PSEUDO phonemes from an audio model on the 75-speaker
    corpus. (`artijepa.audio_phoneme`; needs a one-off label-gen pass.)

Per-token features: the ViT emits tokens in temporal-major order, so
`[B, N, D] -> [B, T', S', D] -> mean over S' -> [B, T', D]` gives one feature per
80 ms token (tubelet 2 @ 25 fps). Alignment is in seconds (`phonemes.py`), so the
OOD 99 fps is handled by the resample alone.

Run:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_phoneme \
        --config dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml
    # with-T-SSL:
    ... --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt --tag tssl128
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

from artijepa import phonemes as P
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
    if ec.get("type") == "image_baseline":
        # Frozen 2-D image encoder (CLIP/SigLIP/DINOv2/ViT-L/ResNet) applied
        # per-frame + tubelet-pooled to V-JEPA's temporal token grid (Plan Part C
        # baseline). Feed it model-native frames in [0,1] (minmax), no rtMRI norm.
        # pool_spatial (set by run() from the probe head): True -> the model's native
        # pooled embedding [B,T',D] (default heads); False -> the per-frame PATCH-token
        # grid [B,T'*S',D] (temporal-major, so extract()'s reshape yields [B,T',S',D])
        # for the spatial-aware probe -- the fair-fight analogue of V-JEPA's spatial
        # tokens. The grid side is capped at grid_cap (default 16) by adaptive-pool so
        # dinov2's 37x37@518px stays tractable (others are <=16x16 natively, untouched).
        from artijepa.baselines import BaselineEncoder
        enc = BaselineEncoder(ec["model"], tubelet_size=d.get("tubelet_size", 2),
                              frame_batch=ec.get("frame_batch", 64),
                              pool_spatial=d.get("pool_spatial", True),
                              grid_cap=ec.get("grid_cap", 16)).to(device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        d["spatial_size"] = enc.backbone.input_size
        d["intensity_norm"] = "minmax"
        d.pop("grayscale_stats", None)
        ec["spec"] = enc.backbone.name          # differentiates the feature cache/tag
        mode = "pooled [B,T',D]" if enc.backbone.pool_spatial else \
            f"un-pooled grid [B,T',S'<= {enc.backbone.grid_cap}^2,D]"
        print(f"[ph-eval] image-baseline {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"input={enc.backbone.input_size} minmax->[0,1] -> per-frame, {mode}")
        return enc
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
        print(f"[ph-eval] encoder<-pretrained:{key} loaded {n} miss {len(miss)} skip {len(skip)}")
    else:
        ckpt = torch.load(spec, map_location="cpu", weights_only=False)
        key = ec.get("key", "auto")
        if key == "auto":
            key = "target_encoder" if "target_encoder" in ckpt else "encoder"
        n, miss, skip = filtered_load(encoder, ckpt[key])
        print(f"[ph-eval] encoder<-{os.path.basename(spec)}:{key} loaded {n} "
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
        target_fps=d.get("target_fps", 25.0), frames_per_clip=d["frames_per_clip"],
        sampling="tile", spatial_mode=d.get("spatial_mode", "resize"),
        spatial_size=d["spatial_size"], intensity_norm=d.get("intensity_norm", "zscore"),
        grayscale_mean=gmean, grayscale_std=gstd, augment=False,
        random_temporal_crop=False, tubelet_size=d.get("tubelet_size", 2))


def build_dataset(cfg, split):
    d = cfg["data"]
    pc = _preproc(cfg)
    if d.get("kind", "usc_lss") == "usc_lss":
        from artijepa.usc_lss import USCLSSPhonemeDataset, collate
        ds = USCLSSPhonemeDataset(d["manifest"], split=split, cfg=pc)
        return ds, collate
    from artijepa.audio_phoneme import PseudoPhonemeDataset, collate, DEFAULT_LABEL_DIR
    ds = PseudoPhonemeDataset(d["manifest"], split=split, cfg=pc,
                              label_dir=d.get("label_dir", DEFAULT_LABEL_DIR))
    return ds, collate


# --------------------------------------------------------------------------- #
# per-token feature extraction (cached)
# --------------------------------------------------------------------------- #
def _tag(cfg, split):
    d, ec = cfg["data"], cfg["encoder"]
    hd = {
        "spec": ec.get("spec", "pretrained"), "key": ec.get("key", "auto"),
        "sz": d["spatial_size"], "fpc": d["frames_per_clip"],
        "fps": d.get("target_fps", 25.0), "kind": d.get("kind", "usc_lss"),
        "manifest": d["manifest"]}
    unpooled = not d.get("pool_spatial", True)   # keep full [T',S',D] token grid
    if unpooled:                                 # only perturb the hash when un-pooled
        hd["pool_spatial"] = False               # -> existing pooled caches still hit
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or ec.get("spec", "pretrained")
    name = os.path.basename(str(tag)) + ("sp" if unpooled else "")
    return f"{name}_{h}", split


@torch.no_grad()
def extract(encoder, cfg, split, device, dtype):
    """-> feats f16 [N,T',D], labels i64 [N,T'], meta i64 [N,3]=(utt,chunk,n_chunks)."""
    name, split = _tag(cfg, split)
    cdir = os.path.join(cfg["meta"]["cache_dir"], name); os.makedirs(cdir, exist_ok=True)
    fp, lp, mp = (os.path.join(cdir, f"{split}.{x}.npy") for x in ("feats", "labels", "meta"))
    if all(os.path.exists(x) for x in (fp, lp, mp)):
        print(f"[ph-eval] cache hit {split} <- {cdir}")
        return np.load(fp, mmap_mode="r"), np.load(lp), np.load(mp)

    ds, collate = build_dataset(cfg, split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["data"].get("batch_size", 16), shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4), collate_fn=collate)
    tub = cfg["data"].get("tubelet_size", 2)
    Tp = cfg["data"]["frames_per_clip"] // tub
    pool_spatial = cfg["data"].get("pool_spatial", True)
    feats = None; labels = []; meta = []; pos = 0; t0 = time.time()
    N = len(ds)
    for bi, (clips, lab, m) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            tok = encoder.backbone(clips)                 # [B,N,D]
        B, Ntot, D = tok.shape
        tok = tok.float().reshape(B, Tp, Ntot // Tp, D)          # [B,T',S',D]
        if pool_spatial:
            tok = tok.mean(2)                                    # [B,T',D]
        tok = tok.cpu().numpy().astype(np.float16)
        if feats is None:                                        # (N,T',D) or (N,T',S',D)
            feats = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16,
                                              shape=(N,) + tok.shape[1:])
        feats[pos:pos + B] = tok
        labels.append(lab.numpy()); meta += m; pos += B
        if bi % 20 == 0:
            print(f"[ph-eval]  extract {split} {pos}/{N} ({time.time()-t0:.0f}s)")
    feats.flush()
    labels = np.concatenate(labels).astype(np.int64)
    meta = np.asarray(meta, dtype=np.int64)
    np.save(lp, labels); np.save(mp, meta)
    print(f"[ph-eval] extracted {split}: {feats.shape} in {time.time()-t0:.0f}s -> {cdir}")
    return np.load(fp, mmap_mode="r"), labels, meta


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
class TokenProbe(nn.Module):
    """Per temporal-token phoneme classifier over frozen features [B,T',D].

    Heads ordered by temporal-context capacity (Plan B.4 probe-head ablation):
      linear / mlp  -- per-token, no context
      tcn           -- local 1-D conv (+/-2 tokens)
      lstm          -- bi-LSTM, full-sequence recurrence
      transformer   -- self-attention encoder, global context
    All emit per-token logits [B,T',C]; works for both CE and CTC training.

    SPATIAL-AWARE heads (consume the un-pooled [B,T',S',D] token grid; the
    `pool_spatial=False` extraction keeps the S'=(res/patch)^2 spatial tokens that
    the default heads mean-pool away -- *where* in the vocal tract the signal sits
    is phonetically informative). CE only.
      tcn_spatial   -- learned attention-pool over S' per temporal step -> [B,T',D],
                       then the same kernel-3 TCN over time (mean -> learned pool).
      attentive     -- V-JEPA's exact AttentivePooler cross-attention block pools S'
                       per temporal step -> [B,T',D] -> linear (no temporal mixing).
    """

    def __init__(self, dim, num_classes, kind="tcn", hidden=512,
                 layers=2, heads=8, dropout=0.1, max_len=1024):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(dim, num_classes)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                     nn.GELU(), nn.Linear(hidden, num_classes))
        elif kind == "tcn":          # depthwise-ish temporal context, kernel 3 x2
            self.norm = nn.LayerNorm(dim)
            self.c1 = nn.Conv1d(dim, hidden, 3, padding=1)
            self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1)
            self.head = nn.Linear(hidden, num_classes)
        elif kind == "lstm":         # bi-LSTM over the T' tokens
            self.norm = nn.LayerNorm(dim)
            self.rnn = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                               bidirectional=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(2 * hidden, num_classes)
        elif kind == "transformer":  # self-attention encoder over the T' tokens
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, hidden)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
            nn.init.trunc_normal_(self.pos, std=0.02)
            enc = nn.TransformerEncoderLayer(
                hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(hidden, num_classes)
        elif kind == "tcn_spatial":  # learned attn-pool over S' per t, then TCN over t
            self.sp_norm = nn.LayerNorm(dim)
            self.sp_score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(),
                                          nn.Linear(dim, 1))   # additive attn weights
            self.norm = nn.LayerNorm(dim)
            self.c1 = nn.Conv1d(dim, hidden, 3, padding=1)
            self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1)
            self.head = nn.Linear(hidden, num_classes)
        elif kind == "attentive":    # V-JEPA AttentivePooler over S' per t -> classify
            from src.models.attentive_pooler import AttentivePooler
            self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                          num_heads=heads, mlp_ratio=4.0, depth=1)
            self.head = nn.Linear(dim, num_classes)
        else:
            raise ValueError(kind)

    def forward(self, x):                                  # x: [B,T',D]
        if self.kind in ("linear", "mlp"):
            return self.net(x)
        if self.kind == "tcn":
            h = self.norm(x).transpose(1, 2)               # [B,D,T']
            h = torch.relu(self.c1(h)); h = torch.relu(self.c2(h))
            return self.head(h.transpose(1, 2))            # [B,T',C]
        if self.kind == "lstm":
            h, _ = self.rnn(self.norm(x))                  # [B,T',2H]
            return self.head(h)
        if self.kind == "tcn_spatial":                     # x: [B,T',S',D]
            w = self.sp_score(self.sp_norm(x)).softmax(2)  # [B,T',S',1] over S'
            x = (x * w).sum(2)                             # [B,T',D] attn-pooled spatial
            h = self.norm(x).transpose(1, 2)               # [B,D,T']
            h = torch.relu(self.c1(h)); h = torch.relu(self.c2(h))
            return self.head(h.transpose(1, 2))            # [B,T',C]
        if self.kind == "attentive":                       # x: [B,T',S',D]
            B, T, S, D = x.shape
            q = self.pooler(x.reshape(B * T, S, D)).squeeze(1)  # [B*T,D]
            return self.head(q.reshape(B, T, D))                # [B,T',C]
        # transformer
        h = self.proj(self.norm(x)) + self.pos[:, : x.shape[1]]
        return self.head(self.encoder(h))                  # [B,T',C]


class _FeatDS(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats, self.labels = feats, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(np.array(self.feats[i], dtype=np.float32)),
                torch.from_numpy(np.array(self.labels[i])))


@torch.no_grad()
def predict(clf, feats, device, bs=128):
    clf.eval(); out = []
    if feats.ndim == 4:                      # un-pooled [N,T',S',D] is ~S'x bigger
        bs = min(bs, 32)
    for i in range(0, len(feats), bs):
        x = torch.from_numpy(np.array(feats[i:i + bs], dtype=np.float32)).to(device)
        out.append(clf(x).argmax(-1).cpu().numpy())
    return np.concatenate(out)                              # [N,T']


def evaluate(pred, labels, meta, ref_seqs, num_classes, drop=(P.SIL_IDX,)):
    """Frame-level kappa/acc (pooled tokens) + per-utterance PER."""
    flat_p = pred.reshape(-1); flat_t = labels.reshape(-1)
    kappa = P.cohen_kappa(flat_t, flat_p, num_classes)
    facc = P.frame_accuracy(flat_t, flat_p)
    # reassemble per-utterance token streams in chunk order, dropping padded
    # (label==IGNORE) tail tokens so they don't inject spurious phonemes into PER
    by_utt = {}
    for n, (utt, chunk, _) in enumerate(meta):
        valid = labels[n] != P.IGNORE_INDEX
        by_utt.setdefault(int(utt), []).append((int(chunk), pred[n][valid]))
    tot_err = tot_ref = 0; pers = []
    for utt, seq in by_utt.items():
        seq.sort(key=lambda z: z[0])
        stream = np.concatenate([s for _, s in seq])
        hyp = P.collapse_sequence(stream, drop=tuple(drop))
        ref = ref_seqs.get(utt, [])
        per, nref = P.phoneme_error_rate(ref, hyp)
        if nref > 0:
            tot_err += P.edit_distance(ref, hyp); tot_ref += nref; pers.append(per)
    return {
        "kappa": round(kappa, 4), "frame_acc": round(facc, 4),
        "per_micro": round(tot_err / max(1, tot_ref), 4),
        "per_macro": round(float(np.mean(pers)) if pers else 1.0, 4),
        "n_utt": len(pers),
    }


def train_probe(cfg, ftr, ltr, fva, lva, mva, fte, lte, mte,
                ref_va, ref_te, device, num_classes, drop=(P.SIL_IDX,)):
    pc = cfg["probe"]
    dim = ftr.shape[-1]
    clf = TokenProbe(dim, num_classes, kind=pc.get("type", "tcn"),
                     hidden=pc.get("hidden", 512), layers=pc.get("layers", 2),
                     heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    lossf = nn.CrossEntropyLoss(ignore_index=P.IGNORE_INDEX)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    loader = torch.utils.data.DataLoader(
        _FeatDS(ftr, ltr), batch_size=pc.get("batch_size", 32), shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4), drop_last=False)
    best = {"val_kappa": -1.0}
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(clf(x).reshape(-1, num_classes), y.reshape(-1))
            loss.backward(); opt.step(); run += float(loss); nb += 1
        vm = evaluate(predict(clf, fva, device), lva, mva, ref_va, num_classes, drop)
        if vm["kappa"] > best["val_kappa"]:
            tm = evaluate(predict(clf, fte, device), lte, mte, ref_te, num_classes, drop)
            best = {"epoch": ep + 1, "val_kappa": vm["kappa"], "val": vm, "test": tm,
                    "train_loss": run / max(1, nb)}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[probe e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} lr={lr:.2e} "
                  f"val κ={vm['kappa']:.3f} PERμ={vm['per_micro']:.3f} "
                  f"best_val_κ={best['val_kappa']:.3f}")
    return best


# --------------------------------------------------------------------------- #
# CTC (alignment-free) probe -- per-utterance, sequence loss (Plan B.4 loss axis)
# --------------------------------------------------------------------------- #
def _assemble_utts(feats, labels, meta):
    """Cached per-chunk arrays -> {utt: float32 [T_u, D]} of valid (non-pad)
    tokens in chunk order. Drops padded tail tokens (label == IGNORE_INDEX)."""
    by_utt = {}
    for n, (utt, chunk, _) in enumerate(meta):
        valid = labels[n] != P.IGNORE_INDEX
        if not valid.any():
            continue
        by_utt.setdefault(int(utt), []).append(
            (int(chunk), np.asarray(feats[n], dtype=np.float32)[valid]))
    out = {}
    for utt, parts in by_utt.items():
        parts.sort(key=lambda z: z[0])
        out[utt] = np.concatenate([p for _, p in parts], axis=0)   # [T_u, D]
    return out


class _UttDS(torch.utils.data.Dataset):
    """Per-utterance features + collapsed phoneme target (CTC). Only utts with a
    non-empty reference are kept."""

    def __init__(self, utt_feats, refs):
        self.items = [(u, utt_feats[u], refs.get(u, [])) for u in utt_feats
                      if len(refs.get(u, [])) > 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        u, f, r = self.items[i]
        return torch.from_numpy(f), torch.tensor(r, dtype=torch.long), u


def _ctc_collate(batch):
    feats, tgts, utts = zip(*batch)
    in_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tgt_lens = torch.tensor([t.shape[0] for t in tgts], dtype=torch.long)
    T, D = int(in_lens.max()), feats[0].shape[-1]
    x = torch.zeros(len(feats), T, D)
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f
    targets = torch.cat(tgts) if tgts else torch.zeros(0, dtype=torch.long)
    return x, in_lens, targets, tgt_lens, list(utts)


def _greedy_ctc(ids, blank, drop):
    """Greedy CTC decode: collapse repeats, drop blank, then drop the collapse set."""
    out, prev = [], blank
    for t in ids:
        t = int(t)
        if t != prev and t != blank:
            out.append(t)
        prev = t
    return [o for o in out if o not in drop]


@torch.no_grad()
def _ctc_eval(clf, utt_feats, refs, device, blank, drop, bs=16):
    clf.eval()
    loader = torch.utils.data.DataLoader(_UttDS(utt_feats, refs), batch_size=bs,
                                         shuffle=False, collate_fn=_ctc_collate)
    tot_err = tot_ref = 0; pers = []
    for x, in_lens, _t, _tl, utts in loader:
        ids = clf(x.to(device)).argmax(-1).cpu().numpy()      # [B,T] over C+1
        for b, u in enumerate(utts):
            hyp = _greedy_ctc(ids[b][: int(in_lens[b])], blank, drop)
            ref = refs.get(u, [])
            _, nref = P.phoneme_error_rate(ref, hyp)
            if nref > 0:
                tot_err += P.edit_distance(ref, hyp); tot_ref += nref
                pers.append(P.edit_distance(ref, hyp) / nref)
    return {"per_micro": round(tot_err / max(1, tot_ref), 4),
            "per_macro": round(float(np.mean(pers)) if pers else 1.0, 4),
            "n_utt": len(pers)}


def train_probe_ctc(cfg, utt_tr, ref_tr, utt_va, ref_va, utt_te, ref_te,
                    device, num_classes, drop=()):
    """Train the probe with CTC (blank = num_classes). PER-primary; no kappa
    (undefined without forced alignment). Model-selection on val PER (lower=better)."""
    pc = cfg["probe"]
    dim = next(iter(utt_tr.values())).shape[-1]
    blank = num_classes
    clf = TokenProbe(dim, num_classes + 1, kind=pc.get("type", "tcn"),
                     hidden=pc.get("hidden", 512), layers=pc.get("layers", 2),
                     heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    bs = pc.get("batch_size", 16)
    loader = torch.utils.data.DataLoader(
        _UttDS(utt_tr, ref_tr), batch_size=bs, shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2), collate_fn=_ctc_collate)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    best = {"val_per": 2.0}
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0
        for x, in_lens, targets, tgt_lens, _ in loader:
            x, targets = x.to(device), targets.to(device)
            logp = clf(x).log_softmax(-1).transpose(0, 1)      # [T,B,C+1]
            loss = ctc(logp, targets, in_lens, tgt_lens)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); nb += 1
        vm = _ctc_eval(clf, utt_va, ref_va, device, blank, drop, bs)
        if vm["per_micro"] < best["val_per"]:
            tm = _ctc_eval(clf, utt_te, ref_te, device, blank, drop, bs)
            best = {"epoch": ep + 1, "val_per": vm["per_micro"], "val": vm,
                    "test": tm, "train_loss": run / max(1, nb)}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[probe-ctc e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} lr={lr:.2e} "
                  f"val PERµ={vm['per_micro']:.3f} best_val_PERµ={best['val_per']:.3f}")
    return best


# --------------------------------------------------------------------------- #
def run(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "bfloat16").lower(), torch.float32)
    print(f"[ph-eval] device={device} dtype={dtype} kind={cfg['data'].get('kind','usc_lss')}")

    # spatial-aware heads consume the un-pooled [B,T',S',D] grid (separate cache);
    # all other heads use the mean-over-S' [B,T',D] features (default).
    SPATIAL_HEADS = {"tcn_spatial", "attentive"}
    ptype = cfg["probe"].get("type", "tcn")
    cfg["data"]["pool_spatial"] = ptype not in SPATIAL_HEADS
    if ptype in SPATIAL_HEADS:
        if cfg["probe"].get("loss", "ce") == "ctc":
            raise SystemExit("[ph-eval] spatial heads (tcn_spatial/attentive) are CE-only")
        print(f"[ph-eval] spatial-aware probe '{ptype}': caching un-pooled token grid")

    encoder = load_frozen_encoder(cfg, device)
    ftr, ltr, mtr = extract(encoder, cfg, "train", device, dtype)
    fva, lva, mva = extract(encoder, cfg, "val", device, dtype)
    fte, lte, mte = extract(encoder, cfg, "test", device, dtype)
    del encoder; torch.cuda.empty_cache()

    # reference phoneme sequences + label space (gold or pseudo) from the dataset
    val_ds = build_dataset(cfg, "val")[0]
    test_ds = build_dataset(cfg, "test")[0]
    ref_va, ref_te = val_ds.reference_sequences(), test_ds.reference_sequences()
    num_classes = val_ds.num_classes
    drop = tuple(val_ds.collapse_drop)
    print(f"[ph-eval] tokens train/val/test = {len(ltr)}/{len(lva)}/{len(lte)} "
          f"clips; T'={ltr.shape[1]}; num_classes={num_classes} drop={drop}")

    loss_kind = cfg["probe"].get("loss", "ce")
    if loss_kind == "ctc":
        ref_tr = build_dataset(cfg, "train")[0].reference_sequences()
        utt_tr = _assemble_utts(ftr, ltr, mtr)
        utt_va = _assemble_utts(fva, lva, mva)
        utt_te = _assemble_utts(fte, lte, mte)
        print(f"[ph-eval] CTC mode: utts train/val/test = "
              f"{len(utt_tr)}/{len(utt_va)}/{len(utt_te)}; blank={num_classes}")
        best = train_probe_ctc(cfg, utt_tr, ref_tr, utt_va, ref_va, utt_te, ref_te,
                               device, num_classes, drop)
    else:
        best = train_probe(cfg, ftr, ltr, fva, lva, mva, fte, lte, mte,
                           ref_va, ref_te, device, num_classes, drop)
    out = {"encoder": cfg["encoder"].get("spec", "pretrained"),
           "kind": cfg["data"].get("kind", "usc_lss"),
           "probe": cfg["probe"].get("type", "tcn"), "loss": loss_kind,
           "spatial_size": cfg["data"]["spatial_size"], "seed": seed, **best}
    print("\n===== PHONEME RESULT =====")
    print(json.dumps(out, indent=2))
    rp = os.path.join(meta["out"], f"phoneme_{cfg['data'].get('kind','usc_lss')}_"
                      f"{_tag(cfg,'train')[0]}_{cfg['probe'].get('type','tcn')}_{loss_kind}"
                      f"_s{seed}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[ph-eval] wrote {rp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--model", default=None,
                    help="image-baseline encoder (clip|siglip|dinov2|vitl|resnet "
                         "or a full timm name); sets encoder.type=image_baseline")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=None, help="override data.batch_size")
    ap.add_argument("--probe", default=None,
                    choices=["linear", "mlp", "tcn", "lstm", "transformer",
                             "tcn_spatial", "attentive"])
    ap.add_argument("--loss", default=None, choices=["ce", "ctc"],
                    help="probe training loss: ce (per-token, kappa+PER) | ctc (PER-only)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override meta.seed (probe init/shuffle only; the frozen "
                         "feature cache is seed-independent, so multi-seed reuses it)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    if args.model is not None:
        cfg["encoder"]["type"] = "image_baseline"
        cfg["encoder"]["model"] = args.model
    if args.encoder is not None:
        cfg["encoder"]["spec"] = args.encoder
    if args.tag is not None:
        cfg["meta"]["tag"] = args.tag
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch
    if args.probe is not None:
        cfg["probe"]["type"] = args.probe
    if args.loss is not None:
        cfg["probe"]["loss"] = args.loss
    run(cfg)


if __name__ == "__main__":
    main()
