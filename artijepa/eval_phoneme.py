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
import mmap
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
    if ec.get("type") == "videomae":
        # Frozen 3-D VideoMAE video encoder (the video-SSL baseline, vs the per-frame
        # image baselines). Native geometry (224px / 16 frames / tubelet 2) is fixed by
        # the model's positional embeddings, so it OVERRIDES the config's
        # spatial_size/frames_per_clip/tubelet_size. pool_spatial (set by run() from the
        # probe head) selects pooled [B,T',D] vs the un-pooled [B,T',S',D] grid.
        from artijepa.videomae_baseline import VideoMAEEncoder, VIDEOMAE_LARGE
        enc = VideoMAEEncoder(ec.get("model", VIDEOMAE_LARGE),
                              pool_spatial=d.get("pool_spatial", True),
                              grid_cap=ec.get("grid_cap", 16),
                              checkpoint=ec.get("checkpoint")).to(device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        d["spatial_size"] = enc.backbone.input_size            # 224
        d["frames_per_clip"] = enc.backbone.num_frames          # 16 (native, fixed)
        d["tubelet_size"] = enc.backbone.tubelet                # 2 -> T'=8 tokens
        d["intensity_norm"] = "minmax"
        d.pop("grayscale_stats", None)
        ec["spec"] = "videomae_rtmri" if ec.get("checkpoint") else "videomae_large"  # feature cache/tag key
        mode = "pooled [B,T',D]" if enc.backbone.pool_spatial else \
            f"un-pooled grid [B,T',S'<= {enc.backbone.grid_cap}^2,D]"
        print(f"[ph-eval] videomae {enc.backbone.name} D={enc.backbone.embed_dim} "
              f"input={enc.backbone.input_size} 16f/tubelet{enc.backbone.tubelet} "
              f"minmax->[0,1] -> 3-D tubelet, {mode}")
        return enc
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
        if bi % 10 == 0:
            # Bound RSS: for the spatial (un-pooled) grid the memmap is tens-hundreds of
            # GB, and its written pages stay resident and are charged to the job's memory
            # cgroup -> OOM under a small --mem. Flush to disk, then MADV_DONTNEED the
            # mapping so those pages are reclaimed (re-read from disk only if accessed;
            # extraction never reads back, so RSS stays flat). NOTE: fadvise() on a
            # separate fd does NOT reclaim pages still mapped by the memmap -- madvise on
            # the mmap itself does. Does NOT change the on-disk output.
            feats.flush()
            try:
                feats._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
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
      attentive_lstm-- same AttentivePooler spatial pool -> bi-LSTM over time -> linear.
                       CE *and* CTC (the only spatial head allowed with CTC): it is
                       trained in UTTERANCE mode (whole-utterance sequences, see
                       _UttSpatialDS) so the recurrence and the alignment-free loss
                       both see the full utterance rather than a 1.28 s chunk.
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
            from artijepa._vendor.src.models.attentive_pooler import AttentivePooler
            self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                          num_heads=heads, mlp_ratio=4.0, depth=1)
            self.head = nn.Linear(dim, num_classes)
        elif kind == "attentive_lstm":   # AttentivePooler over S' per t, then bi-LSTM
            from artijepa._vendor.src.models.attentive_pooler import AttentivePooler
            self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                          num_heads=heads, mlp_ratio=4.0, depth=1)
            self.norm = nn.LayerNorm(dim)
            self.rnn = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                               bidirectional=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(2 * hidden, num_classes)
        else:
            raise ValueError(kind)

    def forward(self, x, lens=None):                       # x: [B,T',D]
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
        if self.kind == "attentive_lstm":                  # x: [B,T',S',D]
            B, T, S, D = x.shape
            q = self.pooler(x.reshape(B * T, S, D)).squeeze(1)  # [B*T,D]
            h = self.norm(q.reshape(B, T, D))                   # [B,T',D]
            if lens is None:
                h, _ = self.rnn(h)
            else:
                # Pad tokens must not enter the recurrence: the backward direction
                # would consume them FIRST and leak zeros into every real timestep.
                p = nn.utils.rnn.pack_padded_sequence(
                    h, lens.cpu(), batch_first=True, enforce_sorted=False)
                h, _ = self.rnn(p)
                h, _ = nn.utils.rnn.pad_packed_sequence(
                    h, batch_first=True, total_length=T)         # [B,T',2H]
            return self.head(h)                                 # [B,T',C]
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


class _FeatStream(torch.utils.data.IterableDataset):
    """Streaming, bounded-RAM TRAIN loader for the spatial (un-pooled) feature cache.

    A full random-shuffle pass over that cache re-reads tens-hundreds of GB from disk
    every epoch at random-access speed (~tens of MB/s) when it can't fit in page cache
    under a small --mem job -- the probe's I/O wall. Instead, each DataLoader worker
    reads ONE CONTIGUOUS block of rows SEQUENTIALLY (readahead-friendly, ~bandwidth
    speed; on a striped FS the workers' concurrent streams parallelise across targets)
    and emits them through a small in-RAM shuffle buffer of `buf` rows. Resident RAM is
    bounded at ~= workers * buf * row_bytes (f16). Global mixing comes from the workers
    streaming different file regions at once + the local buffer. Optional `max_samples`
    (= probe.ipe * batch_size) caps rows/epoch for a fixed iterations-per-epoch budget.
    Val/test keep the deterministic full pass (predict()/_FeatDS), so metrics are exact.
    """
    def __init__(self, feats, labels, buf=96, max_samples=None):
        self.feats, self.labels = feats, labels
        self.buf = max(1, int(buf))
        self.max_samples = max_samples

    def __iter__(self):
        N = len(self.labels)
        wi = torch.utils.data.get_worker_info()
        nw = wi.num_workers if wi else 1
        wid = wi.id if wi else 0
        lo, hi = (N * wid) // nw, (N * (wid + 1)) // nw      # this worker's block
        cap = None if self.max_samples is None else max(1, -(-self.max_samples // nw))
        buf, out = [], 0
        for i in range(lo, hi):
            buf.append((np.array(self.feats[i], dtype=np.float16),   # sequential disk read
                        np.asarray(self.labels[i])))
            if len(buf) >= self.buf:
                f, l = buf.pop(np.random.randint(len(buf)))          # local shuffle
                yield torch.from_numpy(f.astype(np.float32)), torch.from_numpy(l.copy())
                out += 1
                if cap is not None and out >= cap:
                    return
        while buf:                                                   # drain remainder
            f, l = buf.pop(np.random.randint(len(buf)))
            yield torch.from_numpy(f.astype(np.float32)), torch.from_numpy(l.copy())
            out += 1
            if cap is not None and out >= cap:
                return


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


def _cpu_state(model):
    """Detached CPU copy of a model's params, for a best-epoch probe snapshot."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_probe(path, device="cpu"):
    """Reconstruct a trained TokenProbe from a probe checkpoint written by run().

    Returns (probe.eval(), meta_dict). The frozen features it was trained on are
    reproducible from meta['feature_tag'] (see extract's cache dir)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    clf = TokenProbe(ck["dim"], ck["clf_classes"], kind=ck["probe_kind"],
                     hidden=ck["hidden"], layers=ck["layers"],
                     heads=ck["heads"], dropout=ck["dropout"])
    clf.load_state_dict(ck["probe_state"])
    return clf.to(device).eval(), ck


def _focal_loss(logits, target, gamma=2.0, ignore_index=P.IGNORE_INDEX):
    """Multi-class focal loss (Lin et al. 2017): per-frame CE reweighted by (1-p_t)^gamma
    so the probe stops pouring gradient into easy, high-frequency frames (silence, the
    dominant vowels) and instead learns the hard / rare phonemes. This is the CE-FAMILY
    angle on the OOD over-prediction (test_lss PERµ>1.0) that CTC attacks alignment-free:
    a calmer per-frame posterior emits fewer spurious phonemes at collapse time.
    gamma=0 reduces exactly to plain CE. Frame-level, so kappa is defined (unlike CTC)."""
    ce = nn.functional.cross_entropy(logits, target, ignore_index=ignore_index,
                                     reduction="none")          # [N]; 0 at ignored frames
    pt = torch.exp(-ce)                                          # prob of the true class
    fl = ((1.0 - pt) ** gamma) * ce
    mask = target != ignore_index                               # mean over VALID frames only
    return fl[mask].mean() if bool(mask.any()) else fl.sum() * 0.0


def train_probe(cfg, ftr, ltr, fva, lva, mva, fte, lte, mte,
                ref_va, ref_te, device, num_classes, drop=(P.SIL_IDX,),
                extra_tests=None):
    """extra_tests: optional {name: (feats, labels, meta, ref_seqs)} evaluated with
    the best-val probe alongside the primary `test` split (e.g. a cross-domain
    held-out set). Results land in best['tests'][name]."""
    pc = cfg["probe"]
    dim = ftr.shape[-1]
    clf = TokenProbe(dim, num_classes, kind=pc.get("type", "tcn"),
                     hidden=pc.get("hidden", 512), layers=pc.get("layers", 2),
                     heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    lossf = nn.CrossEntropyLoss(ignore_index=P.IGNORE_INDEX)
    focal = (pc.get("loss", "ce") == "focal"); fgamma = pc.get("focal_gamma", 2.0)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    # TRAIN loader: bounded-RAM sequential streaming (see _FeatStream). Sequential per-
    # worker reads + concurrent workers recover FS bandwidth vs the ~tens-of-MB/s random
    # full-shuffle pass; probe.ipe (micro-batches/epoch) optionally caps rows/epoch.
    p_workers = pc.get("workers", 8)
    train_stream = _FeatStream(
        ftr, ltr, buf=pc.get("buf", 96),
        max_samples=(pc["ipe"] * pc.get("batch_size", 32)) if pc.get("ipe") else None)
    loader = torch.utils.data.DataLoader(
        train_stream, batch_size=pc.get("batch_size", 32),
        num_workers=p_workers, persistent_workers=(p_workers > 0),
        drop_last=False)
    best = {"val_kappa": -1.0}; best_state = None; history = []
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = clf(x).reshape(-1, num_classes); tgt = y.reshape(-1)
            loss = _focal_loss(logits, tgt, fgamma) if focal else lossf(logits, tgt)
            loss.backward(); opt.step(); run += float(loss); nb += 1
        tr_loss = run / max(1, nb)
        vm = evaluate(predict(clf, fva, device), lva, mva, ref_va, num_classes, drop)
        history.append({"epoch": ep + 1, "lr": round(lr, 6),
                        "train_loss": round(tr_loss, 4), "val_kappa": vm["kappa"],
                        "val_per_micro": vm["per_micro"], "val_per_macro": vm["per_macro"]})
        if vm["kappa"] > best["val_kappa"]:
            tm = evaluate(predict(clf, fte, device), lte, mte, ref_te, num_classes, drop)
            best = {"epoch": ep + 1, "val_kappa": vm["kappa"], "val": vm, "test": tm,
                    "train_loss": tr_loss}
            if extra_tests:
                best["tests"] = {
                    nm: evaluate(predict(clf, ef, device), el, em, er, num_classes, drop)
                    for nm, (ef, el, em, er) in extra_tests.items()}
            best_state = _cpu_state(clf)            # snapshot best-val weights
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[probe e{ep+1}/{epochs}] loss={tr_loss:.3f} lr={lr:.2e} "
                  f"val κ={vm['kappa']:.3f} PERμ={vm['per_micro']:.3f} "
                  f"best_val_κ={best['val_kappa']:.3f}")
    return best, best_state, history


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
                    device, num_classes, drop=(), extra_tests=None):
    """Train the probe with CTC (blank = num_classes). PER-primary; no kappa
    (undefined without forced alignment). Model-selection on val PER (lower=better).
    extra_tests: optional {name: (utt_feats, ref_seqs)} evaluated with the best-val
    probe alongside the primary test split. Results land in best['tests'][name]."""
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
    best = {"val_per": 2.0}; best_state = None; history = []
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
        tr_loss = run / max(1, nb)
        vm = _ctc_eval(clf, utt_va, ref_va, device, blank, drop, bs)
        history.append({"epoch": ep + 1, "lr": round(lr, 6),
                        "train_loss": round(tr_loss, 4),
                        "val_per_micro": vm["per_micro"], "val_per_macro": vm["per_macro"]})
        if vm["per_micro"] < best["val_per"]:
            tm = _ctc_eval(clf, utt_te, ref_te, device, blank, drop, bs)
            best = {"epoch": ep + 1, "val_per": vm["per_micro"], "val": vm,
                    "test": tm, "train_loss": tr_loss}
            if extra_tests:
                best["tests"] = {
                    nm: _ctc_eval(clf, uf, rf, device, blank, drop, bs)
                    for nm, (uf, rf) in extra_tests.items()}
            best_state = _cpu_state(clf)            # snapshot best-val weights
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[probe-ctc e{ep+1}/{epochs}] loss={tr_loss:.3f} lr={lr:.2e} "
                  f"val PERµ={vm['per_micro']:.3f} best_val_PERµ={best['val_per']:.3f}")
    return best, best_state, history


# --------------------------------------------------------------------------- #
# UTTERANCE mode -- whole-utterance sequences over the SPATIAL cache (Phase 2)
#
# Why a second utterance path when train_probe_ctc already exists: that one calls
# _assemble_utts, which materialises EVERY utterance in RAM. That is fine for the
# pooled cache (tens of MB) but the spatial cache is ~149 GiB -> instant OOM. Here
# each utterance is read from the memmap on demand; rows of one utterance are
# CONTIGUOUS in the cache (extract() writes in manifest order, chunk-ordered), so
# one item = one ~430 MB SEQUENTIAL read rather than T_u scattered row reads --
# the same access pattern that fixed the _FeatStream I/O wall (docs/phonePred.md §4.1).
#
# Both CE and CTC train through this path so the Phase-2 CE-vs-CTC comparison varies
# ONLY the loss: same head, same whole-utterance temporal context. (Phase 1's
# `attentive` x CE keeps its original chunk-mode path untouched.)
# --------------------------------------------------------------------------- #
def _utt_runs(meta):
    """-> [(utt, lo, hi)] one contiguous row run per utterance, in chunk order."""
    runs = []
    for n, m in enumerate(meta):
        u = int(m[0])
        if runs and runs[-1][0] == u and runs[-1][2] == n:
            runs[-1][2] = n + 1                      # extend the current run
        else:
            runs.append([u, n, n + 1])
    utts = [r[0] for r in runs]
    if len(set(utts)) != len(utts):                  # an utt split across >1 run
        raise SystemExit("[ph-eval] utterance rows are not contiguous in this cache; "
                         "utterance-mode heads require a contiguous cache")
    return [tuple(r) for r in runs]


class _UttSpatialDS(torch.utils.data.Dataset):
    """One item = one whole utterance read on demand from the feature cache:
    (feats [T_u,S',D] f16, labels [T_u], ref [L], utt). Padded tail tokens
    (label == IGNORE_INDEX) are dropped, so T_u is the true token count."""

    def __init__(self, feats, labels, meta, refs, require_ref=True):
        self.feats, self.labels, self.refs = feats, labels, refs
        runs = _utt_runs(meta)
        # CTC needs a non-empty reference; CE would also score nothing without one.
        self.runs = [r for r in runs
                     if not require_ref or len(refs.get(r[0], [])) > 0]
        lab = np.asarray(labels)
        self.lengths = [int((lab[lo:hi] != P.IGNORE_INDEX).sum())
                        for _u, lo, hi in self.runs]

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        utt, lo, hi = self.runs[i]
        blk = np.array(self.feats[lo:hi], dtype=np.float16)   # ONE sequential read
        lab = np.asarray(self.labels[lo:hi])
        valid = lab != P.IGNORE_INDEX                         # [n,T'] over rows+time
        return (torch.from_numpy(blk[valid]),                 # [T_u,S',D] (f16)
                torch.from_numpy(lab[valid].astype(np.int64)),
                torch.tensor(self.refs.get(utt, []), dtype=torch.long),
                int(utt))


def _utt_collate(batch):
    feats, labs, refs, utts = zip(*batch)
    in_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tgt_lens = torch.tensor([r.shape[0] for r in refs], dtype=torch.long)
    T = int(in_lens.max())
    x = torch.zeros((len(feats), T) + tuple(feats[0].shape[1:]), dtype=feats[0].dtype)
    y = torch.full((len(feats), T), P.IGNORE_INDEX, dtype=torch.long)
    for i, (f, l) in enumerate(zip(feats, labs)):
        x[i, : f.shape[0]] = f
        y[i, : l.shape[0]] = l
    targets = torch.cat(refs) if refs else torch.zeros(0, dtype=torch.long)
    return x, y, in_lens, targets, tgt_lens, list(utts)


class _TokenBudget(torch.utils.data.Sampler):
    """Batch whole utterances up to ~max_tokens temporal tokens (always >=1 utt).

    Utterances here run 24-123 chunks (T_u ~ 400-2000 tokens) and the spatial grid
    makes activations scale with the batch's token count * S', so a fixed utterance
    batch_size would swing peak memory ~5x with utterance length. Budgeting tokens
    keeps it flat.

    The budget is on the PADDED size len(batch) * max(T_u), NOT sum(T_u): _utt_collate
    pads every utterance up to the batch's longest, so the padded size is what actually
    gets allocated. They coincide only for equal-length batches -- on test_lss, whose
    utterances span 16-560 tokens, budgeting sum() packed twelve 80-token utts next to
    a 560-token one and allocated [12,560,256,1024] f16 = 3.5 GB per batch (x workers
    x prefetch -> host OOM). Padded budgeting also stops the wasted compute on pad.
    """

    def __init__(self, lengths, max_tokens, shuffle=True, seed=0):
        self.lengths, self.max_tokens = list(lengths), int(max_tokens)
        self.shuffle, self.seed, self.epoch = shuffle, int(seed), 0

    def __iter__(self):
        order = np.arange(len(self.lengths))
        if self.shuffle:
            np.random.RandomState(self.seed + self.epoch).shuffle(order)
        batch, mx = [], 0
        for i in order:
            L = int(self.lengths[i])
            m = max(mx, L)
            if batch and (len(batch) + 1) * m > self.max_tokens:
                yield batch; batch, mx = [int(i)], L      # i starts the next batch
            else:
                batch.append(int(i)); mx = m
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(1 for _ in iter(self)))


def _utt_loader(ds, max_tokens, workers, shuffle, seed=0, persistent=False,
                prefetch_factor=None):
    # persistent only for the long-lived TRAIN loader; the eval loaders are rebuilt
    # every epoch, and persistent workers there would leak a worker pool per call.
    # host RAM ~= workers * prefetch_factor * padded_batch_bytes; the eval loaders pass
    # prefetch_factor=1 (see _utt_eval) so a big test_lss batch can't stack 2-deep per
    # worker and OOM the cgroup (the Jul-15 crash: 4 workers x 2 prefetch x ~1 GB batch).
    kw = {}
    if workers > 0 and prefetch_factor is not None:
        kw["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(
        ds, batch_sampler=_TokenBudget(ds.lengths, max_tokens, shuffle, seed),
        num_workers=workers, collate_fn=_utt_collate,
        persistent_workers=(persistent and workers > 0), **kw)


@torch.no_grad()
def _utt_eval(clf, ds, device, loss_kind, num_classes, drop, max_tokens, workers,
              amp):
    """CE -> kappa + frame-acc + PER (argmax->collapse). CTC -> PER only (greedy
    CTC decode). Deterministic full pass; frames accumulate in RAM (tokens are
    ~1e5 ints per split, the [S',D] grid never leaves the GPU step)."""
    clf.eval()
    blank = num_classes
    tot_err = tot_ref = 0; pers = []; ps = []; ts = []
    for x, y, in_lens, _tg, _tl, utts in _utt_loader(ds, max_tokens, workers, False,
                                                      prefetch_factor=1):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = clf(x, lens=in_lens)
        ids = logits.float().argmax(-1).cpu().numpy()          # [B,T]
        for b, u in enumerate(utts):
            n = int(in_lens[b])
            seq = ids[b][:n]
            ref = ds.refs.get(u, [])
            if loss_kind == "ctc":
                hyp = _greedy_ctc(seq, blank, drop)
            else:
                hyp = P.collapse_sequence(seq, drop=tuple(drop))
                ps.append(seq); ts.append(y[b][:n].numpy())
            _, nref = P.phoneme_error_rate(ref, hyp)
            if nref > 0:
                e = P.edit_distance(ref, hyp)
                tot_err += e; tot_ref += nref; pers.append(e / nref)
    out = {"per_micro": round(tot_err / max(1, tot_ref), 4),
           "per_macro": round(float(np.mean(pers)) if pers else 1.0, 4),
           "n_utt": len(pers)}
    if loss_kind != "ctc":                    # frame metrics need an alignment
        fp = np.concatenate(ps); ft = np.concatenate(ts)
        out["kappa"] = P.cohen_kappa(ft, fp, num_classes)
        out["frame_acc"] = P.frame_accuracy(ft, fp)
    return out


def train_probe_utt(cfg, ds_tr, ds_va, ds_te, device, num_classes, drop=(),
                    extra_tests=None, loss_kind="ce"):
    """Utterance-mode probe over the spatial cache: CE, CTC, or FOCAL (Phase 2 loss axis).

    Model-selection: CE/FOCAL on val kappa (higher=better, comparable to Phase 1); CTC on
    val PER (lower=better -- kappa is undefined without a forced alignment). Focal is a
    per-frame loss like CE (same head, blank-free, kappa+PER reported), differing only in
    the (1-p_t)^gamma reweighting -- so it shares CE's entire eval/selection path here.
    extra_tests: {name: _UttSpatialDS} scored with the same best-val probe.
    """
    pc = cfg["probe"]
    dim = int(ds_tr.feats.shape[-1])
    blank = num_classes
    nc = num_classes + (1 if loss_kind == "ctc" else 0)     # CTC adds the blank
    clf = TokenProbe(dim, nc, kind=pc.get("type", "attentive_lstm"),
                     hidden=pc.get("hidden", 512), layers=pc.get("layers", 2),
                     heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    ce = nn.CrossEntropyLoss(ignore_index=P.IGNORE_INDEX)
    fgamma = pc.get("focal_gamma", 2.0)          # focal: CE reweighted by (1-p_t)^gamma
    mt = pc.get("utt_max_tokens", 1024)
    # Separate from probe.workers (tuned for the chunk stream's 8 MB rows): one utt-mode
    # item is a ~0.5 GB read and a collated batch is up to mt*S'*D f16 (~0.54 GB at
    # mt=1024), and collate runs IN the worker -> host RAM ~= workers * prefetch(2) *
    # batch_bytes. Keep this modest; the read is FS-bandwidth-bound, not worker-bound.
    workers = pc.get("utt_workers", 4)
    # Eval loaders read the big OOD test_lss set and are single-pass (order-invariant),
    # so they don't need the train loader's throughput. Fewer workers x prefetch_factor=1
    # (below) bounds their peak host RAM -- the Jul-15 OOM was the eval path stacking
    # 4 workers x 2 prefetched ~1 GB batches on top of the 64 GB cgroup.
    eval_workers = pc.get("utt_eval_workers", max(1, min(2, workers)))
    amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs = pc.get("epochs", 30); warmup = pc.get("warmup", 4); base = pc.get("lr", 1e-3)
    seed = cfg["meta"].get("seed", 0)
    loader = _utt_loader(ds_tr, mt, workers, True, seed, persistent=True)
    better = (lambda m, b: m["per_micro"] < b) if loss_kind == "ctc" else \
             (lambda m, b: m["kappa"] > b)
    best = {"val_score": (2.0 if loss_kind == "ctc" else -1.0)}
    best_state = None; history = []
    print(f"[ph-eval] utt-mode {loss_kind}: utts train/val/test = "
          f"{len(ds_tr)}/{len(ds_va)}/{len(ds_te)}; max_tokens={mt}"
          + (f"; blank={blank}" if loss_kind == "ctc" else ""))
    for ep in range(epochs):
        loader.batch_sampler.epoch = ep                  # reshuffle the batches
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0; t0 = time.time()
        for x, y, in_lens, targets, tgt_lens, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
                logits = clf(x, lens=in_lens)                  # [B,T,C(+1)]
            if loss_kind == "ctc":
                # CTC/log_softmax in fp32: the fp16 dynamic range loses the small
                # log-probs the forward-backward sums over.
                logp = logits.float().log_softmax(-1).transpose(0, 1)   # [T,B,C+1]
                loss = ctc(logp, targets.to(device), in_lens, tgt_lens)
            else:                                              # per-frame: ce | focal
                flat = logits.float().reshape(-1, nc); tgt = y.to(device).reshape(-1)
                loss = _focal_loss(flat, tgt, fgamma) if loss_kind == "focal" \
                    else ce(flat, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss); nb += 1
        tr_loss = run / max(1, nb)
        vm = _utt_eval(clf, ds_va, device, loss_kind, num_classes, drop, mt,
                       eval_workers, amp)
        score = vm["per_micro"] if loss_kind == "ctc" else vm["kappa"]
        h = {"epoch": ep + 1, "lr": round(lr, 6), "train_loss": round(tr_loss, 4),
             "val_per_micro": vm["per_micro"], "mins": round((time.time() - t0) / 60, 1)}
        if loss_kind != "ctc":
            h["val_kappa"] = vm["kappa"]
        history.append(h)
        if better(vm, best["val_score"]):
            tm = _utt_eval(clf, ds_te, device, loss_kind, num_classes, drop, mt,
                           eval_workers, amp)
            best = {"epoch": ep + 1, "val_score": score, "val": vm, "test": tm,
                    "train_loss": tr_loss}
            if extra_tests:
                best["tests"] = {
                    nm: _utt_eval(clf, d, device, loss_kind, num_classes, drop, mt,
                                  eval_workers, amp)
                    for nm, d in extra_tests.items()}
            best_state = _cpu_state(clf)
        msg = (f"val PERµ={vm['per_micro']:.3f}" if loss_kind == "ctc"
               else f"val κ={vm['kappa']:.3f} PERµ={vm['per_micro']:.3f}")
        print(f"[probe-utt-{loss_kind} e{ep+1}/{epochs}] loss={tr_loss:.3f} "
              f"lr={lr:.2e} {msg} best={best['val_score']:.3f} "
              f"({h['mins']:.1f} min)", flush=True)
    # match the chunk-mode schema: val_kappa (CE) / val_per (CTC) at the top level
    best["val_per" if loss_kind == "ctc" else "val_kappa"] = best.pop("val_score")
    return best, best_state, history


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
    SPATIAL_HEADS = {"tcn_spatial", "attentive", "attentive_lstm"}
    UTT_HEADS = {"attentive_lstm"}       # trained on whole utterances (CE and CTC)
    ptype = cfg["probe"].get("type", "tcn")
    cfg["data"]["pool_spatial"] = ptype not in SPATIAL_HEADS
    if ptype in SPATIAL_HEADS:
        if cfg["probe"].get("loss", "ce") == "ctc" and ptype not in UTT_HEADS:
            raise SystemExit(f"[ph-eval] spatial head '{ptype}' is CE-only "
                             f"(chunk-mode); use attentive_lstm for spatial + CTC")
        print(f"[ph-eval] spatial-aware probe '{ptype}': caching un-pooled token grid")

    # optional extra held-out test splits (e.g. a cross-domain set) evaluated with
    # the same best-val probe: data.extra_test_splits: [test_lss]
    extra_names = cfg["data"].get("extra_test_splits", []) or []

    encoder = load_frozen_encoder(cfg, device)
    ftr, ltr, mtr = extract(encoder, cfg, "train", device, dtype)
    fva, lva, mva = extract(encoder, cfg, "val", device, dtype)
    fte, lte, mte = extract(encoder, cfg, "test", device, dtype)
    extra_raw = {nm: extract(encoder, cfg, nm, device, dtype) for nm in extra_names}
    del encoder; torch.cuda.empty_cache()

    # reference phoneme sequences + label space (gold or pseudo) from the dataset
    val_ds = build_dataset(cfg, "val")[0]
    test_ds = build_dataset(cfg, "test")[0]
    ref_va, ref_te = val_ds.reference_sequences(), test_ds.reference_sequences()
    extra_ref = {nm: build_dataset(cfg, nm)[0].reference_sequences() for nm in extra_names}
    num_classes = val_ds.num_classes
    drop = tuple(val_ds.collapse_drop)
    print(f"[ph-eval] tokens train/val/test = {len(ltr)}/{len(lva)}/{len(lte)} "
          f"clips; T'={ltr.shape[1]}; num_classes={num_classes} drop={drop}")
    for nm in extra_names:
        print(f"[ph-eval] extra test '{nm}' = {len(extra_raw[nm][1])} clips")

    loss_kind = cfg["probe"].get("loss", "ce")
    if ptype in UTT_HEADS:
        # Whole-utterance mode over the spatial cache. Same head + same temporal
        # context under BOTH losses, so CE-vs-CTC isolates the loss (Phase 2).
        ref_tr = build_dataset(cfg, "train")[0].reference_sequences()
        ds_tr = _UttSpatialDS(ftr, ltr, mtr, ref_tr)
        ds_va = _UttSpatialDS(fva, lva, mva, ref_va)
        ds_te = _UttSpatialDS(fte, lte, mte, ref_te)
        extra_tests = {nm: _UttSpatialDS(*extra_raw[nm], extra_ref[nm])
                       for nm in extra_names}
        best, best_state, history = train_probe_utt(
            cfg, ds_tr, ds_va, ds_te, device, num_classes, drop,
            extra_tests=extra_tests, loss_kind=loss_kind)
    elif loss_kind == "ctc":
        ref_tr = build_dataset(cfg, "train")[0].reference_sequences()
        utt_tr = _assemble_utts(ftr, ltr, mtr)
        utt_va = _assemble_utts(fva, lva, mva)
        utt_te = _assemble_utts(fte, lte, mte)
        extra_tests = {nm: (_assemble_utts(*extra_raw[nm]), extra_ref[nm])
                       for nm in extra_names}
        print(f"[ph-eval] CTC mode: utts train/val/test = "
              f"{len(utt_tr)}/{len(utt_va)}/{len(utt_te)}; blank={num_classes}")
        best, best_state, history = train_probe_ctc(
            cfg, utt_tr, ref_tr, utt_va, ref_va, utt_te, ref_te,
            device, num_classes, drop, extra_tests=extra_tests)
    else:
        extra_tests = {nm: (extra_raw[nm][0], extra_raw[nm][1], extra_raw[nm][2],
                            extra_ref[nm]) for nm in extra_names}
        best, best_state, history = train_probe(
            cfg, ftr, ltr, fva, lva, mva, fte, lte, mte,
            ref_va, ref_te, device, num_classes, drop, extra_tests=extra_tests)
    out = {"encoder": cfg["encoder"].get("spec", "pretrained"),
           "kind": cfg["data"].get("kind", "usc_lss"),
           "probe": cfg["probe"].get("type", "tcn"), "loss": loss_kind,
           "spatial_size": cfg["data"]["spatial_size"], "seed": seed,
           "history": history, **best}
    print("\n===== PHONEME RESULT =====")
    if best.get("tests"):
        print(f"[ph-eval] primary test: {best.get('test')}")
        for nm, tm in best["tests"].items():
            print(f"[ph-eval] extra test '{nm}': {tm}")
    print(json.dumps(out, indent=2))
    stem = os.path.join(meta["out"], f"phoneme_{cfg['data'].get('kind','usc_lss')}_"
                        f"{_tag(cfg,'train')[0]}_{cfg['probe'].get('type','tcn')}_{loss_kind}"
                        f"_s{seed}")
    rp = stem + ".json"
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[ph-eval] wrote {rp}")

    # Save the best-val probe weights + everything needed to reload them, so the
    # result is reproducible without retraining (frozen-feature cache is keyed by
    # 'feature_tag'). Toggle off with meta.save_probe: false.
    if best_state is not None and meta.get("save_probe", True):
        pc = cfg["probe"]
        wp = stem + ".pt"
        torch.save({
            "probe_state": best_state, "probe_kind": ptype, "loss": loss_kind,
            "dim": int(ftr.shape[-1]),
            "clf_classes": num_classes + (1 if loss_kind == "ctc" else 0),
            "num_classes": num_classes,
            "hidden": pc.get("hidden", 512), "layers": pc.get("layers", 2),
            "heads": pc.get("heads", 8), "dropout": pc.get("dropout", 0.1),
            "encoder_spec": cfg["encoder"].get("spec", "pretrained"),
            "encoder_checkpoint": cfg["encoder"].get("checkpoint"),
            "feature_tag": _tag(cfg, "train")[0], "seed": seed,
            "best_epoch": best.get("epoch"), "manifest": cfg["data"].get("manifest"),
            "spatial_size": cfg["data"]["spatial_size"], "metrics": out,
        }, wp)
        print(f"[ph-eval] wrote probe weights {wp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--model", default=None,
                    help="baseline encoder: image (clip|siglip|dinov2|vitl|resnet or a "
                         "full timm name) -> encoder.type=image_baseline; or 'videomae' "
                         "(video 3-D, MCG-NJU/videomae-large) -> encoder.type=videomae")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=None, help="override data.batch_size")
    ap.add_argument("--probe", default=None,
                    choices=["linear", "mlp", "tcn", "lstm", "transformer",
                             "tcn_spatial", "attentive", "attentive_lstm"])
    ap.add_argument("--loss", default=None, choices=["ce", "ctc", "focal"],
                    help="probe training loss: ce (per-token, kappa+PER) | ctc (PER-only) "
                         "| focal (per-token CE reweighted by (1-p_t)^gamma; kappa+PER, "
                         "gamma via probe.focal_gamma, default 2.0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override meta.seed (probe init/shuffle only; the frozen "
                         "feature cache is seed-independent, so multi-seed reuses it)")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                    help="override meta.dtype (use float16 on Pascal/V100 -- no bf16)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    if args.dtype is not None:
        cfg["meta"]["dtype"] = args.dtype
    if args.model is not None:
        if args.model.lower().startswith("videomae"):
            cfg["encoder"]["type"] = "videomae"
            cfg["encoder"]["model"] = ("MCG-NJU/videomae-large"
                                       if args.model.lower() == "videomae" else args.model)
        else:
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
