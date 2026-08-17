"""Dynamic-length binary FLUENT-vs-DISFLUENT eval (frozen T-SSL / V-JEPA).

Companion to ``eval_stutter_binary`` (fixed-length) that accepts **variable-length**
inputs: each event is sampled at a target FPS (or native), tiled into in-distribution
32f windows (K ~ duration), each window is **mean-pooled over its S' spatial tokens**
(the pooled-spatial strategy -> tiny cache), and the K windows form a variable-length
temporal sequence ``[L=K*T', D]`` classified by a **sequence model**:

  * ``seq_attentive`` -- a learned query attends over the L frame-vectors with a
    key-padding mask (length-agnostic; the default).
  * ``seq_lstm``      -- a (bi-)LSTM over the packed sequence, masked mean of outputs.

Because K scales with duration, a 0.5 s disfluency and a 6 s one keep their real
temporal extent/rate (not resampled to a fixed frame budget), and every window is
32f so the frozen 32f-pretrained encoder stays in-distribution.

The feature cache is **ragged**: one flat ``[sum L_i, D]`` fp16 memmap + per-clip
``offsets`` -- still tiny thanks to spatial pooling.

FULL-GRID mode (``--pool-mode none``, probe ``seq_attentive_lstm``)
-------------------------------------------------------------------
The spatial mean-pool above destroys within-frame structure -- and the fixed-32f
`attentive` result (docs/STUTTERING.md §12 phase 2c) showed that is exactly where the
rt-MRI fine-tune's gain lives (tssl256 +0.032 grid-over-pooled). ``--pool-mode none``
keeps the S'=256 spatial tokens, so the ragged cache is ``[sum L_i, S', D]`` -- **S'x
bigger: 62 GiB @25fps, 110 GiB @50fps, 204 GiB @native(99fps)**. The probe
``seq_attentive_lstm`` is the dynamic-length counterpart of `eval_disfluency`'s
`attentive_lstm`: a chunked AttentivePooler over S' per frame -> [B,L,D] -> packed
bi-LSTM -> masked mean -> linear.

At that size the cache no longer fits in page cache or host RAM, so this path reuses
the **four OOM fixes from the phoneme Phase-2 utterance-mode probe** (RESULTS_phonepred.md
"Loss landscape (Phase 2)"; `eval_phoneme.py`):
  1. ``madvise(MADV_DONTNEED)`` on the write-side memmap during extraction -- written
     pages stay resident and are charged to the job's cgroup otherwise (204 GiB > any
     sane ``--mem``). Flush + advise-away; extraction never reads back, so RSS stays flat.
  2. A **padded-token batch sampler** (``_TokenBudget``) instead of a fixed clip
     ``batch_size``: budget ``len(batch) * max(L)`` because the collate pads to the
     batch max. A fixed 64-clip batch padded to L=400 would allocate
     64*400*256*1024 -- budgeting keeps peak flat across the 16..400-token spread.
  3. Eval loaders at ``eval_workers`` (default 2) x ``prefetch_factor=1`` so a big
     fold's batches cannot stack 2-deep per worker.
  4. Items stay **fp16** out of the dataset (upcast on GPU, not in the worker), halving
     collate + pinned-transfer host RAM.

Run:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_stutter_binary_dynamic \
        --config dev_artiJEPA/configs/eval_stutter_binary_dynamic.yaml
    ... --sample-fps 25 --window 32 --probe seq_lstm
    ... --sample-fps native
    # full-grid, native fps, ckpt_215 (its own 204 GiB cache + tag):
    ... --probe seq_attentive_lstm --sample-fps native \
        --checkpoint /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt \
        --tag tssl256comb215
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

from artijepa import stutter as S
from artijepa import stutter_dynamic as SD
from artijepa.eval_disfluency import _class_weights, _val_split
from artijepa.eval_stutter_binary import (
    BINARY_CLASSES, build_rows, load_frozen_encoder, load_config, _pin_threads,
    _worker_init)


# --------------------------------------------------------------------------- #
# ragged (variable-length) feature extraction -- spatial-pooled, tiny cache
# --------------------------------------------------------------------------- #
def pool_mode_for_probe(ptype):
    """How the [T',S',D] window grid is reduced AT EXTRACTION, per probe.

      spatial -> [T',D]     mean over S'   (seq_attentive / seq_lstm; tiny cache)
      none    -> [T',S',D]  keep the grid  (seq_attentive_lstm; S'x = ~204 GiB @native)
    """
    return "none" if ptype == "seq_attentive_lstm" else "spatial"


def _tag(cfg):
    d, ec = cfg["data"], cfg["encoder"]
    pm = d.get("pool_mode", "spatial")
    hd = {"ckpt": ec["checkpoint"], "key": ec.get("key", "target_encoder"),
          "sz": d["spatial_size"], "window": d.get("window", 32),
          "sample_fps": d.get("sample_fps", 25), "tub": d.get("tubelet_size", 2),
          "pad": d.get("event_pad_s", 0.0), "neg_per_pos": d.get("neg_per_pos", 1.0),
          "build_seed": d.get("build_seed", 0), "tiers": d.get("tiers", ["disfluency"]),
          "min_dur": d.get("min_dur", 0.20), "max_dur": d.get("max_dur", 8.0),
          "merge_gap": d.get("merge_gap", 0.25),
          # full-grid gets a DISTINCT mode string -> distinct hash; every existing
          # spatial-pooled cache keeps its current directory name untouched.
          "mode": "dyn_fullgrid" if pm == "none" else "dyn_spatialpool"}
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or os.path.basename(os.path.dirname(ec["checkpoint"]))
    return f"{tag}_dyn_{h}"


@torch.no_grad()
def extract(encoder, cfg, rows, device, dtype):
    """Ragged features -> (feats, offsets [N+1], y [N], spk); offsets index the L axis.

    Each clip's K windows are encoded to [K,T',S',D], then per ``data.pool_mode``:
      spatial -> mean over S' -> flat [K*T', D]      (tiny cache)
      none    -> keep S'      -> flat [K*T', S', D]  (~204 GiB @native fps)
    written into a flat memmap at the clip's ``offsets`` slot, cached under
    ``meta.cache_dir/<tag>``.

    OOM fix 1 (phoneme Phase 2): for the full-grid mode the memmap's written pages stay
    resident and are charged to the job's memory cgroup -> OOM well before the 204 GiB is
    written. Periodically ``flush()`` + ``madvise(MADV_DONTNEED)`` the mapping so the
    kernel reclaims them (re-read from disk only if touched; extraction never reads back,
    so RSS stays flat). NOTE: fadvise() on a separate fd does NOT reclaim pages still
    mapped by the memmap -- madvise on the mmap itself does. On-disk output is identical.
    """
    d = cfg["data"]
    name = _tag(cfg)
    cdir = os.path.join(cfg["meta"]["cache_dir"], name); os.makedirs(cdir, exist_ok=True)
    fp, op, yp, kp = (os.path.join(cdir, f"all.{x}.npy")
                      for x in ("feats", "offset", "label", "spk"))
    if all(os.path.exists(x) for x in (fp, op, yp, kp)):
        print(f"[dyn-eval] cache hit <- {cdir}")
        return (np.load(fp, mmap_mode="r"), np.load(op), np.load(yp), list(np.load(kp)))

    ds = SD.make_dataset(
        rows, sample_fps=d.get("sample_fps", 25), window=d.get("window", 32),
        spatial_size=d["spatial_size"], spatial_mode=d.get("spatial_mode", "resize"),
        intensity_norm=d.get("intensity_norm", "zscore"),
        grayscale_stats=d.get("grayscale_stats", SD.SB.GRAYSCALE_STATS),
        tubelet_size=d.get("tubelet_size", 2), event_pad_s=d.get("event_pad_s", 0.0),
        classes=tuple(BINARY_CLASSES))
    seq_len = np.asarray(ds.seq_len, dtype=np.int64)
    offsets = np.zeros(len(seq_len) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(seq_len)
    total_L = int(offsets[-1])
    nwin = np.asarray(ds.n_win)
    pool_mode = d.get("pool_mode", "spatial")
    Sp = (d["spatial_size"] // d.get("patch_size", 16)) ** 2
    est = total_L * (1 if pool_mode == "spatial" else Sp) * 1024 * 2 / 2**30
    print(f"[dyn-eval] dataset={len(ds)} class_counts={ds.class_counts().tolist()} "
          f"windows/clip: min={nwin.min()} p50={int(np.median(nwin))} max={nwin.max()} "
          f"seq_len(tokens): p50={int(np.median(seq_len))} max={int(seq_len.max())} "
          f"total_tokens={total_L} | pool_mode={pool_mode} est_cache~{est:.1f} GiB")

    nw = d.get("num_workers", 6)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=d.get("extract_clip_batch", 4), shuffle=False, num_workers=nw,
        collate_fn=SD.collate_windows, worker_init_fn=_worker_init if nw > 0 else None)
    Tp = d.get("window", 32) // d.get("tubelet_size", 2)
    feats = None; labels = []; spks = []; ci = 0; t0 = time.time(); N = len(ds)
    for batch in loader:
        for item in batch:
            clips = item["clips"].to(device, non_blocking=True)   # [K,3,window,S,S]
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32 and device.type == "cuda")):
                tok = encoder.backbone(clips)                     # [K, T'*S', D]
            K, Ntot, D = tok.shape
            tok = tok.float().reshape(K, Tp, Ntot // Tp, D)       # [K,T',S',D]
            if pool_mode == "spatial":
                seq = tok.mean(2).reshape(K * Tp, D)              # [L,D]
            else:
                seq = tok.reshape(K * Tp, tok.shape[2], D)        # [L,S',D]
            seq = seq.cpu().numpy().astype(np.float16)
            if feats is None:
                feats = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16,
                                                  shape=(total_L,) + seq.shape[1:])
            a = int(offsets[ci])
            feats[a:a + seq.shape[0]] = seq
            labels.append(item["label"]); spks.append(item["speaker"]); ci += 1
        if ci % 200 < len(batch):
            # OOM fix 1: reclaim the written pages (see docstring). Cheap for the
            # spatial cache, load-bearing for the ~204 GiB full-grid one.
            feats.flush()
            try:
                feats._mmap.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            print(f"[dyn-eval]  extract {ci}/{N} ({time.time()-t0:.0f}s)")
    feats.flush()
    labels = np.asarray(labels, dtype=np.int64)
    np.save(op, offsets); np.save(yp, labels); np.save(kp, np.asarray(spks))
    print(f"[dyn-eval] extracted {feats.shape} ({total_L} tokens) in "
          f"{time.time()-t0:.0f}s -> {cdir}")
    return np.load(fp, mmap_mode="r"), offsets, labels, spks


# --------------------------------------------------------------------------- #
# variable-length sequence probe
# --------------------------------------------------------------------------- #
class DynamicSeqProbe(nn.Module):
    """Variable-length [B,Lmax,D] (or [B,Lmax,S',D]) + lengths -> binary logits (pad-masked).

      seq_attentive      -- learned query, nn.MultiheadAttention over the L frame-vectors
                            with a key-padding mask -> [B,D] -> linear.
      seq_lstm           -- packed (bi-)LSTM -> masked mean of outputs -> linear.
      seq_attentive_lstm -- full-grid input [B,Lmax,S',D]: one shared AttentivePooler
                            pools the S'=256 SPATIAL tokens per frame -> [B,Lmax,D],
                            then the packed bi-LSTM + masked mean above. The dynamic-
                            length counterpart of eval_disfluency's `attentive_lstm`.
                            The spatial pool is swept over the L axis in ``chunk``-sized
                            blocks with optional gradient checkpointing, so peak
                            activation memory is O(B*chunk*S') not O(B*Lmax*S') --
                            the difference between fitting and not at Lmax=400.
    """

    def __init__(self, dim, num_classes, kind="seq_attentive", heads=8, dropout=0.1,
                 lstm_hidden=256, lstm_layers=1, bidirectional=True,
                 chunk=0, checkpoint=False):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "seq_attentive_lstm":
            from src.models.attentive_pooler import AttentivePooler
            self.chunk = int(chunk) or 32
            self.use_ckpt = bool(checkpoint)
            self.spatial_pool = AttentivePooler(num_queries=1, embed_dim=dim,
                                                num_heads=heads, mlp_ratio=4.0, depth=1)
            self.lstm = nn.LSTM(dim, lstm_hidden, num_layers=lstm_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            out = lstm_hidden * (2 if bidirectional else 1)
            self.norm = nn.LayerNorm(out)
            self.head = nn.Linear(out, num_classes)
        elif kind == "seq_attentive":
            self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes)
        elif kind == "seq_lstm":
            self.lstm = nn.LSTM(dim, lstm_hidden, num_layers=lstm_layers,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            out = lstm_hidden * (2 if bidirectional else 1)
            self.norm = nn.LayerNorm(out)
            self.head = nn.Linear(out, num_classes)
        else:
            raise ValueError(kind)

    def _spatial_pool_chunked(self, x):
        """[B,L,S',D] -> [B,L,D]: attentive-pool the S' spatial tokens per frame.

        Frames pool independently, so sweep the L axis in ``chunk``-sized blocks; under
        training optionally gradient-checkpoint each block so its activations are
        recomputed in backward (peak ~O(B*chunk*S') not O(B*L*S')).
        """
        B, L, S, D = x.shape
        outs = []
        for c0 in range(0, L, self.chunk):
            xc = x[:, c0:c0 + self.chunk].reshape(-1, S, D)      # [B*lc, S', D]
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                q = checkpoint(lambda t: self.spatial_pool(t).squeeze(1), xc,
                               use_reentrant=False)
            else:
                q = self.spatial_pool(xc).squeeze(1)             # [B*lc, D]
            outs.append(q.reshape(B, -1, D))
        return torch.cat(outs, dim=1)                            # [B, L, D]

    def forward(self, x, lengths):            # x: [B,Lmax,D] or [B,Lmax,S',D] (full grid)
        if self.kind == "seq_attentive_lstm":
            # Pad frames are pooled too (wasted compute only) -- the LSTM is packed to
            # `lengths` and the masked mean below drops them, so they never reach a metric.
            x = self._spatial_pool_chunked(x)                    # [B,Lmax,S',D]->[B,Lmax,D]
        B, Lmax, D = x.shape
        pad = torch.arange(Lmax, device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        if self.kind == "seq_attentive":
            q = self.q.expand(B, 1, D)
            z, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)  # [B,1,D]
            z = self.norm(z.squeeze(1))
            return self.head(self.drop(z))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)     # [B,L',H]
        m = (~pad[:, :out.shape[1]]).float().unsqueeze(-1)
        z = (out * m).sum(1) / m.sum(1).clamp(min=1.0)                       # masked mean
        return self.head(self.drop(self.norm(z)))


class _RaggedDS(torch.utils.data.Dataset):
    """One item = one clip read on demand from the ragged cache: (feats, label).

    OOM fix 4: rows stay **fp16** (the cache dtype). The old float32 upcast here doubled
    every collated batch and every pinned transfer in host RAM -- irrelevant at [L,D]
    (~200 KB/clip), 2 x 200 MiB/clip on the full grid. The probe upcasts on the GPU.
    """

    def __init__(self, feats, offsets, idx, y):
        self.feats, self.offsets, self.idx, self.y = feats, offsets, idx, y
        # padded-token budgeting needs each item's length up front (no read required)
        self.lengths = [int(offsets[int(j) + 1]) - int(offsets[int(j)]) for j in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        # `i` (the position within idx) rides along: the token-budget sampler emits
        # batches out of idx order, so predictions must be scattered back by position
        # or they silently misalign with y[te] at metric time.
        return (torch.from_numpy(np.array(self.feats[a:b], dtype=np.float16)),  # ONE seq read
                int(self.y[i]), i)


def _collate_ragged(batch):
    seqs, ys, pos = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # [B,Lmax,D] or [B,Lmax,S',D]
    return (x, lengths, torch.tensor(ys, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long))


class _TokenBudget(torch.utils.data.Sampler):
    """Batch clips up to ~``max_tokens`` PADDED temporal tokens (always >= 1 clip).

    OOM fix 2 (phoneme Phase 2). On the full grid a clip's cost is L*S'*D, and L spans
    16..400 tokens here (native fps) -- so a fixed clip batch_size swings peak memory
    ~25x with duration. The budget is on the padded size ``len(batch) * max(L)``, NOT
    ``sum(L)``, because _collate_ragged pads every clip up to the batch's longest: they
    coincide only for equal-length batches, and budgeting sum() is exactly what packed a
    crowd of short clips next to one 400-token clip and allocated the batch that OOM'd
    the phoneme run. Padded budgeting also stops the wasted compute on pad.
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
            L = int(self.lengths[i]); m = max(mx, L)
            if batch and (len(batch) + 1) * m > self.max_tokens:
                yield batch; batch, mx = [int(i)], L       # i starts the next batch
            else:
                batch.append(int(i)); mx = m
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(1 for _ in iter(self)))


def _ragged_loader(ds, cfg, shuffle, seed=0, eval_mode=False):
    """DataLoader over a _RaggedDS: token-budgeted on the full grid, fixed-batch otherwise.

    OOM fix 3: the eval loaders (val each epoch + test) use ``probe.eval_workers``
    (default 2) x ``prefetch_factor=1``, so a fold's batches cannot stack 2-deep per
    worker on top of the cgroup -- host RAM ~= workers * prefetch * padded_batch_bytes.
    """
    pc, d = cfg["probe"], cfg["data"]
    full = d.get("pool_mode", "spatial") == "none"
    workers = pc.get("eval_workers", 2) if eval_mode else \
        pc.get("train_workers", d.get("num_workers", 4))
    kw = {"num_workers": workers, "collate_fn": _collate_ragged}
    if workers > 0:
        # Host RAM is bounded HERE, and it is independent of the cache size (ΣL):
        #   bytes ~= workers * prefetch * max_tokens * S' * D * 2
        # At max_tokens=1024, S'=256, D=1024 one padded batch is 512 MiB, so the
        # defaults below (train 4x2, eval 2x1) hold the loaders to ~5 GiB total --
        # a 64 GB cgroup is ample even for the 204 GiB native-fps cache. The 204 GiB
        # never needs to be resident: extraction madvise'es its dirty pages away, and
        # the probe's sequential reads leave only RECLAIMABLE page cache behind.
        kw["prefetch_factor"] = 1 if eval_mode else pc.get("train_prefetch", 2)
    if full:
        kw["batch_sampler"] = _TokenBudget(ds.lengths, pc.get("max_tokens", 1024),
                                           shuffle, seed)
    else:                       # spatial-pooled path: unchanged fixed-clip batching
        kw["batch_size"] = pc.get("batch_size", 64); kw["shuffle"] = shuffle
    return torch.utils.data.DataLoader(ds, **kw)


@torch.no_grad()
def _infer(clf, feats, offsets, idx, y_idx, device, cfg, lossf=None):
    """Full deterministic pass -> (preds aligned to ``idx`` order, mean loss).

    ``y_idx`` must be y sliced to ``idx`` (matches _RaggedDS). Predictions are scattered
    back by the position each item carries, so the token-budget sampler's reordering
    cannot misalign them with ``y[idx]``.
    """
    clf.eval()
    amp = device.type == "cuda"
    out = np.full(len(idx), -1, dtype=np.int64); tot = 0.0; nb = 0
    for x, lengths, yy, pos in _ragged_loader(_RaggedDS(feats, offsets, idx, y_idx), cfg,
                                              shuffle=False, eval_mode=True):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = clf(x.to(device, non_blocking=True).float(), lengths)
        logits = logits.float()
        if lossf is not None:
            tot += float(lossf(logits, yy.to(device))); nb += 1
        out[pos.numpy()] = logits.argmax(-1).cpu().numpy()
    return out, tot / max(1, nb)


def _cpu_state(model):
    """Detached CPU copy of a model's params, for a best-epoch probe snapshot."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_probe(path, device="cpu"):
    """Reconstruct a trained DynamicSeqProbe from a checkpoint written by run().

    Returns (probe.eval(), meta_dict). LOSO trains one probe PER FOLD, so the checkpoint
    holds ``probe_states`` keyed by held-out speaker; pass ``fold`` to pick one. The frozen
    features it was trained on are reproducible from meta['feature_tag'] (extract's cache dir).

        clf, meta = load_probe(".../stutter_binary_dyn_..._s0.pt", fold="PWS8")
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def build_probe_from_ckpt(ck, fold, device="cpu"):
    """Instantiate one fold's probe from a checkpoint dict returned by load_probe."""
    clf = DynamicSeqProbe(ck["dim"], len(ck["classes"]), kind=ck["probe_kind"],
                          heads=ck["heads"], dropout=ck["dropout"],
                          lstm_hidden=ck["lstm_hidden"], lstm_layers=ck["lstm_layers"],
                          bidirectional=ck["bidirectional"], chunk=ck["chunk"],
                          checkpoint=False)      # ckpt only affects training memory
    clf.load_state_dict(ck["probe_states"][fold])
    return clf.to(device).eval()


def train_probe(cfg, feats, offsets, y, tr, va, te, spk_te, device, classes):
    """Train DynamicSeqProbe on tr, model-select on va (macro-F1), predict te.

    Returns (test_metrics, test_pred, best_val_f1, best_state) -- best_state is a CPU
    snapshot of the weights at the best-val epoch (mirrors eval_phoneme.py), so the
    trained probe survives the run for attention-map / transfer analysis.
    """
    pc = cfg["probe"]; nc = len(classes); dim = feats.shape[-1]
    full = cfg["data"].get("pool_mode", "spatial") == "none"
    clf = DynamicSeqProbe(dim, nc, kind=pc.get("type", "seq_attentive"),
                          heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1),
                          lstm_hidden=pc.get("lstm_hidden", 256),
                          lstm_layers=pc.get("lstm_layers", 1),
                          bidirectional=pc.get("bidirectional", True),
                          chunk=pc.get("chunk", 0),
                          checkpoint=pc.get("checkpoint", False)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    w = _class_weights(y[tr], nc, device) if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    # fp16 autocast + GradScaler on the full grid: the chunked spatial pool over
    # S'=256 per frame is the dominant cost and is what the VRAM budget turns on.
    amp = device.type == "cuda" and full
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ds_tr = _RaggedDS(feats, offsets, tr, y[tr])
    loader = _ragged_loader(ds_tr, cfg, shuffle=True, seed=cfg["meta"].get("seed", 0))
    best = {"val_f1": -1.0, "test": None, "pred": None}
    best_state = None
    for ep in range(epochs):
        if full:
            loader.batch_sampler.epoch = ep            # reshuffle the token-budget batches
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0; t0 = time.time()
        for x, lengths, yy, _pos in loader:
            x = x.to(device, non_blocking=True).float(); yy = yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = lossf(clf(x, lengths), yy)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss); nb += 1
        vp, vloss = _infer(clf, feats, offsets, va, y[va], device, cfg, lossf)
        vm = S.classification_metrics(y[va], vp, nc, classes)
        if vm["macro_f1"] > best["val_f1"]:
            tp, _ = _infer(clf, feats, offsets, te, y[te], device, cfg)
            tm = S.classification_metrics(y[te], tp, nc, classes)
            best = {"val_f1": vm["macro_f1"], "test": tm, "pred": tp, "epoch": ep + 1}
            best_state = _cpu_state(clf)          # snapshot best-val weights
        rec = "/".join(f"{c[:3]}={vm['per_class'][c]['recall']}" for c in classes)
        print(f"[dyn-probe {spk_te} e{ep+1}/{epochs}] tr_loss={run/max(1,nb):.3f} "
              f"val_loss={vloss:.3f} val_kappa={vm['cohen_kappa']:.3f} "
              f"val_macroF1={vm['macro_f1']:.3f} best={best['val_f1']:.3f} "
              f"({(time.time()-t0)/60:.1f} min) | recall {rec}", flush=True)
    return best["test"], best["pred"], best["val_f1"], best_state, best.get("epoch")


# --------------------------------------------------------------------------- #
# run (LOSO / fixed / random) -- mirrors eval_stutter_binary
# --------------------------------------------------------------------------- #
def run(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    _pin_threads(cfg["data"].get("cpu_threads", 2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "float16").lower(), torch.float32)
    classes = BINARY_CLASSES
    ptype = cfg["probe"].get("type", "seq_attentive")
    # How the [T',S',D] window grid is reduced at extraction (drives the cache shape
    # AND the batching strategy: the full grid is token-budgeted, see _ragged_loader).
    cfg["data"]["pool_mode"] = pool_mode_for_probe(ptype)
    # The encoder is built/loaded at the WINDOW geometry (each window is an
    # in-distribution `window`-frame clip); load_frozen_encoder reads frames_per_clip.
    cfg["data"]["frames_per_clip"] = cfg["data"].get("window", 32)
    print(f"[dyn-eval] dynamic binary | classes={classes} probe={ptype} "
          f"pool_mode={cfg['data']['pool_mode']} "
          f"sample_fps={cfg['data'].get('sample_fps', 25)} window={cfg['data'].get('window', 32)} "
          f"device={device}")

    gs = cfg["data"].get("grayscale_stats")
    print(f"[dyn-eval] grayscale stats {'<- '+gs if gs and os.path.exists(gs) else str(gs)+' absent -> mean=0/std=1'}")

    rows, row_stats = build_rows(cfg)
    encoder = load_frozen_encoder(cfg, device)
    feats, offsets, y, speakers = extract(encoder, cfg, rows, device, dtype)
    del encoder; torch.cuda.empty_cache()

    y = np.asarray(y, dtype=np.int64); spk = np.asarray(speakers)
    keep = np.arange(len(y), dtype=np.int64)
    uniq_spk = sorted(set(spk.tolist()))
    print(f"[dyn-eval] {len(keep)} clips; per-class(fluent,disfluent)="
          f"{np.bincount(y, minlength=2).tolist()}; speakers {uniq_spk}")

    split_mode = cfg["data"].get("split_mode", "loso")
    val_frac = cfg["data"].get("val_frac", 0.15)
    folds, all_true, all_pred = [], [], []
    probe_states = {}          # held-out speaker -> best-val CPU state_dict

    def fold(tr, va, te, name):
        tm, pred, vf1, state, bep = train_probe(cfg, feats, offsets, y, tr, va, te, name,
                                                device, classes)
        if state is not None:
            probe_states[name] = state
        folds.append({"speaker": name, "n_test": int(len(te)), "val_f1": round(vf1, 4),
                      "best_epoch": bep, **tm})
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())

    if split_mode == "loso":
        # `only` runs a SUBSET of the LOSO folds so the 7 folds can be split across
        # parallel jobs (the full-grid native-fps cell is ~7h/fold -- 49h serial, past
        # any wall). The loop still visits EVERY speaker and still calls _val_split for
        # each, so `rng` is consumed identically: fold k's train/val split is bit-
        # identical to the all-folds run. Only the training is skipped. Merge the
        # per-fold JSONs afterwards to get the pooled metric.
        only = cfg["data"].get("only_folds")
        for ts in uniq_spk:
            te = keep[spk == ts]; tr_all = keep[spk != ts]
            if len(np.unique(y[tr_all])) < 2 or len(te) == 0:
                print(f"[dyn-eval] skip fold {ts}: degenerate"); continue
            tr, va = _val_split(tr_all, y, len(classes), val_frac, rng)  # consume rng
            if only and ts not in only:
                print(f"[dyn-eval] skip fold {ts}: not in --folds {sorted(only)}")
                continue
            fold(tr, va, te, ts)
    elif split_mode == "fixed":
        ts = cfg["data"]["test_speaker"]; vs = cfg["data"].get("val_speaker")
        te = keep[spk == ts]
        if vs:
            va = keep[spk == vs]; tr = keep[(spk != ts) & (spk != vs)]
        else:
            tr, va = _val_split(keep[spk != ts], y, len(classes), val_frac, rng)
        if len(te) == 0 or len(tr) == 0:
            raise SystemExit(f"[dyn-eval] fixed split degenerate (test={ts!r} val={vs!r})")
        fold(tr, va, te, ts)
    else:
        perm = keep.copy(); rng.shuffle(perm)
        n = len(perm); te = perm[: int(0.2 * n)]; rest = perm[int(0.2 * n):]
        tr, va = _val_split(rest, y, len(classes), val_frac, rng)
        fold(tr, va, te, "random")

    if not folds:
        raise SystemExit("[dyn-eval] no usable folds")
    pooled = S.classification_metrics(np.asarray(all_true), np.asarray(all_pred),
                                      len(classes), classes)
    out = {"encoder": cfg["encoder"]["checkpoint"], "type": "vjepa", "mode": "dynamic",
           "task": "binary", "classes": classes, "probe": ptype,
           "tag": cfg["meta"].get("tag"),
           "pool_mode": cfg["data"]["pool_mode"],
           "sample_fps": cfg["data"].get("sample_fps", 25),
           "window": cfg["data"].get("window", 32),
           "spatial_size": cfg["data"]["spatial_size"], "seed": seed,
           "split_mode": split_mode, "pooled": pooled,
           "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
           "balanced_acc_mean": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
           "accuracy_mean": round(float(np.mean([f["accuracy"] for f in folds])), 4),
           "folds": folds}
    rp = _report(cfg, out)
    # Save the per-fold best-val probe weights alongside the metrics JSON (mirrors
    # eval_phoneme.py). LOSO trains one probe per held-out speaker, so `probe_states` is
    # keyed by speaker -- a single .pt next to the .json, reconstructable via
    # load_probe() + build_probe_from_ckpt(ck, fold). Small: the probe is an
    # AttentivePooler(D=1024) + 1-layer biLSTM ~ tens of MB per fold, not the encoder.
    if probe_states and cfg["meta"].get("save_probe", True):
        pc = cfg["probe"]
        wp = rp[:-len(".json")] + ".pt"
        torch.save({
            "probe_states": probe_states, "probe_kind": ptype,
            "dim": int(feats.shape[-1]), "classes": classes,
            "heads": pc.get("heads", 8), "dropout": pc.get("dropout", 0.1),
            "lstm_hidden": pc.get("lstm_hidden", 256),
            "lstm_layers": pc.get("lstm_layers", 1),
            "bidirectional": pc.get("bidirectional", True),
            "chunk": pc.get("chunk", 0),
            "encoder_checkpoint": cfg["encoder"].get("checkpoint"),
            "feature_tag": _tag(cfg), "seed": seed,
            "sample_fps": cfg["data"].get("sample_fps", 25),
            "window": cfg["data"].get("window", 32),
            "pool_mode": cfg["data"]["pool_mode"],
            "spatial_size": cfg["data"]["spatial_size"], "metrics": out,
        }, wp)
        print(f"[dyn-eval] wrote probe weights ({len(probe_states)} folds) {wp}")
    return out


def _report(cfg, out):
    print("\n===== STUTTER BINARY DYNAMIC (fluent vs disfluent) RESULT =====")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=2))
    print("[dyn-eval] per-fold macro-F1:", {f["speaker"]: f["macro_f1"] for f in out["folds"]})
    # Parallel per-fold jobs share every other component of this name, so WITHOUT the
    # fold suffix all 7 would race to write (and clobber) one file. `_full` marks the
    # all-folds run, whose `pooled` block is the real pooled metric; per-fold files carry
    # only their own fold and must be merged to get it.
    only = cfg["data"].get("only_folds")
    sfx = "_full" if not only else "_" + "-".join(sorted(only))
    rp = os.path.join(cfg["meta"]["out"],
                      f"stutter_binary_dyn_{_tag(cfg)}_{out['probe']}_"
                      f"{out['split_mode']}{sfx}_s{out['seed']}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[dyn-eval] wrote {rp}")
    return rp                      # run() derives the probe-weights .pt path from this


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--probe", default=None,
                    choices=["seq_attentive", "seq_lstm", "seq_attentive_lstm"],
                    help="seq_attentive_lstm keeps the full S' grid (own ~204 GiB cache)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="full-grid: padded temporal tokens per batch (default 1024)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="full-grid: temporal chunk for the spatial pool (default 32)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="full-grid: gradient-checkpoint the chunked spatial pool")
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="workers for the val/test loaders (default 2; prefetch=1)")
    ap.add_argument("--sample-fps", default=None,
                    help="frames/sec to sample per event, or 'native' (~99)")
    ap.add_argument("--window", type=int, default=None, help="frames per encoder window (default 32)")
    ap.add_argument("--split", default=None, choices=["loso", "fixed", "random"])
    ap.add_argument("--folds", default=None,
                    help="LOSO: comma-separated speakers to run in THIS job (e.g. PWS3,PWS8); "
                         "val splits stay bit-identical to the all-folds run")
    ap.add_argument("--test-speaker", default=None)
    ap.add_argument("--val-speaker", default=None)
    ap.add_argument("--batch", type=int, default=None, help="probe batch (clips)")
    ap.add_argument("--extract-clip-batch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg["encoder"]["checkpoint"] = args.checkpoint
    if args.probe is not None:
        cfg["probe"]["type"] = args.probe
    if args.sample_fps is not None:
        cfg["data"]["sample_fps"] = args.sample_fps if args.sample_fps == "native" else float(args.sample_fps)
    if args.window is not None:
        cfg["data"]["window"] = args.window
    if args.split is not None:
        cfg["data"]["split_mode"] = args.split
    if args.folds is not None:
        cfg["data"]["only_folds"] = {s.strip() for s in args.folds.split(",") if s.strip()}
    if args.test_speaker is not None:
        cfg["data"]["test_speaker"] = args.test_speaker
    if args.val_speaker is not None:
        cfg["data"]["val_speaker"] = args.val_speaker
    if args.batch is not None:
        cfg["probe"]["batch_size"] = args.batch
    if args.max_tokens is not None:
        cfg["probe"]["max_tokens"] = args.max_tokens
    if args.chunk is not None:
        cfg["probe"]["chunk"] = args.chunk
    if args.grad_checkpoint:
        cfg["probe"]["checkpoint"] = True
    if args.eval_workers is not None:
        cfg["probe"]["eval_workers"] = args.eval_workers
    if args.extract_clip_batch is not None:
        cfg["data"]["extract_clip_batch"] = args.extract_clip_batch
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.tag is not None:
        cfg["meta"]["tag"] = args.tag
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    run(cfg)


if __name__ == "__main__":
    main()
