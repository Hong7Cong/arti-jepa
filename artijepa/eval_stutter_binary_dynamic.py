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

Run:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_stutter_binary_dynamic \
        --config dev_artiJEPA/configs/eval_stutter_binary_dynamic.yaml
    ... --sample-fps 25 --window 32 --probe seq_lstm
    ... --sample-fps native
"""

import argparse
import hashlib
import json
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
def _tag(cfg):
    d, ec = cfg["data"], cfg["encoder"]
    hd = {"ckpt": ec["checkpoint"], "key": ec.get("key", "target_encoder"),
          "sz": d["spatial_size"], "window": d.get("window", 32),
          "sample_fps": d.get("sample_fps", 25), "tub": d.get("tubelet_size", 2),
          "pad": d.get("event_pad_s", 0.0), "neg_per_pos": d.get("neg_per_pos", 1.0),
          "build_seed": d.get("build_seed", 0), "tiers": d.get("tiers", ["disfluency"]),
          "min_dur": d.get("min_dur", 0.20), "max_dur": d.get("max_dur", 8.0),
          "merge_gap": d.get("merge_gap", 0.25), "mode": "dyn_spatialpool"}
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or os.path.basename(os.path.dirname(ec["checkpoint"]))
    return f"{tag}_dyn_{h}"


@torch.no_grad()
def extract(encoder, cfg, rows, device, dtype):
    """Ragged spatial-pooled features -> (feats [sumL,D], offsets [N+1], y [N], spk).

    Each clip's K windows are encoded, mean-pooled over S' spatial tokens -> [K,T',D],
    flattened to a [K*T', D] temporal sequence, and written into a flat memmap at the
    clip's ``offsets`` slot. Cached under ``meta.cache_dir/<tag>``.
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
    print(f"[dyn-eval] dataset={len(ds)} class_counts={ds.class_counts().tolist()} "
          f"windows/clip: min={nwin.min()} p50={int(np.median(nwin))} max={nwin.max()} "
          f"seq_len(tokens): p50={int(np.median(seq_len))} max={int(seq_len.max())} "
          f"total_tokens={total_L}")

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
            tok = tok.float().reshape(K, Tp, Ntot // Tp, D).mean(2)   # [K,T',D] spatial pool
            seq = tok.reshape(K * Tp, D).cpu().numpy().astype(np.float16)  # [L,D]
            if feats is None:
                feats = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16,
                                                  shape=(total_L, D))
            a = int(offsets[ci])
            feats[a:a + seq.shape[0]] = seq
            labels.append(item["label"]); spks.append(item["speaker"]); ci += 1
        if ci % 200 < len(batch):
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
    """Variable-length [B,Lmax,D] + lengths -> binary logits (pad-masked).

      seq_attentive -- learned query, nn.MultiheadAttention over the L frame-vectors
                       with a key-padding mask -> [B,D] -> linear.
      seq_lstm      -- packed (bi-)LSTM -> masked mean of outputs -> linear.
    """

    def __init__(self, dim, num_classes, kind="seq_attentive", heads=8, dropout=0.1,
                 lstm_hidden=256, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "seq_attentive":
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

    def forward(self, x, lengths):                          # x: [B, Lmax, D]
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
    def __init__(self, feats, offsets, idx, y):
        self.feats, self.offsets, self.idx, self.y = feats, offsets, idx, y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        return (torch.from_numpy(np.array(self.feats[a:b], dtype=np.float32)),
                int(self.y[i]))


def _collate_ragged(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x = nn.utils.rnn.pad_sequence(seqs, batch_first=True)     # [B, Lmax, D]
    return x, lengths, torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def _predict(clf, feats, offsets, idx, y, device, bs=128):
    clf.eval()
    loader = torch.utils.data.DataLoader(_RaggedDS(feats, offsets, idx, y),
                                         batch_size=bs, shuffle=False,
                                         collate_fn=_collate_ragged)
    out = []
    for x, lengths, _yy in loader:
        out.append(clf(x.to(device), lengths).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def train_probe(cfg, feats, offsets, y, tr, va, te, spk_te, device, classes):
    """Train DynamicSeqProbe on tr, model-select on va (macro-F1), predict te."""
    pc = cfg["probe"]; nc = len(classes); dim = feats.shape[-1]
    clf = DynamicSeqProbe(dim, nc, kind=pc.get("type", "seq_attentive"),
                          heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1),
                          lstm_hidden=pc.get("lstm_hidden", 256),
                          lstm_layers=pc.get("lstm_layers", 1),
                          bidirectional=pc.get("bidirectional", True)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    w = _class_weights(y[tr], nc, device) if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    loader = torch.utils.data.DataLoader(
        _RaggedDS(feats, offsets, tr, y[tr]), batch_size=pc.get("batch_size", 64),
        shuffle=True, num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=_collate_ragged)
    best = {"val_f1": -1.0, "test": None, "pred": None}
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0
        for x, lengths, yy in loader:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad()
            loss = lossf(clf(x, lengths), yy)
            loss.backward(); opt.step(); run += float(loss); nb += 1
        vp = _predict(clf, feats, offsets, va, y, device)
        vm = S.classification_metrics(y[va], vp, nc, classes)
        if vm["macro_f1"] > best["val_f1"]:
            tp = _predict(clf, feats, offsets, te, y, device)
            tm = S.classification_metrics(y[te], tp, nc, classes)
            best = {"val_f1": vm["macro_f1"], "test": tm, "pred": tp}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[dyn-probe {spk_te} e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} "
                  f"val macroF1={vm['macro_f1']:.3f} best={best['val_f1']:.3f}")
    return best["test"], best["pred"], best["val_f1"]


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
    # The encoder is built/loaded at the WINDOW geometry (each window is an
    # in-distribution `window`-frame clip); load_frozen_encoder reads frames_per_clip.
    cfg["data"]["frames_per_clip"] = cfg["data"].get("window", 32)
    print(f"[dyn-eval] dynamic binary | classes={classes} probe={ptype} "
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

    def fold(tr, va, te, name):
        tm, pred, vf1 = train_probe(cfg, feats, offsets, y, tr, va, te, name, device, classes)
        folds.append({"speaker": name, "n_test": int(len(te)), "val_f1": round(vf1, 4), **tm})
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())

    if split_mode == "loso":
        for ts in uniq_spk:
            te = keep[spk == ts]; tr_all = keep[spk != ts]
            if len(np.unique(y[tr_all])) < 2 or len(te) == 0:
                print(f"[dyn-eval] skip fold {ts}: degenerate"); continue
            tr, va = _val_split(tr_all, y, len(classes), val_frac, rng)
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
           "sample_fps": cfg["data"].get("sample_fps", 25),
           "window": cfg["data"].get("window", 32),
           "spatial_size": cfg["data"]["spatial_size"], "seed": seed,
           "split_mode": split_mode, "pooled": pooled,
           "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
           "balanced_acc_mean": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
           "accuracy_mean": round(float(np.mean([f["accuracy"] for f in folds])), 4),
           "folds": folds}
    _report(cfg, out)
    return out


def _report(cfg, out):
    print("\n===== STUTTER BINARY DYNAMIC (fluent vs disfluent) RESULT =====")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=2))
    print("[dyn-eval] per-fold macro-F1:", {f["speaker"]: f["macro_f1"] for f in out["folds"]})
    rp = os.path.join(cfg["meta"]["out"],
                      f"stutter_binary_dyn_{_tag(cfg)}_{out['probe']}_{out['split_mode']}_s{out['seed']}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[dyn-eval] wrote {rp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--probe", default=None, choices=["seq_attentive", "seq_lstm"])
    ap.add_argument("--sample-fps", default=None,
                    help="frames/sec to sample per event, or 'native' (~99)")
    ap.add_argument("--window", type=int, default=None, help="frames per encoder window (default 32)")
    ap.add_argument("--split", default=None, choices=["loso", "fixed", "random"])
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
    if args.test_speaker is not None:
        cfg["data"]["test_speaker"] = args.test_speaker
    if args.val_speaker is not None:
        cfg["data"]["val_speaker"] = args.val_speaker
    if args.batch is not None:
        cfg["probe"]["batch_size"] = args.batch
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
