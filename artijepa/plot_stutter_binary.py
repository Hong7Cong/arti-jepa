"""Per-epoch training/val curves for the binary fluent-vs-disfluent probe.

The main eval (``eval_stutter_binary.py``) logs only sparse points (every 10th
epoch) and keeps just the best-epoch summary. This utility re-runs the SAME probe
on the **cached** per-clip features (no re-extraction -> seconds/fold) while
recording train/val **loss** and **macro-F1** at *every* epoch, for every LOSO fold,
then writes the history (JSON + CSV) and a 4-panel figure next to the eval outputs.

It faithfully mirrors ``eval_disfluency.train_probe`` (same SegmentProbe, LR
schedule, optimizer, class weighting, val split) so the curves reflect the real run;
the extra per-epoch train/val evaluation is the only addition, so weight-init /
shuffle RNG differs slightly from the original run and best-epoch test metrics land
very close to (not bit-identical with) the reported numbers.

Run (needs the cache from a prior eval_stutter_binary run):
    python -m artijepa.plot_stutter_binary --config configs/eval_stutter_binary.yaml
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from artijepa import stutter as S
from artijepa.eval_stutter_binary import _tag, BINARY_CLASSES, load_config
from artijepa.eval_disfluency import SegmentProbe, _class_weights, _val_split


def _load_cache(cfg):
    cdir = os.path.join(cfg["meta"]["cache_dir"], _tag(cfg))
    fp, yp, kp = (os.path.join(cdir, f"all.{x}.npy") for x in ("feats", "label", "spk"))
    if not all(os.path.exists(x) for x in (fp, yp, kp)):
        raise SystemExit(f"[plot] no feature cache at {cdir} -- run eval_stutter_binary first")
    return (fp, np.load(yp).astype(np.int64), np.asarray(list(np.load(kp))), cdir)


def _preload_feats(fp, device):
    """Load the whole [N,L,D] fp16 feature cache ONTO the GPU once.

    Repeatedly streaming the ~8 MB/clip token grid host->device per batch is the
    bottleneck (it dominated the original eval too). With the full tensor resident on
    the GPU, every epoch/fold indexes on-device with zero transfers -> ~100x faster.
    Requires feats_fp16_bytes + a few GB headroom of free VRAM (checked by caller).
    """
    arr = np.load(fp)                                    # -> host RAM (fp16)
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    del arr
    return t                                             # [N,L,D] fp16 on GPU


# The attentive pool projects K/V from all 4096 tokens, so each forward is heavy.
# For the monitoring curves we run it under fp16 autocast (curves are unaffected) and
# batch large; exact metrics still come from eval_stutter_binary's fp32 run.
_AMP = dict(device_type="cuda", dtype=torch.float16, enabled=True)


@torch.no_grad()
def _eval_split(clf, Fg, idx, y, lossf, device, bs=256):
    """(mean weighted CE loss, macro-F1) over the on-GPU features Fg[idx] vs y[idx]."""
    clf.eval()
    tot_loss, n, preds = 0.0, 0, []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        x = Fg.index_select(0, torch.as_tensor(j, device=device))
        yy = torch.as_tensor(y[j], device=device)
        with torch.autocast(**_AMP):
            logit = clf(x)
            loss = lossf(logit, yy)
        tot_loss += float(loss) * len(j); n += len(j)
        preds.append(logit.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, np.int64)
    m = S.classification_metrics(y[idx], pred, len(BINARY_CLASSES), BINARY_CLASSES)
    return tot_loss / max(1, n), m["macro_f1"]


def train_fold_history(cfg, Fg, y, tr, va, te, spk_te, device):
    """Train the probe on one fold (on-GPU feats), returning per-epoch history + test.

    Mirrors ``eval_disfluency.train_probe`` (same SegmentProbe / schedule / optimizer /
    balanced CE / batch size); the only differences are on-GPU indexing and the extra
    per-epoch train+val evaluation used to draw the curves.
    """
    pc = cfg["probe"]; nc = len(BINARY_CLASSES); dim = Fg.shape[-1]
    bs = pc.get("batch_size", 64)
    clf = SegmentProbe(dim, nc, kind=pc.get("type", "attentive"),
                       hidden=pc.get("hidden", 512), heads=pc.get("heads", 8),
                       dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    w = _class_weights(y[tr], nc, device) if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    tr_gpu = torch.as_tensor(tr, device=device)
    g = torch.Generator(device=device); g.manual_seed(cfg["meta"].get("seed", 0))
    scaler = torch.cuda.amp.GradScaler()
    # per-epoch train macro-F1 is a monitoring curve -> estimate on a fixed random
    # subsample of train (full train each epoch would dominate runtime).
    rng = np.random.default_rng(cfg["meta"].get("seed", 0))
    tr_eval = tr if len(tr) <= 1500 else np.sort(rng.choice(tr, 1500, replace=False))

    hist = {k: [] for k in ("epoch", "lr", "train_loss", "val_loss", "train_f1", "val_f1")}
    best = {"val_f1": -1.0, "epoch": -1, "test_f1": None}
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for gg in opt.param_groups:
            gg["lr"] = lr
        clf.train(); run = nb = 0
        perm = tr_gpu[torch.randperm(len(tr_gpu), generator=g, device=device)]
        for i in range(0, len(perm), bs):
            bidx = perm[i:i + bs]
            x = Fg.index_select(0, bidx)
            yy = torch.as_tensor(y[bidx.cpu().numpy()], device=device)
            opt.zero_grad()
            with torch.autocast(**_AMP):
                loss = lossf(clf(x), yy)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss); nb += 1
        tr_loss = run / max(1, nb)
        vl, vf1 = _eval_split(clf, Fg, va, y, lossf, device)
        _, trf1 = _eval_split(clf, Fg, tr_eval, y, lossf, device)
        hist["epoch"].append(ep + 1); hist["lr"].append(round(lr, 7))
        hist["train_loss"].append(round(tr_loss, 5)); hist["val_loss"].append(round(vl, 5))
        hist["train_f1"].append(round(trf1, 5)); hist["val_f1"].append(round(vf1, 5))
        if vf1 > best["val_f1"]:
            _, tef1 = _eval_split(clf, Fg, te, y, lossf, device)
            best = {"val_f1": vf1, "epoch": ep + 1, "test_f1": tef1}
    print(f"[plot] fold {spk_te:6s}: best val-F1={best['val_f1']:.3f} "
          f"@e{best['epoch']} -> test-F1={best['test_f1']:.3f}", flush=True)
    return hist, best


def _plot(histories, bests, out_png, pooled_note=""):
    """4-panel: mean±std loss & macro-F1 curves + per-fold val-F1 and train-loss."""
    folds = sorted(histories)
    ep = np.asarray(histories[folds[0]]["epoch"])
    def stack(key):
        return np.vstack([histories[f][key] for f in folds])
    tr_l, va_l = stack("train_loss"), stack("val_loss")
    tr_f, va_f = stack("train_f1"), stack("val_f1")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    def band(a, arr, color, label):
        m, sd = arr.mean(0), arr.std(0)
        a.plot(ep, m, color=color, lw=2, label=label)
        a.fill_between(ep, m - sd, m + sd, color=color, alpha=0.18)

    band(ax[0, 0], tr_l, "#1f77b4", "train"); band(ax[0, 0], va_l, "#d62728", "val")
    ax[0, 0].set_title("Loss per epoch (mean ± std over 7 folds)")
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("weighted CE loss"); ax[0, 0].legend()

    band(ax[0, 1], tr_f, "#1f77b4", "train"); band(ax[0, 1], va_f, "#d62728", "val")
    ax[0, 1].set_title("Macro-F1 per epoch (mean ± std over 7 folds)")
    ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("macro-F1"); ax[0, 1].legend()

    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(folds):
        c = cmap(i)
        ax[1, 0].plot(ep, histories[f]["val_f1"], color=c, lw=1.4, label=f)
        be = bests[f]["epoch"]
        ax[1, 0].scatter([be], [bests[f]["val_f1"]], color=c, s=28, zorder=3)
        ax[1, 1].plot(ep, histories[f]["train_loss"], color=c, lw=1.4, label=f)
    ax[1, 0].set_title("Val macro-F1 per held-out speaker (• = best epoch)")
    ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("val macro-F1")
    ax[1, 0].legend(ncol=2, fontsize=8)
    ax[1, 1].set_title("Train loss per held-out speaker")
    ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("train loss"); ax[1, 1].legend(ncol=2, fontsize=8)

    fig.suptitle("Stutter binary (fluent vs disfluent) — frozen T-SSL 256, attentive "
                 f"probe, LOSO{pooled_note}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    from artijepa.eval_disfluency import pool_mode_for_probe
    cfg["data"]["pool_mode"] = pool_mode_for_probe(cfg["probe"].get("type", "attentive"))

    seed = cfg["meta"].get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fp, y, spk, cdir = _load_cache(cfg)
    Fg = _preload_feats(fp, device)
    print(f"[plot] cache {cdir}  feats={tuple(Fg.shape)} {Fg.dtype} on {device} "
          f"({Fg.element_size()*Fg.nelement()/2**30:.1f} GiB)  "
          f"per-class={np.bincount(y).tolist()}", flush=True)

    val_frac = cfg["data"].get("val_frac", 0.15)
    keep = np.arange(len(y), dtype=np.int64)
    histories, bests = {}, {}
    for test_spk in sorted(set(spk.tolist())):
        te = keep[spk == test_spk]; tr_all = keep[spk != test_spk]
        if len(np.unique(y[tr_all])) < 2 or len(te) == 0:
            print(f"[plot] skip {test_spk}: degenerate"); continue
        tr, va = _val_split(tr_all, y, len(BINARY_CLASSES), val_frac, rng)
        h, b = train_fold_history(cfg, Fg, y, tr, va, te, test_spk, device)
        histories[test_spk] = h; bests[test_spk] = b

    mean_best = float(np.mean([b["test_f1"] for b in bests.values()]))
    out_dir = cfg["meta"]["out"]; os.makedirs(out_dir, exist_ok=True)
    tag = _tag(cfg)
    hp = os.path.join(out_dir, f"stutter_binary_{tag}_curves_s{seed}.json")
    json.dump({"histories": histories, "bests": bests,
               "mean_best_test_f1": round(mean_best, 4)}, open(hp, "w"), indent=2)
    # long-format CSV for easy re-plotting elsewhere
    cp = os.path.join(out_dir, f"stutter_binary_{tag}_curves_s{seed}.csv")
    with open(cp, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["speaker", "epoch", "lr", "train_loss", "val_loss", "train_f1", "val_f1"])
        for spk_te, h in histories.items():
            for i in range(len(h["epoch"])):
                wtr.writerow([spk_te, h["epoch"][i], h["lr"][i], h["train_loss"][i],
                              h["val_loss"][i], h["train_f1"][i], h["val_f1"][i]])
    png = os.path.join(out_dir, f"stutter_binary_{tag}_curves_s{seed}.png")
    _plot(histories, bests, png, pooled_note=f"  (mean best test macro-F1={mean_best:.3f})")
    print(f"[plot] history -> {hp}\n[plot] csv     -> {cp}")


if __name__ == "__main__":
    main()
