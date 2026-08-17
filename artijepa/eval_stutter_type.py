"""Frozen-encoder DISFLUENCY-TYPE classification (block / rep / pro), Task 8c.

The binary eval (``eval_stutter_binary.py``) answers *is this stretch disfluent*.
This one answers the harder follow-up: **which kind of disfluency is it** --
a **block** (silent articulatory hold), a **repetition** (repeated gesture cycles),
or a **prolongation** (a held, sustained gesture). All three are defined by
articulator *dynamics*, so this is the eval that actually stresses what a video
encoder is supposed to buy over a per-frame image encoder.

Label spaces (``--task``, via ``stutter.label_space``):
  ``type3`` (default) block / rep / pro                 -- 2007 clips, near-balanced
  ``type4``           block / rep / pro / fluent        -- + the duration-matched
                      fluent negatives, i.e. detect-and-type in one head
  ``type5``           block / rep / pro / osci / other  -- adds the rare tail (n=42/21)

Everything else is deliberately identical to the binary eval so the two numbers are
comparable: the same OpenCV (pal8-safe) ``stutter_binary`` row builder and clip
loader, the same frozen-encoder dispatch, the same LOSO protocol, the same
``SegmentProbe``. **The feature cache is shared with the binary eval** -- the type
rows are a subset of the binary rows and their clip windows are identical, so a
matching ``--tag``/geometry/probe combination is a cache HIT and costs no extraction.

What this script adds over the binary eval: **the trained probe weights are saved**
(``meta.out/probes/<tag>/...``), one checkpoint per fold, holding the best-val
state_dict plus everything needed to rebuild the probe (classes, dim, t_steps,
probe hyper-params, val macro-F1, epoch).

Run:
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_stutter_type \
        --config dev_artiJEPA/configs/eval_stutter_type.yaml \
        --checkpoint /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt \
        --tag tssl256comb215 --probe attentive_lstm --task type3
    ... --task type4          # add the fluent class (detection + typing in one head)
    ... --split fixed --test-speaker PWS10 --val-speaker PWS7
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from artijepa import stutter as S
from artijepa import stutter_binary as SB
from artijepa.eval_disfluency import (SegmentProbe, _FeatDS, _class_weights,
                                      _predict, _val_split, pool_mode_for_probe)
# Row building / encoder loading / feature extraction are shared verbatim with the
# binary eval -- that is what makes the feature cache interchangeable between them.
from artijepa.eval_stutter_binary import (_pin_threads, _tag, build_rows, extract,
                                          load_config, load_frozen_encoder)

TYPE_TASKS = ("type3", "type4", "type5")


# --------------------------------------------------------------------------- #
# probe training (binary's train_probe + checkpoint saving + per-epoch logging)
# --------------------------------------------------------------------------- #
def train_probe_ckpt(cfg, feats, y, tr, va, te, spk_te, device, classes, ckpt_dir):
    """Train ``SegmentProbe`` on ``tr``, model-select on ``va`` by macro-F1, predict ``te``.

    Identical optimisation to ``eval_disfluency.train_probe`` (AdamW, warmup+cosine,
    optional balanced CE, best-val model selection) with three additions:

      * the best-val ``state_dict`` is kept in memory and written to
        ``ckpt_dir/fold_<spk_te>.pt`` together with the config needed to rebuild it;
      * per-epoch val metrics are logged (macro-F1, kappa, per-class recall) so a
        collapsed run is visible in the log rather than only in the final number;
      * optional ``probe.patience`` early stopping (0 = off) -- the dynamic binary
        runs showed folds that peak by epoch ~3 and then sit at chance for 25 more.

    Returns ``(test_metrics, test_pred, best_val_f1, ckpt_path)``.
    """
    pc = cfg["probe"]; num_classes = len(classes)
    dim = feats.shape[-1]
    t_steps = cfg["data"]["frames_per_clip"] // cfg["data"].get("tubelet_size", 2)
    probe_kw = dict(kind=pc.get("type", "attentive_lstm"), hidden=pc.get("hidden", 512),
                    heads=pc.get("heads", 8), dropout=pc.get("dropout", 0.1),
                    t_steps=t_steps, lstm_hidden=pc.get("lstm_hidden", 256),
                    lstm_layers=pc.get("lstm_layers", 1),
                    bidirectional=pc.get("bidirectional", True),
                    temporal_pool=pc.get("temporal_pool", "mean"),
                    chunk=pc.get("chunk", 0), checkpoint=pc.get("checkpoint", False))
    clf = SegmentProbe(dim, num_classes, **probe_kw).to(device)
    n_par = sum(p.numel() for p in clf.parameters())
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    w = _class_weights(y[tr], num_classes, device) if pc.get("class_weight") == "balanced" else None
    lossf = nn.CrossEntropyLoss(weight=w)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    patience = int(pc.get("patience", 0))
    loader = torch.utils.data.DataLoader(
        _FeatDS(feats, tr, y[tr]), batch_size=pc.get("batch_size", 64), shuffle=True,
        num_workers=cfg["data"].get("probe_workers", 2), drop_last=False)
    print(f"[type-probe {spk_te}] train={len(tr)} val={len(va)} test={len(te)} "
          f"| train per-class {np.bincount(y[tr], minlength=num_classes).tolist()} "
          f"| test per-class {np.bincount(y[te], minlength=num_classes).tolist()} "
          f"| probe {probe_kw['kind']} {n_par/1e6:.2f}M params")

    os.makedirs(ckpt_dir, exist_ok=True)
    cp = os.path.join(ckpt_dir, f"fold_{spk_te}.pt")

    def _save(best):
        """Persist the best-val probe. Written on EVERY improvement (not just at the
        end of the fold) so a wall-clock kill / OOM mid-fold still leaves the best
        weights on disk. Atomic via tmp+rename so a kill during the write cannot
        leave a truncated .pt behind."""
        torch.save({"state_dict": best["state"], "probe_kwargs": probe_kw, "dim": int(dim),
                    "num_classes": num_classes, "classes": list(classes),
                    "task": cfg["data"].get("task", "type3"),
                    "t_steps": t_steps, "fold": spk_te, "seed": cfg["meta"].get("seed", 0),
                    "val_macro_f1": best["val_f1"], "epoch": best["epoch"],
                    "test_metrics": best["test"], "encoder_tag": cfg["meta"].get("tag"),
                    "feature_tag": _tag(cfg), "n_params": int(n_par)}, cp + ".tmp")
        os.replace(cp + ".tmp", cp)

    best = {"val_f1": -1.0, "test": None, "pred": None, "epoch": 0, "state": None}
    since = 0
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0; t0 = time.time()
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
            best = {"val_f1": vm["macro_f1"], "test": tm, "pred": tp, "epoch": ep + 1,
                    "train_loss": run / max(1, nb),
                    "state": {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}}
            _save(best)
            since = 0
        else:
            since += 1
        rec = " ".join(f"{c}={vm['per_class'][c]['recall'] if vm['per_class'][c]['recall'] is not None else float('nan'):.2f}"
                       for c in classes)
        print(f"[type-probe {spk_te} e{ep+1}/{epochs}] tr_loss={run/max(1,nb):.3f} "
              f"val_macroF1={vm['macro_f1']:.3f} val_kappa={vm['cohen_kappa']:.3f} "
              f"best={best['val_f1']:.3f}@e{best['epoch']} ({time.time()-t0:.0f}s) "
              f"| val recall {rec}", flush=True)
        if patience and since >= patience:
            print(f"[type-probe {spk_te}] early stop at e{ep+1} "
                  f"({patience} epochs without a val improvement)")
            break

    if best["state"] is None:
        raise SystemExit(f"[type-probe {spk_te}] no epoch completed -- nothing to save "
                         f"(epochs={epochs})")
    _save(best)                                   # final rewrite: cheap, keeps it obvious
    print(f"[type-probe {spk_te}] saved probe -> {cp} "
          f"(best val macro-F1 {best['val_f1']:.3f} @ epoch {best['epoch']})")
    return best["test"], best["pred"], best["val_f1"], cp


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _type_labels(rows, task, classes):
    """Per-row class index for ``task`` (None where the row is not in the label space).

    ``rows`` is the FULL binary row list, so this stays index-aligned with the shared
    feature cache (which the binary eval extracted over every row).
    """
    return [S.row_label(r, task, classes) for r in rows]


def run(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    _pin_threads(cfg["data"].get("cpu_threads", 2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "float16").lower(), torch.float32)

    task = cfg["data"].get("task", "type3")
    if task not in TYPE_TASKS:
        raise SystemExit(f"[type-eval] --task must be one of {TYPE_TASKS}, got {task!r}")
    classes, _ = S.label_space(task)
    ptype = cfg["probe"].get("type", "attentive_lstm")
    cfg["data"]["pool_mode"] = pool_mode_for_probe(ptype)
    print(f"[type-eval] task={task} classes={classes} probe={ptype} "
          f"pool_mode={cfg['data']['pool_mode']} device={device}")

    # Rows + features are built EXACTLY as in the binary eval (same _tag inputs), so
    # an existing binary cache for this encoder/geometry/probe is reused as-is.
    rows, row_stats = build_rows(cfg)
    ylab = _type_labels(rows, task, classes)
    keep_rows = [i for i, v in enumerate(ylab) if v is not None]
    print(f"[type-eval] {len(keep_rows)}/{len(rows)} rows in the {task} label space; "
          f"per-class {np.bincount([ylab[i] for i in keep_rows], minlength=len(classes)).tolist()}")

    encoder = load_frozen_encoder(cfg, device)   # may mutate geometry / intensity_norm
    feats, ybin, speakers = extract(encoder, cfg, rows, device, dtype)
    del encoder; torch.cuda.empty_cache()
    if len(ybin) != len(rows):
        raise SystemExit(f"[type-eval] cache/rows mismatch: {len(ybin)} feats vs "
                         f"{len(rows)} rows -- rebuild the cache")
    # alignment guard: the cached binary label must agree with the row's own bucket5
    exp = np.asarray([0 if r["bucket5"] == S.FLUENT else 1 for r in rows], dtype=np.int64)
    if not np.array_equal(np.asarray(ybin, dtype=np.int64), exp):
        raise SystemExit("[type-eval] feature cache is NOT row-aligned (binary labels "
                         "differ from the rebuilt rows) -- delete the cache and re-extract")

    keep = np.asarray(keep_rows, dtype=np.int64)
    y = np.full(len(rows), -1, dtype=np.int64)
    for i in keep_rows:
        y[i] = ylab[i]
    spk = np.asarray(speakers)
    uniq_spk = sorted(set(spk[keep].tolist()))

    split_mode = cfg["data"].get("split_mode", "loso")
    val_frac = cfg["data"].get("val_frac", 0.15)
    ckpt_dir = os.path.join(meta["out"], "probes",
                            f"{_tag(cfg)}_{ptype}_{task}_{split_mode}_s{seed}")
    folds, all_true, all_pred, all_spk = [], [], [], []

    def _fold(tr, va, te, name):
        tm, pred, vf1, cp = train_probe_ckpt(cfg, feats, y, tr, va, te, name,
                                             device, classes, ckpt_dir)
        folds.append({"speaker": name, "n_test": int(len(te)), "val_f1": round(vf1, 4),
                      "probe_ckpt": cp, **tm})
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())
        all_spk.extend([name] * len(te))

    if split_mode == "loso":
        for test_spk in uniq_spk:
            te = keep[spk[keep] == test_spk]
            tr_all = keep[spk[keep] != test_spk]
            if len(te) == 0 or len(np.unique(y[tr_all])) < 2:
                print(f"[type-eval] skip fold {test_spk}: degenerate")
                continue
            miss = [c for ci, c in enumerate(classes) if (y[te] == ci).sum() == 0]
            if miss:
                print(f"[type-eval] NOTE fold {test_spk}: class(es) {miss} absent from "
                      f"the held-out set -- macro-F1 averages the present classes only")
            tr, va = _val_split(tr_all, y, len(classes), val_frac, rng)
            _fold(tr, va, te, test_spk)
    elif split_mode == "fixed":
        test_spk = cfg["data"]["test_speaker"]; val_spk = cfg["data"].get("val_speaker")
        te = keep[spk[keep] == test_spk]
        if val_spk:
            va = keep[spk[keep] == val_spk]
            tr = keep[(spk[keep] != test_spk) & (spk[keep] != val_spk)]
        else:
            tr, va = _val_split(keep[spk[keep] != test_spk], y, len(classes), val_frac, rng)
        if len(te) == 0 or len(tr) == 0:
            raise SystemExit(f"[type-eval] fixed split degenerate "
                             f"(test={test_spk!r} val={val_spk!r})")
        _fold(tr, va, te, test_spk)
    else:                                   # random stratified 60/20/20 over clips
        perm = keep.copy(); rng.shuffle(perm)
        te = perm[: int(0.2 * len(perm))]; rest = perm[int(0.2 * len(perm)):]
        tr, va = _val_split(rest, y, len(classes), val_frac, rng)
        _fold(tr, va, te, "random")

    if not folds:
        raise SystemExit("[type-eval] no usable folds")
    pooled = S.classification_metrics(np.asarray(all_true), np.asarray(all_pred),
                                      len(classes), classes)
    ec = cfg["encoder"]
    out = {"encoder": cfg["meta"].get("tag"), "type": ec.get("type", "vjepa"),
           "mode": "frozen", "tag": cfg["meta"].get("tag"), "task": task,
           "classes": classes, "probe": ptype,
           "spatial_size": cfg["data"]["spatial_size"],
           "frames_per_clip": cfg["data"]["frames_per_clip"], "seed": seed,
           "split_mode": split_mode, "probe_ckpt_dir": ckpt_dir,
           "n_clips": int(len(keep)),
           "class_counts": np.bincount(y[keep], minlength=len(classes)).tolist(),
           "row_stats": {k: row_stats[k] for k in ("n_pos", "n_neg", "pos_dur", "neg_dur")},
           "pooled": pooled,
           "macro_f1_mean": round(float(np.mean([f["macro_f1"] for f in folds])), 4),
           "balanced_acc_mean": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
           "accuracy_mean": round(float(np.mean([f["accuracy"] for f in folds])), 4),
           "folds": folds}
    _report(cfg, out)
    return out


def _report(cfg, out):
    print(f"\n===== STUTTER {out['task'].upper()} ({'/'.join(out['classes'])}) RESULT =====")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=2))
    print("[type-eval] per-fold macro-F1:",
          {f["speaker"]: f["macro_f1"] for f in out["folds"]})
    rp = os.path.join(cfg["meta"]["out"],
                      f"stutter_{out['task']}_{_tag(cfg)}_{out['probe']}_"
                      f"{out['split_mode']}_s{out['seed']}.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"[type-eval] wrote {rp}")
    print(f"[type-eval] probe checkpoints in {out['probe_ckpt_dir']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, choices=list(TYPE_TASKS),
                    help="label space (default type3 = block/rep/pro)")
    ap.add_argument("--enc-type", default=None,
                    choices=["vjepa", "videomae", "image_baseline"])
    ap.add_argument("--model", default=None, help="image_baseline alias: vitl|dinov2|...")
    ap.add_argument("--spec", default=None, help="vjepa: 'pretrained' or a local .pt")
    ap.add_argument("--videomae-name", default=None)
    ap.add_argument("--videomae-checkpoint", default=None)
    ap.add_argument("--grid-cap", type=int, default=None)
    ap.add_argument("--checkpoint", default=None, help="T-SSL/V-JEPA checkpoint (.pt)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--probe", default=None,
                    choices=["attentive", "pooled_attentive", "attentive_lstm", "mean", "mlp"])
    ap.add_argument("--lstm-hidden", type=int, default=None)
    ap.add_argument("--lstm-layers", type=int, default=None)
    ap.add_argument("--lstm-chunk", type=int, default=None)
    ap.add_argument("--lstm-checkpoint", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after N epochs without a val improvement (0=off)")
    ap.add_argument("--split", default=None, choices=["loso", "fixed", "random"])
    ap.add_argument("--test-speaker", default=None)
    ap.add_argument("--val-speaker", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--neg-per-pos", type=float, default=None,
                    help="fluent negatives per positive (only used by --task type4)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.task is not None:
        cfg["data"]["task"] = args.task
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
    if args.epochs is not None:
        cfg["probe"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["probe"]["patience"] = args.patience
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
