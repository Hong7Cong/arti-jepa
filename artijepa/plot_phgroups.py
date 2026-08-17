"""t-SNE + confusion matrices for the PHONEME-CLIP probes (Phase 3).

Companion to `artijepa/eval_phoneme_groups.py`. That script trains one clip-level
probe per task (vowels / consonants / manner / place) and writes metrics; this one
re-runs the saved probes over a cached split to produce the two figures:

  * **confusion matrix** -- rows = TRUE, cols = PREDICTED, row-normalized to recall.
    The diagonal is per-class recall; off-diagonal is the substitution structure.
  * **t-SNE** -- the clips projected to 2-D, colored by task class, in TWO reps:

      - **A (phoneme vector)** -- the bi-LSTM last hidden state (fwd+bwd concat),
        i.e. the single vector the linear head actually reads. This is the probe's
        DECISION space, so tight clusters are partly circular (the pooler + LSTM
        were trained on these labels). Matches the rep-A convention of the Phase-1/2
        `tsne_phonemes.py` figures.
      - **B (raw-pooled)** -- the frozen encoder's own tokens, mean-pooled over the
        S' spatial tokens AND over the clip's temporal tokens. NO trained probe
        touches it, so cluster structure here is a NON-circular statement about the
        encoder's intrinsic phoneme geometry.

`place` supports `--drop-place-glottal` (default ON): the Glottal class {h,hh} is
near-chance because glottal constriction is largely invisible in the mid-sagittal
field of view. Dropping it excludes Glottal-TRUE clips *and* masks the Glottal
logit, so predictions are re-argmaxed over the remaining 7 classes -- a genuine 7x7
matrix whose rows sum to support, not an 8x8 with a row sliced off. The re-scored
macro-F1 is printed (and differs from the 8-class number in the eval JSON).

Reads the task-agnostic ragged clip cache written by `eval_phoneme_groups.py`
(located via each probe .pt's `feature_tag`), so nothing is re-extracted.

Usage:
    cd /project2/shrikann_35/hongn/vjepa2
    source dev_artiJEPA/scripts/_env.sh
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.plot_phgroups \
        --tag tssl256comb215 --split test_lss --seed 0
    # -> eval/phgroups/figs/phgroups_confmat_<tag>_<split>_s0.png
    #    eval/phgroups/figs/phgroups_tsne<rep>_<tag>_<split>_s0.png
"""

import argparse
import json
import mmap
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from artijepa.eval_phoneme_groups import (
    ClipProbe, _ClipDS, _clip_collate, _LenBucketSampler, build_tasks, PLACE_ORDER,
)

DEFAULT_EVAL = "/scratch1/hongn/artijepa/eval/phgroups"
DEFAULT_CACHE = "/scratch1/hongn/artijepa/feat_cache/phgroups"
TASKS = ["vowels", "consonants", "manner", "place"]


def task_colors(task, names):
    """Match the Phase-1/2 figure convention in tsne_phonemes.py: tab10 for the
    small grouped label spaces (manner/place), tab20 for per-phoneme identity.

    `consonants` has 25 classes -- more than tab20 holds -- so it spills into
    tab20b rather than wrapping (a wrap would give 5 pairs of identical colors)."""
    if task in ("manner", "place", "vowcons"):
        pal = [plt.get_cmap("tab10")(i) for i in range(10)]
    else:
        pal = ([plt.get_cmap("tab20")(i) for i in range(20)]
               + [plt.get_cmap("tab20b")(i) for i in range(20)])
    assert len(names) <= len(pal), f"{task}: {len(names)} classes > {len(pal)} colors"
    return {c: pal[i] for i, c in enumerate(names)}


# --------------------------------------------------------------------------- #
# forward pass: logits + both representations, in ONE sweep over the cache
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_probe(clf, ds, device, bs, workers, amp):
    """-> (y_true, logits [N,C], repA [N,2H], repB [N,D]).

    repA is the bi-LSTM last hidden (what the head reads); repB is the raw encoder
    token mean-pooled over S' then over the clip's VALID temporal tokens (padding
    excluded via `lens`, otherwise short clips would be dragged toward zero)."""
    clf.eval()
    nw = 0 if ds.mem is not None else workers
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=_LenBucketSampler(ds.lengths, bs, shuffle=False),
        num_workers=nw, collate_fn=_clip_collate)
    ys, lg, ra, rb = [], [], [], []
    for x, y, lens in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            B, T, S, D = x.shape
            q = clf.pooler(x.reshape(B * T, S, D)).squeeze(1)
            h = clf.norm(q.reshape(B, T, D))
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                h, lens.cpu(), batch_first=True, enforce_sorted=False)
            _o, (hn, _c) = clf.rnn(packed)
            vec = torch.cat([hn[-2], hn[-1]], dim=1)             # [B,2H] = rep A
            logits = clf.head(vec)
        # rep B: mean over S', then mean over the clip's real temporal tokens only
        m = (torch.arange(T, device=x.device)[None, :] < lens.to(x.device)[:, None])
        pooled = x.mean(2)                                       # [B,T,D]
        rawb = (pooled * m[..., None]).sum(1) / lens.to(x.device)[:, None].float()
        ys.append(y.numpy())
        lg.append(logits.float().cpu().numpy())
        ra.append(vec.float().cpu().numpy())
        rb.append(rawb.float().cpu().numpy())
    return (np.concatenate(ys), np.concatenate(lg),
            np.concatenate(ra), np.concatenate(rb))


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_confmat(ax, cm, names, title):
    """rows = true, cols = predicted, row-normalized to recall (matches
    confmat_phonemes.py: Blues, vmin/vmax pinned to [0,1], cells annotated when
    the matrix is small enough to stay legible)."""
    rs = cm.sum(1, keepdims=True)
    Mn = np.divide(cm, np.maximum(rs, 1), where=rs > 0).astype(float)
    im = ax.imshow(Mn, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  (n={int(r)})" for n, r in zip(names, rs[:, 0])],
                       fontsize=7)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(title, fontsize=9)
    if len(names) <= 16:
        for i in range(len(names)):
            for j in range(len(names)):
                v = Mn[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "0.15")
    return im


def scatter(ax, xy, labs, names, colors, title):
    for ci, c in enumerate(names):
        m = labs == ci
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, lw=0,
                   color=colors[c], label=c)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def embed2d(X, pca_dims, perplexity, seed):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = X.astype(np.float32)
    if pca_dims and X.shape[1] > pca_dims:
        X = PCA(n_components=pca_dims, random_state=seed).fit_transform(X)
    perp = min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed).fit_transform(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tssl256comb215")
    ap.add_argument("--split", default="test_lss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="default = the 4 multiclass tasks; add `vowcons` for the "
                         "binary vowel-vs-consonant figures (needs its probe .pt)")
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=None, help="default <eval-dir>/figs")
    ap.add_argument("--drop-place-glottal", action="store_true", default=True,
                    help="place: exclude Glottal {h,hh} clips and mask the logit (7 classes)")
    ap.add_argument("--keep-place-glottal", dest="drop_place_glottal",
                    action="store_false")
    ap.add_argument("--per-class", type=int, default=600,
                    help="max clips per class fed to t-SNE (class balance)")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="DEBUG: keep only this many clips/task (strided, so all "
                         "classes survive) -- smoke-tests the whole path in ~1 min")
    args = ap.parse_args()

    out = args.out or os.path.join(args.eval_dir, "figs")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tasks = build_tasks()
    want = [t.strip() for t in args.tasks.split(",") if t.strip()]

    results = {}
    for tname in want:
        pt = os.path.join(args.eval_dir,
                          f"phgroups_{args.tag}_s{args.seed}_{tname}.pt")
        if not os.path.exists(pt):
            print(f"[plot] SKIP {tname}: no probe at {pt}", flush=True)
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        cdir = os.path.join(args.cache_dir, ck["feature_tag"])
        fp = os.path.join(cdir, f"{args.split}.feats.npy")
        if not os.path.exists(fp):
            print(f"[plot] SKIP {tname}: no cache {fp}", flush=True)
            continue
        feats = np.load(fp, mmap_mode="r")
        off = np.load(os.path.join(cdir, f"{args.split}.off.npy"))
        phon = np.load(os.path.join(cdir, f"{args.split}.phon.npy"))

        members, remap, names = tasks[tname]
        clf = ClipProbe(ck["dim"], ck["num_classes"], hidden=ck["hidden"],
                        layers=ck["layers"], heads=ck["heads"],
                        dropout=ck["dropout"]).to(device)
        clf.load_state_dict(ck["probe_state"])

        drop_ci = None
        if tname == "place" and args.drop_place_glottal:
            drop_ci = PLACE_ORDER.index("Glottal")
            members = {p for p in members if remap[p] != drop_ci}

        ds = _ClipDS(feats, off, phon, members, remap, preload=False)
        if args.limit and len(ds) > args.limit:
            k = max(1, len(ds) // args.limit)
            keep_j = list(range(0, len(ds), k))[: args.limit]
            ds.idx = [ds.idx[j] for j in keep_j]
            ds.labels = [ds.labels[j] for j in keep_j]
            ds.lengths = [ds.lengths[j] for j in keep_j]
        print(f"[plot] {tname}: {len(ds)} clips from {args.split}", flush=True)
        y, logits, repA, repB = run_probe(clf, ds, device, args.batch_size,
                                          args.workers, amp)
        try:
            feats._mmap.madvise(mmap.MADV_DONTNEED)   # bound RSS (see eval_phoneme)
        except (AttributeError, OSError):
            pass
        del clf; torch.cuda.empty_cache()

        if drop_ci is not None:
            # mask the dropped logit so predictions re-argmax over the kept classes,
            # then compact the label space so 0..C-2 index `names` contiguously
            logits[:, drop_ci] = -np.inf
            keep = [i for i in range(len(names)) if i != drop_ci]
            names = [names[i] for i in keep]
            old2new = {o: n for n, o in enumerate(keep)}
            y = np.array([old2new[int(v)] for v in y], dtype=np.int64)
            pred = np.array([old2new[int(v)] for v in logits.argmax(1)], dtype=np.int64)
        else:
            pred = logits.argmax(1)

        C = len(names)
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        # macro-F1 recomputed here: for `place` it is the 7-class re-scored number
        # and will NOT match the 8-class value in the eval JSON.
        f1s = []
        for c in range(C):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        macro = float(np.mean(f1s))
        print(f"[plot] {tname}: {C} classes  acc={float((y==pred).mean()):.4f}  "
              f"macro_f1={macro:.4f}", flush=True)

        # class-balanced subsample for t-SNE
        rng = np.random.RandomState(args.seed)
        sel = []
        for c in range(C):
            idx = np.where(y == c)[0]
            if len(idx) > args.per_class:
                idx = rng.choice(idx, args.per_class, replace=False)
            sel.append(idx)
        sel = np.concatenate(sel)
        results[tname] = dict(names=names, cm=cm, macro=macro, y=y, pred=pred,
                              sel=sel, repA=repA[sel], repB=repB[sel])

    if not results:
        print("[plot] nothing to plot"); return

    order = [t for t in want if t in results]
    sfx = f"{args.tag}_{args.split}_s{args.seed}"

    # ---- confusion matrices (one row, one panel per task) ----
    fig, axes = plt.subplots(1, len(order), figsize=(5.2 * len(order), 5.4),
                             squeeze=False, layout="constrained")
    for j, t in enumerate(order):
        r = results[t]
        extra = " (Glottal dropped)" if t == "place" and args.drop_place_glottal else ""
        plot_confmat(axes[0][j], r["cm"], r["names"],
                     f"{t}{extra} · {len(r['names'])}c\nmacro-F1 {r['macro']:.3f}")
    fig.suptitle(f"Phase-3 phoneme-clip confusion · {args.tag} · {args.split} "
                 f"(row-normalized = recall)", fontsize=11)
    fp = os.path.join(out, f"phgroups_confmat_{sfx}.png")
    fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {fp}", flush=True)

    # ---- t-SNE, one figure per representation ----
    for rep, key, desc in (("A", "repA", "phoneme vector (bi-LSTM last hidden, probe decision space)"),
                           ("B", "repB", "raw encoder tokens mean-pooled (no probe)")):
        fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 5.2),
                                 squeeze=False, layout="constrained")
        for j, t in enumerate(order):
            r = results[t]
            xy = embed2d(r[key], args.pca, args.perplexity, args.seed)
            labs = r["y"][r["sel"]]
            colors = task_colors(t, r["names"])
            extra = " (Glottal dropped)" if t == "place" and args.drop_place_glottal else ""
            scatter(axes[0][j], xy, labs, r["names"], colors, f"{t}{extra}")
            axes[0][j].legend(loc="upper right", fontsize=5.5, markerscale=2.4,
                              framealpha=0.85, ncol=1 if len(r["names"]) <= 8 else 2)
            print(f"[plot] tsne rep{rep} {t}: {xy.shape[0]} pts", flush=True)
        fig.suptitle(f"Phase-3 phoneme-clip t-SNE · rep {rep} = {desc}\n"
                     f"{args.tag} · {args.split}", fontsize=11)
        fp = os.path.join(out, f"phgroups_tsne{rep}_{sfx}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[plot] wrote {fp}", flush=True)

    js = os.path.join(out, f"phgroups_confmat_{sfx}.json")
    json.dump({t: {"class_names": results[t]["names"],
                   "confusion_matrix": results[t]["cm"].tolist(),
                   "f1_macro": results[t]["macro"],
                   "place_glottal_dropped": bool(t == "place" and args.drop_place_glottal)}
               for t in order}, open(js, "w"), indent=2)
    print(f"[plot] wrote {js}", flush=True)


if __name__ == "__main__":
    main()
