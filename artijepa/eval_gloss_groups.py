"""Pre/post-glossectomy SEGMENT-level phoneme classification + clustering (Test 4).

`eval_gloss.py` transfers a **frame-level** probe: a window slides over the utterance,
every 40 ms token gets a label, score = kappa over tokens. This driver redoes the *same*
eval-only transfer at the **segment level** with the phonePred **Phase-3** clip probes
(`eval_phoneme_groups.py`, RESULTS_phonepred.md "Phoneme-CLIP classification (Phase 3)"):

    one gold phoneme segment = one sample = ONE label,
    in four label spaces: vowels(15) | consonants(25) | manner(5) | place(8).

Why it is not a re-run of eval_gloss.py (docs/GLOSS.md §12.1): gold boundaries make this an
identity-only (oracle-segmentation) read, so it separates *which phoneme* from *when the
boundary falls*; the Phase-3 label spaces contain **no `sil` class**, so the frame-level
"uniform smear toward Vowel/sil" cannot hide there; and place/manner are the probe's own
output space -- where the physical-articulator hypothesis actually makes a prediction.

NO RETRAINING. The Annot-16-trained Phase-3 probes are loaded from
`eval/phgroups/phgroups_<enc>_s<seed>_<task>.pt` and applied to gloss features; their own
`test` (in-domain Annot-16 sub043) and `test_lss` (healthy-OOD usc_lss) metrics ship in the
checkpoint and become the top two rungs of the ladder. Everything is reported pooled AND per
**speaker x session** -- spk2 has two post-op sessions (`post1`, `post2`).

Pipeline (all reused, no new model code):
  1. per condition: point `data.manifest` at the gloss manifest and `meta.tag` at
     `glossphg_<cond>_<enc>` -> `eval_phoneme_groups.extract` builds its own ragged clip
     cache (the tag hash keys on the manifest, so no collision with Annot-16);
     `clip_rows` gives each clip's manifest row -> speaker/session label.
  2. per task: `load_probe` the .pt, `_ClipDS` filters+relabels the shared cache,
     predict -> F1/P/R/kappa/per-class/confusion, pooled and per speaker/session.
  3. `--cluster`: silhouette (by phoneme / speaker / condition), k-means ARI+NMI, and a
     pre->post nearest-centroid **drift matrix** with a leave-one-utterance-out pre-vs-pre
     baseline -- on rep A (probe space) and rep B (raw encoder geometry, probe-independent).

spk3's post video FILENAMES were corrected in place on 2026-08-06 (see RESULTS_gloss.md),
so the canonical manifest now carries the full n=16 directly and `--spk3-repaired` defaults
to `none`. The repaired manifests are keyed to the OLD (wrong) filenames and are OBSOLETE --
do not pass `all`/`high` unless you have regenerated them. Historically it swapped in the
duration-repaired manifest (n=16) built by
`gloss_repair_pairs.py`. See RESULTS_gloss.md "spk3 post-op recovered by duration re-pairing".

Usage:
    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    python -m artijepa.gloss                       # only if the manifests are missing
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_gloss_groups \
        --config dev_artiJEPA/configs/eval_gloss_groups.yaml \
        --encoder tssl256comb215 --seed 0 --cluster --figs
"""

import argparse
import copy
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from artijepa.eval_gloss import _speaker_groups
from artijepa.eval_phoneme import load_frozen_encoder
from artijepa.eval_phoneme_groups import (
    build_tasks, extract, clip_rows, load_config, load_probe, kappa_from_cm,
    _metrics, _tag, _ClipDS, _clip_collate, _LenBucketSampler)

CONDS = ["pre", "post"]
GLOSS_ROOT = "/scratch1/hongn/gloss"
ARTI_OUT = os.environ.get("ARTI_OUT", "/scratch1/hongn/artijepa")
TASK_ORDER = ["vowels", "consonants", "manner", "place"]
# (per-panel width, height, fontscale) per task -- mirrors confmat_phonemes.PANEL
PANEL = {"vowels": (5.0, 4.6, 0.9), "consonants": (7.6, 7.0, 0.72),
         "manner": (3.4, 3.2, 1.0), "place": (4.2, 4.0, 0.95)}


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def _rows(path):
    return list(csv.DictReader(open(path)))


def _write_rows(rows, dst, cols):
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return dst


def cond_manifest(cond, speaker, scratch, spk3_repaired="none"):
    """-> (manifest_path, rows). Optionally speaker-filtered, and for `post` optionally
    with spk3's scrambled 3-utterance session replaced by the duration-repaired one.

    The repaired manifest carries extra columns (tg_source/residual_s/confidence) and
    points at `phonemes_json_repaired/`; only the canonical columns are written out, so
    the merged file is consumed by `_flatten_segments` exactly like the original."""
    src = os.path.join(GLOSS_ROOT, f"gloss_{cond}_manifest.csv")
    if not os.path.exists(src):
        raise SystemExit(f"[glossphg] missing {src}; run `python -m artijepa.gloss` first")
    rows = _rows(src)
    cols = list(rows[0].keys())
    name = f"gloss_{cond}"
    if cond == "post" and spk3_repaired != "none":
        rp = os.path.join(GLOSS_ROOT, "gloss_post_spk3_repaired_manifest.csv")
        if not os.path.exists(rp):
            raise SystemExit(f"[glossphg] missing {rp}; run `python -m artijepa."
                             f"gloss_repair_pairs` or pass --spk3-repaired none")
        rep = _rows(rp)
        if spk3_repaired == "high":
            rep = [r for r in rep if r.get("confidence") == "high"]
        rows = [r for r in rows if r["speaker"] != "spk3"] + rep
        name += f"_spk3rep{spk3_repaired}"
        print(f"[glossphg] {cond}: spk3 -> duration-repaired manifest "
              f"({len(rep)} utts, --spk3-repaired {spk3_repaired})")
    if speaker:
        rows = [r for r in rows if r["speaker"] == speaker]
        name += f"_{speaker}"
    if not rows:
        raise SystemExit(f"[glossphg] no rows for cond={cond} speaker={speaker}")
    # always rewrite: row ORDER defines the cache's clip order, so it must be pinned
    return _write_rows(rows, os.path.join(scratch, f"{name}.csv"), cols), rows


# --------------------------------------------------------------------------- #
# feature cache per condition (+ each clip's speaker/session label)
# --------------------------------------------------------------------------- #
def _cache_ready(cfg):
    name, split = _tag(cfg, "test")
    cdir = os.path.join(cfg["meta"]["cache_dir"], name)
    return all(os.path.exists(os.path.join(cdir, f"{split}.{x}.npy"))
               for x in ("feats", "off", "phon"))


def condition_cfg(base_cfg, cond, manifest, tag):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["manifest"] = manifest
    cfg["data"]["pool_spatial"] = False          # the clip head needs the S' grid
    cfg["meta"]["tag"] = tag
    return cfg


def load_condition(cfg, cond, rows, encoder, device, dtype):
    """-> dict with the ragged cache, each clip's manifest row, and its speaker/session."""
    feats, off, phon = extract(encoder, cfg, "test", device, dtype)
    row = clip_rows(cfg, "test")
    grp_of_utt = _speaker_groups(rows, cond)                  # utt row idx -> "spk2/post1"
    grp = np.array([grp_of_utt[int(r)] for r in row], dtype=object)
    print(f"[glossphg] {cond}: {len(phon)} segments over {len(rows)} utts; "
          f"cells " + " ".join(f"{g}={int((grp == g).sum())}"
                               for g in sorted(set(grp_of_utt))))
    return {"feats": feats, "off": off, "phon": phon, "row": row, "grp": grp,
            "rows": rows, "n_utt": len(rows)}


# --------------------------------------------------------------------------- #
# inference (dataset-ordered, so predictions can be sliced per speaker/session)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_task(clf, ds, device, bs, workers, amp, want_vec=False):
    """-> (y_true, y_pred, vecA|None) in DATASET order.

    `_predict` in eval_phoneme_groups returns bucket order (fine for pooled metrics,
    wrong for per-speaker slicing), so the length-bucketed batches are materialised here
    and the outputs are un-permuted."""
    clf.eval()
    batches = list(_LenBucketSampler(ds.lengths, bs, shuffle=False))
    nw = 0 if ds.mem is not None else workers
    loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=nw,
                                         collate_fn=_clip_collate)
    ps, ts, vs = [], [], []
    for x, y, lens in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = clf(x, lens, return_vec=want_vec)
        logits, vec = out if want_vec else (out, None)
        ps.append(logits.float().argmax(-1).cpu().numpy())
        ts.append(y.numpy())
        if want_vec:
            vs.append(vec.float().cpu().numpy())
    order = np.argsort(np.concatenate([np.asarray(b) for b in batches]))
    y_t = np.concatenate(ts)[order]
    y_p = np.concatenate(ps)[order]
    vec = np.concatenate(vs)[order] if want_vec else None
    return y_t, y_p, vec


def rep_b(feats, off, cache_idx):
    """Probe-independent per-segment vector: mean over the clip's valid temporal tokens
    AND over the S' spatial tokens -> [n, D]. The fair cross-speaker view (the same
    rep A / rep B split used for the frame-level gloss t-SNEs)."""
    D = feats.shape[-1]
    out = np.zeros((len(cache_idx), D), dtype=np.float32)
    for k, i in enumerate(cache_idx):
        out[k] = np.asarray(feats[off[i]:off[i + 1]], dtype=np.float32).mean((0, 1))
    return out


def _cell_metrics(y_t, y_p, grp, n_classes, names, n_utt_of):
    """Pooled + per speaker/session metrics for one task x condition."""
    pooled = _metrics(y_t, y_p, n_classes, names)
    pooled["n_utt"] = int(sum(n_utt_of.values()))
    per = {}
    for g in sorted(set(grp.tolist())):
        m = grp == g
        if not m.any():
            continue
        per[g] = _metrics(y_t[m], y_p[m], n_classes, names)
        per[g]["n_utt"] = int(n_utt_of.get(g, 0))
    return pooled, per


# --------------------------------------------------------------------------- #
# clustering readout
# --------------------------------------------------------------------------- #
def cluster_stats(X, y_class, y_spk, y_cond, n_classes, seed, cap):
    """Silhouette under three different labelings of the SAME points + k-means agreement.

    silhouette(speaker) >> silhouette(phoneme) is the geometric form of "the OOD gap is
    speaker/appearance dominated" (docs/GLOSS.md §12.6)."""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rs = np.random.RandomState(seed)
    idx = np.arange(len(X))
    if len(idx) > cap:
        idx = rs.choice(idx, cap, replace=False)
    Xs = X[idx].astype(np.float32)
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-6)     # cosine geometry
    out = {"n": int(len(idx))}
    for nm, lab in (("phoneme", y_class[idx]), ("speaker", y_spk[idx]),
                    ("condition", y_cond[idx])):
        u = np.unique(lab)
        out[f"silhouette_{nm}"] = (round(float(silhouette_score(Xs, lab)), 4)
                                   if 1 < len(u) < len(idx) else None)
        out[f"n_{nm}"] = int(len(u))
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(Xs)
    out["kmeans_ari"] = round(float(adjusted_rand_score(y_class[idx], km)), 4)
    out["kmeans_nmi"] = round(float(normalized_mutual_info_score(y_class[idx], km)), 4)
    return out


def _class_sums(X, y, utt, n_classes):
    """-> (sum per class [C,D], count per class [C], per-utterance sums/counts dicts)."""
    C, D = n_classes, X.shape[1]
    tot, cnt = np.zeros((C, D), np.float64), np.zeros(C, np.float64)
    per_u = {}
    for i in range(len(X)):
        c = int(y[i]); tot[c] += X[i]; cnt[c] += 1
        s, n = per_u.setdefault(int(utt[i]), (np.zeros((C, D)), np.zeros(C)))
        s[c] += X[i]; n[c] += 1
    return tot, cnt, per_u


def _assign(X, cent, ok):
    """Nearest centroid (euclidean) among classes with `ok` support -> class index."""
    d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
    d[:, ~ok] = np.inf
    return d.argmin(1)


def drift_matrix(Xpre, ypre, uttpre, Xpost, ypost, n_classes):
    """Row i, col j = fraction of class i's POST segments whose nearest class centroid,
    computed on PRE segments, is class j. Diagonal = self-retention; a large off-diagonal
    i->j is class i drifting into j (docs/GLOSS.md §12.6, the §10 Task-1 idea at segment
    granularity). Returns (matrix [C,C], post support [C])."""
    tot, cnt, _ = _class_sums(Xpre, ypre, uttpre, n_classes)
    ok = cnt > 0
    cent = np.divide(tot, np.where(cnt[:, None] == 0, 1, cnt[:, None]))
    a = _assign(Xpost, cent, ok)
    M = np.zeros((n_classes, n_classes))
    for t, p in zip(ypost.astype(int), a):
        M[t, p] += 1
    return M, M.sum(1)


def drift_baseline(Xpre, ypre, uttpre, n_classes):
    """Same matrix built PRE-vs-PRE with leave-one-utterance-out centroids, so a class
    that was already confusable before surgery is not read as a post-op change."""
    tot, cnt, per_u = _class_sums(Xpre, ypre, uttpre, n_classes)
    M = np.zeros((n_classes, n_classes))
    for u, (su, nu) in per_u.items():
        t2, c2 = tot - su, cnt - nu
        ok = c2 > 0
        cent = np.divide(t2, np.where(c2[:, None] == 0, 1, c2[:, None]))
        m = uttpre == u
        if not m.any() or not ok.any():
            continue
        for t, p in zip(ypre[m].astype(int), _assign(Xpre[m], cent, ok)):
            M[t, p] += 1
    return M, M.sum(1)


def _rownorm(M):
    s = M.sum(1, keepdims=True)
    return M / np.where(s == 0, 1, s)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _cm_panel(ax, M, names, title, fscale):
    Mn = _rownorm(np.asarray(M, dtype=np.float64))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fs = max(3.5, 7 * fscale)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=fs)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=fs)
    ax.set_xlabel("predicted", fontsize=fs + 1); ax.set_ylabel("true", fontsize=fs + 1)
    if len(names) ** 2 <= 20 * 22:
        for i in range(len(names)):
            for j in range(len(names)):
                if Mn[i, j] >= 0.005:
                    ax.text(j, i, f"{int(round(Mn[i, j] * 100))}", ha="center",
                            va="center", fontsize=max(3, fs - 2.5),
                            color="white" if Mn[i, j] > 0.5 else "0.15")
    ax.set_title(title, fontsize=fs + 2)
    return im


def _finish(fig, im, fp, label="recall"):
    fig.tight_layout(rect=[0, 0, 0.97, 0.94])
    cax = fig.add_axes([0.975, 0.12, 0.008, 0.72])
    fig.colorbar(im, cax=cax, label=label)
    fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"[glossphg] wrote {fp}")


def plot_cm_pooled(res, task, names, out_dir, enc, seed, sfx):
    pw, ph, fs = PANEL[task]
    fig, axes = plt.subplots(1, len(CONDS), figsize=(pw * len(CONDS), ph), squeeze=False)
    im = None
    for j, cond in enumerate(CONDS):
        m = res[cond][task]["pooled"]
        im = _cm_panel(axes[0][j], m["confusion_matrix"], names,
                       f"{cond}  (F1={m['f1_macro']:.3f}, κ={m['kappa']:.3f}, n={m['n']})",
                       fs)
    d = res["post"][task]["pooled"]["f1_macro"] - res["pre"][task]["pooled"]["f1_macro"]
    fig.suptitle(f"{enc} — {task}: segment-level confusion, pre vs post glossectomy "
                 f"(row-normalized recall %)   ΔF1(post−pre)={d:+.3f}", fontsize=12)
    _finish(fig, im, os.path.join(out_dir, f"confmat_glossphg_{task}_{enc}{sfx}_s{seed}.png"))


def _grid(res):
    """-> (speaker rows, session columns, {(spk, col): (cond, group)})."""
    cells = {}
    for cond in CONDS:
        for g in res[cond]["_groups"]:
            base, _, sess = g.partition("/")
            cells[(base, sess or cond)] = (cond, g)
    rows = sorted({r for r, _ in cells})
    cols = [c for c in ("pre", "post", "post1", "post2")
            if any((r, c) in cells for r in rows)]
    return rows, cols, cells


def plot_cm_per_speaker(res, task, names, out_dir, enc, seed, sfx):
    rows, cols, cells = _grid(res)
    pw, ph, fs = PANEL[task]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(pw * len(cols), ph * len(rows)), squeeze=False)
    im = None
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            if (r, c) not in cells:
                ax.axis("off"); continue
            cond, g = cells[(r, c)]
            m = res[cond][task]["per_group"].get(g)
            if not m:
                ax.axis("off"); continue
            im = _cm_panel(ax, m["confusion_matrix"], names,
                           f"{g}  (F1={m['f1_macro']:.3f}, κ={m['kappa']:.3f}, "
                           f"n={m['n']})", fs)
    fig.suptitle(f"{enc} — {task}: segment-level confusion per gloss speaker × session "
                 f"(row-normalized recall %)", fontsize=13)
    _finish(fig, im, os.path.join(out_dir,
                                  f"confmat_glossphg_{task}_perspk_{enc}{sfx}_s{seed}.png"))


def plot_tsne_grid(emb, task, names, out_dir, enc, seed, sfx, rep, method, cap,
                   perplexity, pca):
    """Speaker × session grid of the per-segment vectors, colored by the task's class."""
    from artijepa.tsne_phonemes import pca_reduce, embed
    rows, cols, cells = _grid(emb["_res"])
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(4.3 * len(cols), 4.5 * len(rows)), squeeze=False)
    rs = np.random.RandomState(seed)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
            if (r, c) not in cells:
                ax.axis("off"); continue
            cond, g = cells[(r, c)]
            E = emb[cond][task]
            m = E["grp"] == g
            n = int(m.sum())
            if n < 20:
                ax.set_title(f"{g}  (only {n} segments)", fontsize=9); ax.axis("off")
                continue
            idx = np.where(m)[0]
            if len(idx) > cap:
                idx = rs.choice(idx, cap, replace=False)
            X = E[rep][idx]
            xy = embed(pca_reduce(X, pca), method, seed, perplexity)
            y = E["y_true"][idx]
            for cl in range(len(names)):
                s = y == cl
                if s.any():
                    ax.scatter(xy[s, 0], xy[s, 1], s=6, alpha=0.6, linewidths=0,
                               color=cmap(cl % 20), label=names[cl])
            ax.set_title(f"{g}  (n={len(idx)})", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(k % 20), label=nm)
               for k, nm in enumerate(names)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(names), 10),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    lbl = "A: probe phoneme vector" if rep == "repA" else "B: raw encoder geometry"
    fig.suptitle(f"{enc} — {task}: segment {method} per speaker × session, rep {lbl}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fp = os.path.join(out_dir,
                      f"tsne_glossphg_{task}_{rep}_{enc}{sfx}_s{seed}.png")
    fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"[glossphg] wrote {fp}")


def plot_drift(dr, task, names, spk, out_dir, enc, seed, sfx, rep):
    fig, axes = plt.subplots(1, 2, figsize=(PANEL[task][0] * 2, PANEL[task][1]),
                             squeeze=False)
    im = None
    for j, (nm, key) in enumerate((("baseline: pre→pre (LOUO)", "baseline"),
                                   ("post→pre centroids", "drift"))):
        M = np.asarray(dr[key]["matrix"], dtype=np.float64)
        im = _cm_panel(axes[0][j], M, names,
                       f"{nm}   self-retention={dr[key]['self_retention']:.3f} "
                       f"(n={int(M.sum())})", PANEL[task][2])
    fig.suptitle(f"{enc} — {spk} {task}: nearest-centroid drift, rep {rep[-1]} "
                 f"(Δself-retention={dr['delta_self_retention']:+.3f})", fontsize=12)
    _finish(fig, im, os.path.join(
        out_dir, f"drift_glossphg_{task}_{spk}_{rep}_{enc}{sfx}_s{seed}.png"),
        label="fraction")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_gloss_groups.yaml")
    ap.add_argument("--encoder", default="tssl256comb215",
                    help="Phase-3 tag: names the probes (phgroups_<tag>_s<seed>_<task>.pt) "
                         "and the outputs")
    ap.add_argument("--seed", type=int, default=0, help="which Phase-3 probe seed")
    ap.add_argument("--probe-dir", default=os.path.join(ARTI_OUT, "eval", "phgroups"))
    ap.add_argument("--out-dir", default=os.path.join(ARTI_OUT, "eval", "gloss"))
    ap.add_argument("--tasks", default=",".join(TASK_ORDER))
    ap.add_argument("--speaker", default=None, help="restrict to spk1|spk2|spk3")
    ap.add_argument("--spk3-repaired", default="none", choices=["none", "all", "high"],
                    help="swap spk3's scrambled post session for the duration-repaired "
                         "manifest (default all = high+medium confidence)")
    ap.add_argument("--build", action="store_true", help="(re)build gloss manifests first")
    ap.add_argument("--limit", type=int, default=None,
                    help="DEBUG: cap segments per condition (own cache)")
    ap.add_argument("--cluster", action="store_true",
                    help="silhouette / k-means / drift readout (needs rep A + rep B)")
    ap.add_argument("--figs", action="store_true", help="confusion + t-SNE figures")
    ap.add_argument("--figs-tasks", default="manner,place",
                    help="which tasks get the per-speaker grid + t-SNE (all tasks get "
                         "the pooled confusion)")
    ap.add_argument("--rep", default="both", choices=["both", "A", "B"])
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--cap", type=int, default=600, help="segments per t-SNE panel")
    ap.add_argument("--cluster-cap", type=int, default=4000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--scratch", default=os.environ.get(
        "SCRATCHPAD", os.path.join(ARTI_OUT, "eval", "gloss", "manifests")))
    args = ap.parse_args()

    if args.build:
        from artijepa.gloss import build_manifest
        build_manifest(speaker=args.speaker)
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tasks_all = build_tasks()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in tasks_all]
    if bad:
        raise SystemExit(f"[glossphg] unknown --tasks {bad}; choose from {TASK_ORDER}")
    fig_tasks = [t.strip() for t in args.figs_tasks.split(",") if t.strip() in tasks]
    reps = ["A", "B"] if args.rep == "both" else [args.rep]
    need_vec = args.cluster or args.figs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    if args.limit:
        cfg["data"]["max_clips"] = args.limit
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg["meta"].get("dtype", "float16").lower(), torch.float32)
    sfx = (f"_{args.speaker}" if args.speaker else "") + \
          ("" if args.spk3_repaired == "all" else f"_spk3{args.spk3_repaired}")
    enc = args.encoder

    # -- probes (Annot-16-trained Phase-3 heads) + their anchor metrics -------
    probes, anchors = {}, {}
    for t in tasks:
        pt = os.path.join(args.probe_dir, f"phgroups_{enc}_s{args.seed}_{t}.pt")
        if not os.path.exists(pt):
            raise SystemExit(f"[glossphg] no Phase-3 probe for task={t!r}: {pt}\n"
                             f"  (run scripts/28_phoneme_groups.sbatch {enc} {args.seed})")
        clf, ck = load_probe(pt, device=device)
        probes[t] = (clf, ck, os.path.basename(pt))
        # geometry guard: a silent mismatch here produces garbage, not an error
        for k in ("spatial_size", "frames_per_clip", "target_fps"):
            if ck.get(k) is not None and ck[k] != cfg["data"].get(k):
                raise SystemExit(f"[glossphg] probe/{k}={ck[k]} but config {k}="
                                 f"{cfg['data'].get(k)} — geometry must match")
        m = ck.get("metrics", {}) or {}
        anchors[t] = {nm: {"f1_macro": (m.get(nm) or {}).get("f1_macro"),
                           "kappa": (round(kappa_from_cm((m.get(nm) or {})
                                                         ["confusion_matrix"]), 4)
                                     if (m.get(nm) or {}).get("confusion_matrix")
                                     else None)}
                      for nm in ("test", "test_lss")}
        print(f"[glossphg] {t:11s} <- {os.path.basename(pt)}   anchors: "
              f"in-domain F1={anchors[t]['test']['f1_macro']} / "
              f"healthy-OOD F1={anchors[t]['test_lss']['f1_macro']}")

    # -- gloss features per condition ---------------------------------------
    cond_cfgs, manifests = {}, {}
    for cond in CONDS:
        man, rows = cond_manifest(cond, args.speaker, args.scratch, args.spk3_repaired)
        tag = f"glossphg_{cond}{sfx}_{enc}"
        cond_cfgs[cond] = (condition_cfg(cfg, cond, man, tag), rows)
        manifests[cond] = man
    encoder = None
    if not all(_cache_ready(c) for c, _ in cond_cfgs.values()):
        print(f"[glossphg] frozen encoder <- {cfg['encoder'].get('spec')} "
              f"dtype={dtype} device={device}")
        encoder = load_frozen_encoder(cfg, device)
    else:
        print("[glossphg] all condition caches present — skipping encoder load")
    data = {}
    for cond in CONDS:
        ccfg, rows = cond_cfgs[cond]
        data[cond] = load_condition(ccfg, cond, rows, encoder, device, dtype)
    del encoder; torch.cuda.empty_cache()

    # -- predict + score ------------------------------------------------------
    pc = cfg["probe"]
    bs, workers = pc.get("batch_size", 128), pc.get("workers", 4)
    amp = device.type == "cuda"
    res = {c: {"_groups": sorted(set(data[c]["grp"].tolist())),
               "n_utt": data[c]["n_utt"], "n_segments": int(len(data[c]["phon"]))}
           for c in CONDS}
    emb = {c: {} for c in CONDS}
    emb["_res"] = res
    for t in tasks:
        clf, ck, _ = probes[t]
        members, remap, names = tasks_all[t]
        for cond in CONDS:
            d = data[cond]
            ds = _ClipDS(d["feats"], d["off"], d["phon"], members, remap,
                         preload=True, budget_gb=pc.get("preload_gb", 20.0))
            if not len(ds):
                raise SystemExit(f"[glossphg] no {t} segments in {cond}")
            y_t, y_p, vecA = predict_task(clf, ds, device, bs, workers, amp,
                                          want_vec=need_vec)
            cache_idx = np.asarray(ds.idx)
            grp = d["grp"][cache_idx]
            utt = d["row"][cache_idx]
            n_utt_of = {g: len({int(u) for u, gg in zip(utt, grp) if gg == g})
                        for g in set(grp.tolist())}
            pooled, per = _cell_metrics(y_t, y_p, grp, len(names), names, n_utt_of)
            res[cond][t] = {"pooled": pooled, "per_group": per}
            print(f"[glossphg] {t:11s} {cond:4s} F1={pooled['f1_macro']:.3f} "
                  f"κ={pooled['kappa']:.3f} acc={pooled['accuracy']:.3f} n={pooled['n']}")
            for g, m in per.items():
                print(f"[glossphg]     {g:12s} F1={m['f1_macro']:.3f} "
                      f"κ={m['kappa']:.3f} n={m['n']} ({m['n_utt']} utts)")
            if need_vec:
                # rep B re-reads the ragged cache, so only build it when asked for
                emb[cond][t] = {"y_true": y_t, "y_pred": y_p, "grp": grp, "utt": utt,
                                "repA": vecA}
                if "B" in reps:
                    emb[cond][t]["repB"] = rep_b(d["feats"], d["off"], cache_idx)
        clf.to("cpu")

    # -- clustering readout ---------------------------------------------------
    clusters, drifts = {}, {}
    if args.cluster:
        for t in tasks:
            names = tasks_all[t][2]
            clusters[t] = {}
            for rep in reps:
                key = f"rep{rep}"
                X = np.concatenate([emb[c][t][key] for c in CONDS])
                y = np.concatenate([emb[c][t]["y_true"] for c in CONDS])
                spk = np.concatenate([np.array([g.partition("/")[0]
                                                for g in emb[c][t]["grp"]])
                                      for c in CONDS])
                cnd = np.concatenate([np.full(len(emb[c][t]["y_true"]), c) for c in CONDS])
                clusters[t][key] = cluster_stats(X, y, spk, cnd, len(names),
                                                 args.seed, args.cluster_cap)
                s = clusters[t][key]
                print(f"[glossphg] cluster {t:11s} {key}  sil(phoneme)="
                      f"{s['silhouette_phoneme']} sil(speaker)={s['silhouette_speaker']} "
                      f"sil(condition)={s['silhouette_condition']}  "
                      f"kmeans ARI={s['kmeans_ari']} NMI={s['kmeans_nmi']}")
                # pre->post drift, within speaker (the cross-speaker gap dominates
                # otherwise -- RESULTS_gloss.md "Per-vowel cross-speaker t-SNE")
                for spk_id in sorted({g.partition("/")[0] for g in res["pre"]["_groups"]}):
                    P_, O_ = emb["pre"][t], emb["post"][t]
                    mp = np.array([g.partition("/")[0] == spk_id for g in P_["grp"]])
                    mo = np.array([g.partition("/")[0] == spk_id for g in O_["grp"]])
                    if mp.sum() < 50 or mo.sum() < 50:
                        continue
                    M, _ = drift_matrix(P_[key][mp], P_["y_true"][mp], P_["utt"][mp],
                                        O_[key][mo], O_["y_true"][mo], len(names))
                    B, _ = drift_baseline(P_[key][mp], P_["y_true"][mp],
                                          P_["utt"][mp], len(names))
                    sr = lambda Z: float(np.nanmean(np.where(Z.sum(1) > 0,
                                                             np.diag(_rownorm(Z)),
                                                             np.nan)))
                    d = {"drift": {"matrix": M.astype(int).tolist(),
                                   "self_retention": round(sr(M), 4)},
                         "baseline": {"matrix": B.astype(int).tolist(),
                                      "self_retention": round(sr(B), 4)},
                         "class_names": names}
                    d["delta_self_retention"] = round(
                        d["drift"]["self_retention"] - d["baseline"]["self_retention"], 4)
                    drifts.setdefault(t, {}).setdefault(key, {})[spk_id] = d
                    print(f"[glossphg]   drift {t} {key} {spk_id}: self-retention "
                          f"post→pre {d['drift']['self_retention']:.3f} vs baseline "
                          f"{d['baseline']['self_retention']:.3f} "
                          f"({d['delta_self_retention']:+.3f})")

    # -- write ----------------------------------------------------------------
    summary = {"encoder": enc, "seed": args.seed, "head": "attentive_lstm_clip",
               "task_formulation": "segment-level (one gold phoneme = one label)",
               "speaker": args.speaker or "pooled",
               "spk3_repaired": args.spk3_repaired,
               "manifests": manifests,
               "probes": {t: probes[t][2] for t in tasks},
               "anchors": anchors,
               "conditions": {c: {k: v for k, v in res[c].items() if k != "_groups"}
                              for c in CONDS},
               "groups": {c: res[c]["_groups"] for c in CONDS},
               "clusters": clusters, "drift": drifts}
    sp = os.path.join(args.out_dir, f"glossphg_{enc}{sfx}_s{args.seed}.json")
    json.dump(summary, open(sp, "w"), indent=2)
    print(f"[glossphg] wrote {sp}")

    # -- ladder table ---------------------------------------------------------
    print(f"\n===== segment-level ladder (macro-F1 | κ), probe={enc} s{args.seed} =====")
    print(f"{'stage':<34}" + "".join(f"{t:>22}" for t in tasks))
    def _row(label, get):
        print(f"{label:<34}" + "".join(f"{get(t):>22}" for t in tasks))
    _row("in-domain test (Annot-16)",
         lambda t: f"{anchors[t]['test']['f1_macro']} | {anchors[t]['test']['kappa']}")
    _row("healthy-OOD (usc_lss)",
         lambda t: f"{anchors[t]['test_lss']['f1_macro']} | {anchors[t]['test_lss']['kappa']}")
    for cond in CONDS:
        _row(f"gloss {cond} (pooled)",
             lambda t, c=cond: f"{res[c][t]['pooled']['f1_macro']:.4f} | "
                               f"{res[c][t]['pooled']['kappa']:.4f}")
    print(f"\n----- per speaker × session (macro-F1 | κ | n_seg) -----")
    rows, cols, cells = _grid(res)
    for r in rows:
        for c in cols:
            if (r, c) not in cells:
                continue
            cond, g = cells[(r, c)]
            # `g` collapses to a bare speaker for single-session cells (spk1 pre AND
            # post are both "spk1"), so qualify the printed label with the column
            # (pre|post|post1|post2); the JSON stays keyed by condition, unambiguous.
            label = g if "/" in g else f"{g}/{c}"
            cellstr = "".join(
                f"{res[cond][t]['per_group'][g]['f1_macro']:>8.3f} |"
                f"{res[cond][t]['per_group'][g]['kappa']:>7.3f} |"
                f"{res[cond][t]['per_group'][g]['n']:>6d}   " for t in tasks)
            print(f"{label:<14}{cellstr}")

    # -- figures --------------------------------------------------------------
    if args.figs:
        for t in tasks:
            names = tasks_all[t][2]
            plot_cm_pooled(res, t, names, args.out_dir, enc, args.seed, sfx)
            if t in fig_tasks:
                plot_cm_per_speaker(res, t, names, args.out_dir, enc, args.seed, sfx)
                for rep in reps:
                    plot_tsne_grid(emb, t, names, args.out_dir, enc, args.seed, sfx,
                                   f"rep{rep}", args.method, args.cap,
                                   args.perplexity, args.pca)
        for t, byrep in drifts.items():
            for key, byspk in byrep.items():
                for spk_id, d in byspk.items():
                    plot_drift(d, t, tasks_all[t][2], spk_id, args.out_dir, enc,
                               args.seed, sfx, key)
    print(f"[glossphg] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
