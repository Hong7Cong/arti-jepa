"""t-SNE / UMAP of frozen rtMRI features colored by phoneme class, cross-model.

Visualizes whether per-temporal-token features separate by phoneme, for one or
more frozen encoders, in **four views** (the requested plot set):

  * **phoneme**   -- ALL phonemes, colored by manner of articulation
                     (Vowel/Diphthong/Plosive/Fricative/Affricate/Nasal/Approximant).
  * **vowel**     -- vowels + diphthongs ONLY, colored by vowel identity (per-phoneme).
  * **consonant** -- consonants ONLY, colored by manner.
  * **place**     -- consonants ONLY, colored by PLACE of articulation (8 classes:
                     Bilabial/Labiodental/Dental/Alveolar/Postalveolar/Palatal/
                     Velar/Glottal).

Two token representations (do BOTH by default):

  * **B (raw-pooled)** -- the frozen encoder's OWN per-token feature, mean-pooled
    over the S' spatial tokens: ``feats[n,t].mean(S')`` -> ``[D]``. NO trained
    probe. This shows the encoder's *intrinsic* phoneme geometry, so a tighter-
    clustered plot for T-SSL vs a baseline is a NON-circular corroboration of the
    kappa ranking. This is the scientifically load-bearing panel.
  * **A (trained-q)** -- the AttentivePooler output ``q`` (the penultimate vector
    the linear phoneme classifier reads), reconstructed from the saved probe .pt.
    Clusters look clean partly BECAUSE the pooler was trained on phonemes (mildly
    circular); it depicts the probe's decision space, not the raw encoder.

Both drop ``sil`` (silence) and padded (IGNORE_INDEX) tokens, subsample per phoneme
for balance, then embed with t-SNE (and UMAP if available). Each encoder's token
pool is extracted ONCE (the big spatial cache is read a single time); the four
views are t-SNE'd from that shared pool by masking, so re-running views is cheap.

The frozen feature cache + the trained pooler are located via the probe checkpoint
(``eval/phoneme_usc_lss_<enc>sp_*_attentive_ce_s<seed>.pt``); its ``feature_tag``
field names the cache dir, so no stale hash guessing.

Runs CPU-only. Needs numpy + scikit-learn + matplotlib; UMAP optional. Set
``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`` on a shared/login node.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    # cross-model, all 4 views, raw-pooled rep, t-SNE, cross-domain (usc_lss) split:
    python -m artijepa.tsne_phonemes \
        --encoder tssl256comb100,videomae_rtmri,pretrained,videomae,dinov2,vitl \
        --rep B --method tsne --split test_lss
    # single encoder, both reps/methods, all views:
    python -m artijepa.tsne_phonemes --encoder tssl256comb100
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import artijepa.phonemes as P

# --------------------------------------------------------------------------- #
# phonetic classes (manner of articulation) -- the "manner" coloring
# --------------------------------------------------------------------------- #
PHON_CLASS = {
    # vowels (monophthongs, incl. rhotic er)
    "aa": "Vowel", "ae": "Vowel", "ah": "Vowel", "ao": "Vowel", "eh": "Vowel",
    "er": "Vowel", "ih": "Vowel", "iy": "Vowel", "uh": "Vowel", "uw": "Vowel",
    # diphthongs
    "aw": "Diphthong", "ay": "Diphthong", "ey": "Diphthong", "ow": "Diphthong",
    "oy": "Diphthong",
    # plosives / stops
    "b": "Plosive", "d": "Plosive", "g": "Plosive", "k": "Plosive", "p": "Plosive",
    "t": "Plosive",
    # fricatives (incl. glottal h/hh)
    "dh": "Fricative", "f": "Fricative", "h": "Fricative", "hh": "Fricative",
    "s": "Fricative", "sh": "Fricative", "th": "Fricative", "v": "Fricative",
    "z": "Fricative", "zh": "Fricative",
    # affricates
    "ch": "Affricate", "jh": "Affricate",
    # nasals
    "m": "Nasal", "n": "Nasal", "ng": "Nasal",
    # approximants / liquids / glides
    "l": "Approximant", "r": "Approximant", "w": "Approximant", "y": "Approximant",
    # silence (dropped before plotting; here for completeness)
    "sil": "Silence",
}
CLASS_ORDER = ["Vowel", "Diphthong", "Plosive", "Fricative", "Affricate",
               "Nasal", "Approximant"]
CLASS_COLOR = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(CLASS_ORDER)}

CONS_MANNER_ORDER = ["Plosive", "Fricative", "Affricate", "Nasal", "Approximant"]

# --------------------------------------------------------------------------- #
# place of articulation (consonants only) -- the "place" / consonant-group coloring
# --------------------------------------------------------------------------- #
PLACE_CLASS = {
    "b": "Bilabial", "p": "Bilabial", "m": "Bilabial",
    "f": "Labiodental", "v": "Labiodental",
    "th": "Dental", "dh": "Dental",
    "t": "Alveolar", "d": "Alveolar", "s": "Alveolar", "z": "Alveolar",
    "n": "Alveolar", "l": "Alveolar", "r": "Alveolar",
    "ch": "Postalveolar", "jh": "Postalveolar",                 # affricates
    "sh": "Postalveolar", "zh": "Postalveolar",                 # fricatives
    "y": "Palatal",                                             # true palatal glide
    "k": "Velar", "g": "Velar", "ng": "Velar", "w": "Velar",     # w = labio-velar
    "h": "Glottal", "hh": "Glottal",
}
PLACE_ORDER = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Postalveolar",
               "Palatal", "Velar", "Glottal"]
PLACE_COLOR = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(PLACE_ORDER)}

# broad category membership (phoneme strings, canonical ARPABET order)
VOWEL_CLASSES = {"Vowel", "Diphthong"}
CONS_CLASSES = {"Plosive", "Fricative", "Affricate", "Nasal", "Approximant"}
VOWEL_PHON = [p for p in P.ARPABET if PHON_CLASS.get(p) in VOWEL_CLASSES]
CONS_PHON = [p for p in P.ARPABET if PHON_CLASS.get(p) in CONS_CLASSES]
VOWEL_COLOR = {p: plt.get_cmap("tab20")(i) for i, p in enumerate(VOWEL_PHON)}

# --------------------------------------------------------------------------- #
# the 4 views: (focus subset, coloring scheme, title)
# --------------------------------------------------------------------------- #
VIEWS = {
    "phoneme":   dict(focus=None,        color="manner", title="all phonemes · manner"),
    "vowel":     dict(focus="vowel",     color="vowel",  title="vowels · identity"),
    "consonant": dict(focus="consonant", color="manner", title="consonants · manner"),
    "place":     dict(focus="consonant", color="place",  title="consonants · place"),
}
VIEW_ORDER = ["phoneme", "vowel", "consonant", "place"]
FOCUS_IDX = {
    "vowel": {P.PHON2IDX[p] for p in VOWEL_PHON},
    "consonant": {P.PHON2IDX[p] for p in CONS_PHON},
}


def color_scheme(color):
    """(class_fn: phon_str -> label|None, ordered labels, label -> RGBA)."""
    if color == "manner":
        return (lambda ph: PHON_CLASS.get(ph), CLASS_ORDER, CLASS_COLOR)
    if color == "place":
        return (lambda ph: PLACE_CLASS.get(ph), PLACE_ORDER, PLACE_COLOR)
    if color == "vowel":
        return (lambda ph: ph if ph in VOWEL_COLOR else None, VOWEL_PHON, VOWEL_COLOR)
    raise ValueError(color)


DEFAULT_ARTI_OUT = os.environ.get("ARTI_OUT", "/scratch1/hongn/artijepa")


# --------------------------------------------------------------------------- #
# locate probe checkpoint + feature cache for an encoder key
# --------------------------------------------------------------------------- #
def _probe_feature_tag(pt):
    """Derive the feature-cache tag from a probe filename (no torch.load needed):
    phoneme_usc_lss_<TAG>_attentive_ce_s<seed>.pt  ->  <TAG>."""
    base = os.path.basename(pt)
    return base[len("phoneme_usc_lss_"):].split("_attentive_ce_s")[0]


def find_probe(encoder, seed, eval_dir, feat_root=None, split=None):
    """Glob the saved attentive-probe .pt for an encoder key (e.g. 'tssl256comb100',
    'videomae_rtmri', 'pretrained'). The 'sp' suffix marks the un-pooled cache.

    An encoder key can match SEVERAL probes (re-runs leave orphaned tags whose feature
    cache has since been pruned). When ``feat_root``/``split`` are given, prefer a probe
    whose ``<split>`` feature cache still exists on disk -- otherwise the alphabetically
    first hit can point at a deleted cache and abort the run (e.g. videomae_rtmri's stale
    b9178b593e tag vs the live eab1770163). Falls back to the first hit so the caller's
    'missing cache' error still fires when NONE are live."""
    pat = os.path.join(eval_dir, f"phoneme_usc_lss_{encoder}sp_*_attentive_ce_s{seed}.pt")
    hits = sorted(glob.glob(pat))
    if hits and feat_root and split:
        live = [h for h in hits
                if os.path.exists(os.path.join(feat_root, _probe_feature_tag(h),
                                               f"{split}.feats.npy"))]
        if live:
            if live[0] != hits[0]:
                print(f"[tsne] {encoder}: skipping probe(s) with a pruned {split} cache "
                      f"-> using {os.path.basename(live[0])}")
            return live[0]
    if not hits:
        raise SystemExit(
            f"[tsne] no attentive probe for encoder={encoder!r} seed={seed} at\n  {pat}\n"
            f"       (available: "
            + ", ".join(sorted({os.path.basename(p).split('_attentive')[0]
                                .replace('phoneme_usc_lss_', '').rsplit('sp_', 1)[0]
                                for p in glob.glob(os.path.join(eval_dir,
                                    'phoneme_usc_lss_*sp_*_attentive_ce_s*.pt'))})) + ")")
    return hits[0]


def load_split_cache(feature_tag, feat_root, split="test"):
    """mmap the un-pooled features + labels for a feature_tag cache dir + split."""
    cdir = os.path.join(feat_root, feature_tag)
    fp = os.path.join(cdir, f"{split}.feats.npy")
    lp = os.path.join(cdir, f"{split}.labels.npy")
    if not (os.path.exists(fp) and os.path.exists(lp)):
        raise SystemExit(f"[tsne] missing {split} cache for tag {feature_tag!r} at {cdir}")
    feats = np.load(fp, mmap_mode="r")            # [N,T',S',D] f16 (un-pooled)
    labels = np.load(lp)                          # [N,T'] i64
    if feats.ndim != 4:
        raise SystemExit(f"[tsne] expected un-pooled [N,T',S',D] cache, got {feats.shape}; "
                         "this encoder's cache is spatially pooled (not an attentive cache)")
    return feats, labels


# backward-compat alias (confmat_phonemes imports load_test_cache)
def load_test_cache(feature_tag, feat_root):
    return load_split_cache(feature_tag, feat_root, "test")


# --------------------------------------------------------------------------- #
# build the two token representations on a shared subsampled token set
# --------------------------------------------------------------------------- #
def build_reps(feats, labels, reps, pooler, seed, cap):
    """Return (rep -> X[M,D] float32, phon_ids[M]) over ONE balanced token set.

    Tokens are the valid (non-pad, non-sil) temporal tokens, subsampled to <=cap
    per phoneme so no phoneme (or class) dominates. Both reps are built on the
    identical token subset. The four views later mask THIS pool -- so the (large)
    feature cache is read only once per encoder."""
    N, Tp, S, D = feats.shape
    flat = labels.reshape(-1)
    valid = (flat != P.IGNORE_INDEX) & (flat != P.SIL_IDX)
    idx = np.where(valid)[0]
    rng = np.random.default_rng(seed)
    sel = []
    for p in np.unique(flat[idx]):
        ip = idx[flat[idx] == p]
        if len(ip) > cap:
            ip = rng.choice(ip, cap, replace=False)
        sel.append(ip)
    sel = np.sort(np.concatenate(sel))
    phon_ids = flat[sel]
    clips, toks = sel // Tp, sel % Tp
    out = {r: np.empty((len(sel), D), dtype=np.float32) for r in reps}
    with torch.no_grad():
        for n in np.unique(clips):
            m = clips == n
            tset = toks[m]
            fn = np.asarray(feats[n], dtype=np.float32)          # [T',S',D]
            if "B" in out:
                out["B"][m] = fn[tset].mean(1)                   # [k,D] mean over S'
            if "A" in out:
                q = pooler(torch.from_numpy(fn[tset])).squeeze(1)  # [k,S,D]->[k,1,D]->[k,D]
                out["A"][m] = q.numpy()
    print(f"[tsne]   token pool: {len(sel)} kept over {len(np.unique(phon_ids))} phonemes "
          f"(<= {cap}/phoneme); D={D}")
    return out, phon_ids


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def pca_reduce(X, k=50):
    from sklearn.decomposition import PCA
    k = min(k, X.shape[1], X.shape[0])
    return PCA(n_components=k, random_state=0).fit_transform(X.astype(np.float32))


def embed(X, method, seed, perplexity):
    """2-D embedding. X should already be PCA-reduced (standard t-SNE preprocessing)."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(perplexity, max(5, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=seed).fit_transform(X)
    if method == "umap":
        import umap                                             # optional
        return umap.UMAP(n_components=2, random_state=seed,
                         n_neighbors=15, min_dist=0.1).fit_transform(X)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
REP_LABEL = {"A": "A: trained-q (probe space)", "B": "B: raw-pooled (encoder geometry)"}


def scatter(ax, xy, phon_ids, title, color):
    class_fn, order, cmap = color_scheme(color)
    lbl = np.array([class_fn(P.IDX2PHON[int(p)]) for p in phon_ids], dtype=object)
    for c in order:
        m = lbl == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, linewidths=0,
                       color=cmap[c], label=c)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def legend_for(fig, color, ncol=None):
    _, order, cmap = color_scheme(color)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[c], label=c)
               for c in order]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(order),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True,
                    help="encoder key(s), comma-separated: e.g. tssl256comb100,"
                         "videomae_rtmri,pretrained,videomae,dinov2,vitl")
    ap.add_argument("--seed", type=int, default=0, help="which probe seed's cache/probe")
    ap.add_argument("--split", default="test_lss", choices=["test", "test_lss"],
                    help="test = in-domain (Annot-16 sub043); test_lss = cross-domain "
                         "usc_lss (default, the headline)")
    ap.add_argument("--view", default="all",
                    help="comma list of views: phoneme,vowel,consonant,place (or 'all')")
    ap.add_argument("--rep", default="B", choices=["both", "A", "B"],
                    help="A=trained-q, B=raw-pooled (default; non-circular)")
    ap.add_argument("--method", default="tsne", choices=["both", "tsne", "umap"])
    ap.add_argument("--cap", type=int, default=200, help="max tokens per phoneme (balance)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca", type=int, default=50, help="PCA dims before t-SNE/UMAP (0=off)")
    ap.add_argument("--arti-out", default=DEFAULT_ARTI_OUT)
    ap.add_argument("--out", default=None, help="output dir (default <arti-out>/eval/tsne)")
    args = ap.parse_args()

    eval_dir = os.path.join(args.arti_out, "eval")
    feat_root = os.path.join(args.arti_out, "feat_cache", "phoneme")
    out_dir = args.out or os.path.join(eval_dir, "tsne")
    os.makedirs(out_dir, exist_ok=True)
    encoders = [e.strip() for e in args.encoder.split(",") if e.strip()]
    views = VIEW_ORDER if args.view == "all" else [v.strip() for v in args.view.split(",")
                                                   if v.strip()]
    reps = ["A", "B"] if args.rep == "both" else [args.rep]
    methods = ["tsne", "umap"] if args.method == "both" else [args.method]

    if "umap" in methods:
        try:
            import umap  # noqa: F401
        except Exception as e:
            print(f"[tsne] UMAP unavailable ({type(e).__name__}: {e}); skipping UMAP, "
                  "keeping t-SNE. `pip install umap-learn` to enable.")
            methods = [m for m in methods if m != "umap"] or ["tsne"]

    from artijepa.eval_phoneme import load_probe
    # (enc, view, rep, method) -> (xy, phon_ids)
    results, kappas = {}, {}
    for enc in encoders:
        print(f"[tsne] === {enc} (seed {args.seed}, split {args.split}) ===")
        pt = find_probe(enc, args.seed, eval_dir, feat_root, args.split)
        probe, ck = load_probe(pt, device="cpu")
        pooler = probe.pooler if "A" in reps else None
        try:
            kappas[enc] = ck["metrics"]["tests"]["test_lss"]["kappa"] \
                if args.split == "test_lss" else ck["metrics"]["test"]["kappa"]
        except Exception:
            kappas[enc] = None
        print(f"[tsne]   probe={os.path.basename(pt)} tag={ck['feature_tag']} "
              f"{args.split}_kappa={kappas[enc]}")
        feats, labels = load_split_cache(ck["feature_tag"], feat_root, args.split)
        X, phon_ids = build_reps(feats, labels, reps, pooler, args.seed, args.cap)
        for view in views:
            foc = VIEWS[view]["focus"]
            mask = (np.ones(len(phon_ids), bool) if foc is None
                    else np.isin(phon_ids, list(FOCUS_IDX[foc])))
            pid_v = phon_ids[mask]
            for r in reps:
                Xr = X[r][mask]
                Xr = pca_reduce(Xr, args.pca) if args.pca and args.pca > 0 else Xr
                for method in methods:
                    print(f"[tsne]   view {view} / rep {r} / {method}: {len(pid_v)} tokens")
                    xy = embed(Xr, method, args.seed, args.perplexity)
                    results[(enc, view, r, method)] = (xy, pid_v)

    ktag = "kappa_xdom" if args.split == "test_lss" else "kappa"

    # -- cross-encoder comparison: one figure per (view, rep, method); cols=encoders
    # This is the headline "across models" deliverable.
    for view in views:
        color = VIEWS[view]["color"]
        for r in reps:
            for method in methods:
                nc = len(encoders)
                fig, axes = plt.subplots(1, nc, figsize=(4.2 * nc, 4.6), squeeze=False)
                for j, enc in enumerate(encoders):
                    xy, pid = results[(enc, view, r, method)]
                    kap = f"\n{ktag}={kappas[enc]:.3f}" if kappas[enc] is not None else ""
                    scatter(axes[0][j], xy, pid, f"{enc}{kap}", color)
                fig.suptitle(f"{VIEWS[view]['title']}  –  {REP_LABEL[r]}  –  {method} "
                             f"–  {args.split}", fontsize=11)
                legend_for(fig, color, ncol=min(len(color_scheme(color)[1]), 8))
                fig.tight_layout(rect=[0, 0.07, 1, 0.93])
                fp = os.path.join(out_dir,
                                  f"compare_{view}_rep{r}_{method}_{args.split}_s{args.seed}.png")
                fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
                print(f"[tsne] wrote {fp}")

    # -- per-encoder overview: the requested views in a row (rep B / first method)
    r0 = "B" if "B" in reps else reps[0]
    m0 = methods[0]
    for enc in encoders:
        nc = len(views)
        fig, axes = plt.subplots(1, nc, figsize=(4.4 * nc, 4.8), squeeze=False)
        for j, view in enumerate(views):
            xy, pid = results[(enc, view, r0, m0)]
            scatter(axes[0][j], xy, pid, VIEWS[view]["title"], VIEWS[view]["color"])
            _, order, cmap = color_scheme(VIEWS[view]["color"])
            handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[c], label=c)
                       for c in order]
            axes[0][j].legend(handles=handles, fontsize=5, loc="upper right",
                              framealpha=0.6, ncol=1, handletextpad=0.2)
        kap = f"  {ktag}={kappas[enc]:.3f}" if kappas[enc] is not None else ""
        fig.suptitle(f"{enc}{kap}  –  {REP_LABEL[r0]}  –  {m0}  –  {args.split} "
                     "(sil dropped)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = os.path.join(out_dir, f"tsne_{enc}_{args.split}_s{args.seed}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[tsne] wrote {fp}")

    print(f"[tsne] done -> {out_dir}")


if __name__ == "__main__":
    main()
