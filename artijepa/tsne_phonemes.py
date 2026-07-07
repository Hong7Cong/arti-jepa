"""t-SNE / UMAP of frozen rtMRI features colored by phoneme class (usc_lss).

Visualizes whether per-temporal-token features separate by phoneme, for one or
more frozen encoders. Two representations (do BOTH by default):

  * **B (raw-pooled)** -- the frozen encoder's OWN per-token feature, mean-pooled
    over the S' spatial tokens: ``feats[n,t].mean(S')`` -> ``[D]``. NO trained
    probe. This shows the encoder's *intrinsic* phoneme geometry, so a tighter-
    clustered plot for T-SSL vs a baseline is a NON-circular corroboration of the
    kappa ranking. This is the scientifically load-bearing panel.
  * **A (trained-q)** -- the AttentivePooler output ``q`` (the penultimate vector
    the linear phoneme classifier reads), reconstructed from the saved probe .pt.
    Clusters look clean partly BECAUSE the pooler was trained on phonemes (mildly
    circular); it depicts the probe's decision space, not the raw encoder.

Both drop ``sil`` (silence -- the majority class; it forms one huge blob and is
already excluded from the eval kappa) and padded (IGNORE_INDEX) tokens, subsample
per phoneme for balance, then embed with t-SNE and UMAP. Points are colored by
broad phonetic class (manner of articulation) -- far more readable than 41 hues.

The frozen feature cache + the trained pooler are located via the probe checkpoint
(``eval/phoneme_usc_lss_<enc>sp_*_attentive_ce_s<seed>.pt``); its ``feature_tag``
field names the cache dir, so no stale hash guessing.

Runs CPU-only (the pooler is tiny; t-SNE/UMAP are CPU). Needs numpy + scikit-learn
+ matplotlib; UMAP is optional (skipped with a note if not importable).

Usage:
    source dev_artiJEPA/scripts/_env.sh
    # single encoder, both reps, both methods -> one figure:
    python -m artijepa.tsne_phonemes --encoder tssl256comb100
    # cross-encoder comparison panel (one figure per rep x method, cols = encoders):
    python -m artijepa.tsne_phonemes --encoder tssl256comb100,pretrained256,base_resnet
    # loop form (each call = one PNG you assemble into a panel):
    for e in tssl256comb100 pretrained256 videomae base_vitl; do
        python -m artijepa.tsne_phonemes --encoder $e; done
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
# phonetic classes (manner of articulation) -- the coloring
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

DEFAULT_ARTI_OUT = os.environ.get("ARTI_OUT", "/scratch1/hongn/artijepa")


# --------------------------------------------------------------------------- #
# locate probe checkpoint + feature cache for an encoder key
# --------------------------------------------------------------------------- #
def find_probe(encoder, seed, eval_dir):
    """Glob the saved attentive-probe .pt for an encoder key (e.g. 'tssl256comb100',
    'base_vitl', 'videomae'). The 'sp' suffix marks the un-pooled (spatial) cache."""
    pat = os.path.join(eval_dir, f"phoneme_usc_lss_{encoder}sp_*_attentive_ce_s{seed}.pt")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(
            f"[tsne] no attentive probe for encoder={encoder!r} seed={seed} at\n  {pat}\n"
            f"       (available: "
            + ", ".join(sorted({os.path.basename(p).split('_attentive')[0]
                                .replace('phoneme_usc_lss_', '').rsplit('sp_', 1)[0]
                                for p in glob.glob(os.path.join(eval_dir,
                                    'phoneme_usc_lss_*sp_*_attentive_ce_s*.pt'))})) + ")")
    return hits[0]


def load_test_cache(feature_tag, feat_root):
    """mmap the un-pooled test features + labels for a feature_tag cache dir."""
    cdir = os.path.join(feat_root, feature_tag)
    fp = os.path.join(cdir, "test.feats.npy")
    lp = os.path.join(cdir, "test.labels.npy")
    if not (os.path.exists(fp) and os.path.exists(lp)):
        raise SystemExit(f"[tsne] missing test cache for tag {feature_tag!r} at {cdir}")
    feats = np.load(fp, mmap_mode="r")            # [N,T',S',D] f16 (un-pooled)
    labels = np.load(lp)                          # [N,T'] i64
    if feats.ndim != 4:
        raise SystemExit(f"[tsne] expected un-pooled [N,T',S',D] cache, got {feats.shape}; "
                         "this encoder's cache is spatially pooled (not an attentive cache)")
    return feats, labels


# --------------------------------------------------------------------------- #
# build the two token representations on a shared subsampled token set
# --------------------------------------------------------------------------- #
def build_reps(feats, labels, reps, pooler, seed, cap):
    """Return (rep -> X[M,D] float32, phon_ids[M]) over the SAME balanced token set.

    Tokens are the valid (non-pad, non-sil) temporal tokens of the test split,
    subsampled to <=cap per phoneme so no phoneme (or class) dominates the plot.
    Both reps are built on the identical token subset so A vs B are comparable.
    """
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
    print(f"[tsne]   tokens: {len(sel)} kept over {len(np.unique(phon_ids))} phonemes "
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


def scatter(ax, xy, phon_ids, title):
    classes = np.array([PHON_CLASS[P.IDX2PHON[int(p)]] for p in phon_ids])
    for c in CLASS_ORDER:
        m = classes == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, linewidths=0,
                       color=CLASS_COLOR[c], label=c)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def legend_fig(fig):
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=CLASS_COLOR[c], label=c)
               for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASS_ORDER),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True,
                    help="encoder key(s), comma-separated: e.g. tssl256comb100,"
                         "pretrained256,videomae,base_vitl,base_dinov2,base_siglip,"
                         "base_clip,base_resnet,tssl256")
    ap.add_argument("--seed", type=int, default=0, help="which probe seed's cache/probe")
    ap.add_argument("--rep", default="both", choices=["both", "A", "B"],
                    help="A=trained-q, B=raw-pooled, both (default)")
    ap.add_argument("--method", default="both", choices=["both", "tsne", "umap"])
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
    reps = ["A", "B"] if args.rep == "both" else [args.rep]
    methods = ["tsne", "umap"] if args.method == "both" else [args.method]

    # drop UMAP up front if not importable, with a clear note (keeps t-SNE working)
    if "umap" in methods:
        try:
            import umap  # noqa: F401
        except Exception as e:
            print(f"[tsne] UMAP unavailable ({type(e).__name__}: {e}); skipping UMAP, "
                  "keeping t-SNE. `pip install umap-learn` to enable.")
            methods = [m for m in methods if m != "umap"] or ["tsne"]

    from artijepa.eval_phoneme import load_probe
    results = {}                     # (encoder, rep, method) -> (xy, phon_ids)
    kappas = {}
    for enc in encoders:
        print(f"[tsne] === {enc} (seed {args.seed}) ===")
        pt = find_probe(enc, args.seed, eval_dir)
        probe, ck = load_probe(pt, device="cpu")
        pooler = probe.pooler if "A" in reps else None
        try:
            kappas[enc] = ck["metrics"]["test"]["kappa"]
        except Exception:
            kappas[enc] = None
        print(f"[tsne]   probe={os.path.basename(pt)} tag={ck['feature_tag']} "
              f"test_kappa={kappas[enc]}")
        feats, labels = load_test_cache(ck["feature_tag"], feat_root)
        X, phon_ids = build_reps(feats, labels, reps, pooler, args.seed, args.cap)
        for r in reps:
            Xr = pca_reduce(X[r], args.pca) if args.pca and args.pca > 0 else X[r]
            for method in methods:
                print(f"[tsne]   embed rep {r} / {method} ...")
                xy = embed(Xr, method, args.seed, args.perplexity)
                results[(enc, r, method)] = (xy, phon_ids)

    # -- per-encoder figure: grid [reps x methods]
    for enc in encoders:
        nr, nc = len(reps), len(methods)
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), squeeze=False)
        kap = f"  (test kappa={kappas[enc]:.3f})" if kappas[enc] is not None else ""
        for i, r in enumerate(reps):
            for j, method in enumerate(methods):
                xy, pid = results[(enc, r, method)]
                scatter(axes[i][j], xy, pid, f"{REP_LABEL[r]} - {method}")
        fig.suptitle(f"{enc}{kap} - phoneme structure (usc_lss test, sil dropped)",
                     fontsize=11)
        legend_fig(fig)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fp = os.path.join(out_dir, f"tsne_{enc}_s{args.seed}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[tsne] wrote {fp}")

    # -- cross-encoder comparison: one figure per (rep, method), cols = encoders
    if len(encoders) > 1:
        for r in reps:
            for method in methods:
                nc = len(encoders)
                fig, axes = plt.subplots(1, nc, figsize=(4.2 * nc, 4.4), squeeze=False)
                for j, enc in enumerate(encoders):
                    xy, pid = results[(enc, r, method)]
                    kap = f"\nkappa={kappas[enc]:.3f}" if kappas[enc] is not None else ""
                    scatter(axes[0][j], xy, pid, f"{enc}{kap}")
                fig.suptitle(f"{REP_LABEL[r]} - {method} - cross-encoder "
                             "(usc_lss test, sil dropped)", fontsize=11)
                legend_fig(fig)
                fig.tight_layout(rect=[0, 0.06, 1, 0.94])
                fp = os.path.join(out_dir, f"compare_rep{r}_{method}_s{args.seed}.png")
                fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
                print(f"[tsne] wrote {fp}")

    print(f"[tsne] done -> {out_dir}")


if __name__ == "__main__":
    main()
