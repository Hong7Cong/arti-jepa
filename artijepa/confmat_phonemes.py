"""Phoneme confusion matrices for frozen encoders (usc_lss test split).

Runs the trained **attentive** probe (pooler + linear head) end-to-end on the
cached un-pooled test features to get per-temporal-token phoneme predictions, then
tabulates where each encoder's classifier confuses one phoneme (or phonetic class)
for another. This is the *actual classifier confusion*, not a representation
projection -- the diagonal is per-class recall, off-diagonal is the substitution
structure behind the headline kappa.

Four matrix granularities per encoder (rows = TRUE, cols = PREDICTED,
row-normalized to recall %):

  * **groups**     -- 7x7 manner-of-articulation classes (Vowel/Diphthong/Plosive/
                      Fricative/Affricate/Nasal/Approximant) + a folded `sil` column.
                      The coarse "does it even get the manner right" view.
  * **vowels**     -- the 15 vocalic phonemes (Vowel + Diphthong). Off-set predictions
                      collapse to `->cons` (any consonant) / `->sil`.
  * **consonants** -- the 25 consonant phonemes (Plosive/Fricative/Affricate/Nasal/
                      Approximant). Off-set predictions collapse to `->vow` / `->sil`.
  * **place**      -- 8x8 consonant PLACE of articulation (Bilabial/Labiodental/Dental/
                      Alveolar/Postalveolar/Palatal/Velar/Glottal) + folded `->vow`/`->sil`.
                      The "does it get the constriction location right" view.

Reuses tsne_phonemes' probe/cache locators and the same phonetic-class + place maps,
so it sees exactly the encoders you t-SNE'd. CPU-only. Needs numpy + matplotlib.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m artijepa.confmat_phonemes \
          --encoder tssl256comb100,videomae_rtmri,pretrained,videomae,dinov2,vitl \
          --split test_lss
    # -> eval/confmat/confmat_{groups,vowels,consonants,place}_compare_<split>_s0.png
    #    + per-encoder confmat_<enc>_<split>_s0.png ; macro-recall table to stdout
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import artijepa.phonemes as P
from artijepa.tsne_phonemes import (
    PHON_CLASS, CLASS_ORDER, PLACE_CLASS, PLACE_ORDER, VOWEL_PHON, CONS_PHON,
    DEFAULT_ARTI_OUT, find_probe, load_split_cache)


# --------------------------------------------------------------------------- #
# confusion-matrix construction (raw counts; normalization happens at plot time)
# --------------------------------------------------------------------------- #
def _phoneme_confmat(true_ids, pred_ids, row_phon, extra_cols, fold):
    """rows = row_phon (true, restricted); cols = row_phon + extra_cols (predicted).

    `fold(pred_phoneme_str)` maps a predicted phoneme to a column label -- itself if
    it is one of row_phon, else one of the collapse buckets in extra_cols."""
    ridx = {p: i for i, p in enumerate(row_phon)}
    cols = list(row_phon) + list(extra_cols)
    cidx = {c: j for j, c in enumerate(cols)}
    M = np.zeros((len(row_phon), len(cols)), dtype=np.float64)
    for t, p in zip(true_ids, pred_ids):
        ts = P.IDX2PHON[int(t)]
        if ts not in ridx:
            continue
        M[ridx[ts], cidx[fold(P.IDX2PHON[int(p)])]] += 1.0
    return M, list(row_phon), cols


def group_confmat(true_ids, pred_ids):
    """7 manner classes (true, rows) x 7 classes + folded `sil` (predicted, cols)."""
    rows = list(CLASS_ORDER)
    cols = rows + ["sil"]
    ridx = {c: i for i, c in enumerate(rows)}
    cidx = {c: j for j, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=np.float64)
    for t, p in zip(true_ids, pred_ids):
        tc = PHON_CLASS.get(P.IDX2PHON[int(t)])
        if tc not in ridx:                    # true == Silence -> not a row
            continue
        pc = PHON_CLASS.get(P.IDX2PHON[int(p)])
        M[ridx[tc], cidx[pc if pc in ridx else "sil"]] += 1.0
    return M, rows, cols


def place_confmat(true_ids, pred_ids):
    """8 consonant PLACE classes (true, rows) x 8 places + folded `->vow`/`->sil`
    (predicted, cols). True vowels/silence are not consonant places -> not rows."""
    rows = list(PLACE_ORDER)
    cols = rows + ["->vow", "->sil"]
    ridx = {c: i for i, c in enumerate(rows)}
    cidx = {c: j for j, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=np.float64)
    for t, p in zip(true_ids, pred_ids):
        tc = PLACE_CLASS.get(P.IDX2PHON[int(t)])
        if tc is None:                        # true is a vowel / silence -> not a row
            continue
        ps = P.IDX2PHON[int(p)]
        pc = PLACE_CLASS.get(ps)
        col = pc if pc is not None else ("->sil" if ps == "sil" else "->vow")
        M[ridx[tc], cidx[col]] += 1.0
    return M, rows, cols


def build_matrices(true_ids, pred_ids):
    """-> {mtype: (M counts, row_labels, col_labels)} for the 4 granularities."""
    vset, cset = set(VOWEL_PHON), set(CONS_PHON)

    def fold_vowel(ps):
        return ps if ps in vset else ("->sil" if ps == "sil" else "->cons")

    def fold_cons(ps):
        return ps if ps in cset else ("->sil" if ps == "sil" else "->vow")

    return {
        "groups": group_confmat(true_ids, pred_ids),
        "vowels": _phoneme_confmat(true_ids, pred_ids, VOWEL_PHON,
                                   ["->cons", "->sil"], fold_vowel),
        "consonants": _phoneme_confmat(true_ids, pred_ids, CONS_PHON,
                                       ["->vow", "->sil"], fold_cons),
        "place": place_confmat(true_ids, pred_ids),
    }


def macro_recall(M):
    """Mean over rows of diagonal / row-support (leading square block); ignores
    empty rows. The class-balanced 'how often is TRUE t predicted as t'."""
    n = M.shape[0]
    diag = np.diag(M[:, :n])
    supp = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(supp > 0, diag / np.where(supp == 0, 1, supp), np.nan)
    return float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan")


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_confmat(ax, M, rows, cols, title, fontscale=1.0):
    supp = M.sum(1, keepdims=True)
    Mn = M / np.where(supp == 0, 1, supp)                 # row-normalized -> recall
    im = ax.imshow(Mn, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    fs = max(3.5, 7 * fontscale)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=fs)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=fs)
    ax.set_xlabel("predicted", fontsize=fs + 1)
    ax.set_ylabel("true", fontsize=fs + 1)
    # annotate cells (recall %) when the matrix is small enough to stay legible
    if len(rows) * len(cols) <= 20 * 22:
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = Mn[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{int(round(v * 100))}", ha="center",
                            va="center", fontsize=max(3, fs - 2.5),
                            color="white" if v > 0.5 else "0.15")
    ax.set_title(title, fontsize=fs + 2)
    return im


PANEL = {   # (per-panel width, height, fontscale) per matrix type
    "groups": (3.6, 3.4, 1.0),
    "vowels": (5.0, 4.6, 0.9),
    "consonants": (7.6, 7.0, 0.72),
    "place": (4.2, 4.0, 0.95),
}
MTYPES = ["groups", "vowels", "consonants", "place"]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True,
                    help="encoder key(s), comma-separated (same keys as tsne_phonemes): "
                         "tssl256comb100,pretrained256,videomae,base_resnet,base_vitl,...")
    ap.add_argument("--seed", type=int, default=0, help="which probe seed to load")
    ap.add_argument("--split", default="test_lss", choices=["test", "test_lss"],
                    help="test = in-domain (Annot-16 sub043); test_lss = cross-domain "
                         "usc_lss (default, the headline)")
    ap.add_argument("--which", default="all",
                    help="comma list: groups,vowels,consonants,place (or all)")
    ap.add_argument("--arti-out", default=DEFAULT_ARTI_OUT)
    ap.add_argument("--out", default=None, help="output dir (default <arti-out>/eval/confmat)")
    args = ap.parse_args()

    eval_dir = os.path.join(args.arti_out, "eval")
    feat_root = os.path.join(args.arti_out, "feat_cache", "phoneme")
    out_dir = args.out or os.path.join(eval_dir, "confmat")
    os.makedirs(out_dir, exist_ok=True)
    encoders = [e.strip() for e in args.encoder.split(",") if e.strip()]
    mtypes = (list(MTYPES) if args.which == "all"
              else [m.strip() for m in args.which.split(",") if m.strip()])
    device = torch.device("cpu")

    from artijepa.eval_phoneme import load_probe, predict

    mats = {}                              # encoder -> {mtype: (M, rows, cols)}
    kappas, recalls = {}, {}
    kkey = ("tests", "test_lss") if args.split == "test_lss" else ("test",)
    for enc in encoders:
        print(f"[confmat] === {enc} (seed {args.seed}, split {args.split}) ===")
        pt = find_probe(enc, args.seed, eval_dir, feat_root, args.split)
        probe, ck = load_probe(pt, device="cpu")
        km = ck.get("metrics", {}) or {}
        for k in kkey:
            km = (km or {}).get(k, {})
        kappas[enc] = (km or {}).get("kappa")
        feats, labels = load_split_cache(ck["feature_tag"], feat_root, args.split)
        print(f"[confmat]   probe={os.path.basename(pt)} tag={ck['feature_tag']} "
              f"{args.split}_kappa={kappas[enc]} feats={feats.shape}")
        pred = predict(probe, feats, device)              # [N,T'] argmax over classes
        flat_t, flat_p = labels.reshape(-1), pred.reshape(-1)
        keep = flat_t != P.IGNORE_INDEX                   # drop padded tokens only
        true_ids, pred_ids = flat_t[keep], flat_p[keep]
        mats[enc] = build_matrices(true_ids, pred_ids)
        recalls[enc] = {m: macro_recall(mats[enc][m][0]) for m in mats[enc]}
        print(f"[confmat]   macro-recall  "
              + "  ".join(f"{m}={recalls[enc][m]:.3f}" for m in MTYPES))

    # -- macro-recall summary table (class-balanced diagonal) ------------------
    print(f"\n===== macro-recall (mean per-class diagonal, usc_lss {args.split}) =====")
    kcol = f"{args.split}_kappa"
    hdr = f"{'encoder':<18}" + "".join(f"{m:>13}" for m in MTYPES) + f"{kcol:>16}"
    print(hdr)
    for enc in encoders:
        k = kappas[enc]
        print(f"{enc:<18}" + "".join(f"{recalls[enc][m]:>13.3f}" for m in MTYPES)
              + (f"{k:>16.3f}" if k is not None else f"{'-':>16}"))

    # -- per-matrix-type cross-encoder comparison (cols = encoders) ------------
    for mtype in mtypes:
        pw, ph, fscale = PANEL[mtype]
        nc = len(encoders)
        fig, axes = plt.subplots(1, nc, figsize=(pw * nc, ph), squeeze=False)
        im = None
        for j, enc in enumerate(encoders):
            M, rows, cols = mats[enc][mtype]
            kap = f"  k={kappas[enc]:.3f}" if kappas[enc] is not None else ""
            title = f"{enc}{kap}\nmacro-recall={recalls[enc][mtype]:.3f}"
            im = plot_confmat(axes[0][j], M, rows, cols, title, fscale)
        fig.suptitle(f"{mtype} confusion (rows=true, cols=predicted, row-normalized "
                     f"recall %)  -  usc_lss {args.split}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.97, 0.95])
        cax = fig.add_axes([0.975, 0.12, 0.008, 0.72])
        fig.colorbar(im, cax=cax, label="recall")
        fp = os.path.join(out_dir, f"confmat_{mtype}_compare_{args.split}_s{args.seed}.png")
        fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"[confmat] wrote {fp}")

    # -- per-encoder figure: the requested granularities side by side ----------
    for enc in encoders:
        row = [m for m in MTYPES if m in mtypes]
        widths = [PANEL[m][0] for m in row]
        fig, axes = plt.subplots(1, len(row), squeeze=False,
                                 figsize=(sum(widths), max(PANEL[m][1] for m in row)),
                                 gridspec_kw={"width_ratios": widths})
        im = None
        for j, mtype in enumerate(row):
            M, rows, cols = mats[enc][mtype]
            im = plot_confmat(axes[0][j], M, rows, cols,
                              f"{mtype}  (macro-recall={recalls[enc][mtype]:.3f})",
                              PANEL[mtype][2])
        kap = f"  {args.split}-kappa={kappas[enc]:.3f}" if kappas[enc] is not None else ""
        fig.suptitle(f"{enc}{kap}  -  phoneme confusion (usc_lss {args.split}, "
                     "row-normalized)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = os.path.join(out_dir, f"confmat_{enc}_{args.split}_s{args.seed}.png")
        fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"[confmat] wrote {fp}")

    # -- dump raw + normalized matrices to JSON (one file, all encoders) -------
    dump = {"seed": args.seed, "split": f"usc_lss_{args.split}", "encoders": {}}
    for enc in encoders:
        edump = {f"{args.split}_kappa": kappas[enc], "macro_recall": recalls[enc],
                 "matrices": {}}
        for mtype in mtypes:
            M, rows, cols = mats[enc][mtype]
            supp = M.sum(1, keepdims=True)
            Mn = M / np.where(supp == 0, 1, supp)         # row-normalized recall
            edump["matrices"][mtype] = {
                "rows": rows, "cols": cols,
                "counts": M.astype(int).tolist(),
                "recall": Mn.tolist(),
            }
        dump["encoders"][enc] = edump
    jp = os.path.join(out_dir, f"confmat_values_{args.split}_s{args.seed}.json")
    with open(jp, "w") as fh:
        json.dump(dump, fh, indent=2)
    print(f"[confmat] wrote {jp}")

    print(f"[confmat] done -> {out_dir}")


if __name__ == "__main__":
    main()
