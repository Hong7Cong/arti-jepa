#!/usr/bin/env python3
"""Confusion matrix for the stutter BINARY (fluent/disfluent) LOSO probe.

The detection counterpart of `plot_stutter_type3_confmat.py`, on the *same encoder
and byte-identical feature cache* (tag tssl256_215) so the two matrices are directly
comparable: type-3 typing collapses to chance, binary detection does not.

Pooled over all 3901 held-out clips, summed across the available seeds; cells are
row-normalized to recall (%), with the raw summed count underneath. Single-hue
sequential ramp (magnitude); diagonal (correct class) outlined.

NOTE this tag was run at seeds 0 and 1 only (2 seeds), vs 3 for type-3 -- printed
in the subtitle so the two figures aren't misread as equal-N.

PROVENANCE -- .pooled.confusion read back from, summed over:
  stutter_binary_tssl256_215_b5da470386_attentive_loso_s{0,1}.json
frozen combined T-SSL ckpt_215, `attentive` probe, 256px/32f full grid, LOSO/7 PWS,
duration-matched fluent negatives (build_seed 0).

RUN:  python -m artijepa.plot_stutter_binary_confmat
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

EVAL_DIR = "/scratch1/hongn/artijepa/eval/stutter_binary"
OUT_PNG = f"{EVAL_DIR}/binary_confusion_matrix.png"
GLOB = f"{EVAL_DIR}/stutter_binary_tssl256_215_b5da470386_attentive_loso_s*.json"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("seqblue", ["#eef4fc", "#2a78d6", "#123a6b"])


def load():
    fs = [f for f in sorted(glob.glob(GLOB)) if "bak" not in f]
    ds = [json.load(open(f)) for f in fs]
    classes = ds[0]["classes"]
    C = sum(np.array(d["pooled"]["confusion"]) for d in ds)
    kap = np.mean([d["pooled"]["cohen_kappa"] for d in ds])
    mf1 = np.mean([d["pooled"]["macro_f1"] for d in ds])
    return classes, C, len(ds), mf1, kap


def main():
    classes, C, nseed, mf1, kap = load()
    row = C.sum(1, keepdims=True)
    P = C / row  # recall per (true,pred) cell

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(P, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")

    n = len(classes)
    for i in range(n):
        for j in range(n):
            val = P[i, j]
            tc = "#ffffff" if val > 0.45 else INK
            ax.text(j, i - 0.10, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=tc)
            ax.text(j, i + 0.19, f"{C[i, j]}", ha="center", va="center",
                    fontsize=10.5, color=tc if val > 0.45 else MUTED)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               ec="#008300", lw=2.6, zorder=4))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=12.5, color=INK2)
    ax.set_yticklabels(classes, fontsize=12.5, color=INK2, rotation=90, va="center")
    ax.set_xlabel("predicted class", fontsize=12, color=INK2, labelpad=8)
    ax.set_ylabel("true class", fontsize=12, color=INK2, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title("Stutter binary confusion - fluent / disfluent\n"
                 f"LOSO recall (%); macro-F1 {mf1:.2f}, κ {kap:.2f}",
                 fontsize=13, color=INK, pad=40, loc="left")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9, color=INK2, labelcolor=INK2, length=0)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    for i, c in enumerate(classes):
        print(f"  true {c:10s} recall={P[i,i]*100:5.1f}%  -> " +
              ", ".join(f"{classes[j]} {P[i,j]*100:.0f}%" for j in range(n)))


if __name__ == "__main__":
    main()
