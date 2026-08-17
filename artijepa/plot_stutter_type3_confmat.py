#!/usr/bin/env python3
"""Confusion matrix for the stutter TYPE-3 (block/rep/pro) LOSO probe.

Pooled over all held-out clips, summed across the 3 seeds (seeds 0/1/2) so every
disfluent clip contributes 3 predictions; cells are row-normalized to recall (%),
with the raw 3-seed count underneath. Row-normalization is the right reading here
because the story is *recall collapse onto `block`*, not raw volume.

Color is a single-hue sequential ramp (magnitude), per the data-viz form rule for
a confusion matrix; the diagonal (correct class) is outlined so it reads without
relying on the color alone.

PROVENANCE -- .pooled.confusion read back from, summed over:
  stutter_type3_tssl256_215_b5da470386_attentive_lstm_loso_s{0,1,2}.json
frozen combined T-SSL ckpt_215, attentive_lstm, 256px/32f full grid, LOSO/7 PWS.

RUN:  python -m artijepa.plot_stutter_type3_confmat
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

EVAL_DIR = "/scratch1/hongn/artijepa/eval/stutter_type"
OUT_PNG = f"{EVAL_DIR}/type3_confusion_matrix.png"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
# sequential single hue: surface -> categorical blue slot 1 (#2a78d6), light->dark
CMAP = LinearSegmentedColormap.from_list("seqblue", ["#eef4fc", "#2a78d6", "#123a6b"])


def load():
    fs = sorted(glob.glob(f"{EVAL_DIR}/stutter_type3_*loso_s*.json"))
    ds = [json.load(open(f)) for f in fs]
    classes = ds[0]["classes"]
    C = sum(np.array(d["pooled"]["confusion"]) for d in ds)
    return classes, C, len(ds)


def main():
    classes, C, nseed = load()
    row = C.sum(1, keepdims=True)
    P = C / row  # recall per (true,pred) cell, in [0,1]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(P, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")

    n = len(classes)
    for i in range(n):
        for j in range(n):
            val = P[i, j]
            txt_color = "#ffffff" if val > 0.45 else INK
            ax.text(j, i - 0.10, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=txt_color)
            ax.text(j, i + 0.19, f"{C[i, j]}", ha="center", va="center",
                    fontsize=10, color=txt_color if val > 0.45 else MUTED)
    # outline the diagonal = the correct class
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               ec="#e34948", lw=2.4, zorder=4))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=12, color=INK2)
    ax.set_yticklabels(classes, fontsize=12, color=INK2)
    ax.set_xlabel("predicted class", fontsize=12, color=INK2, labelpad=8)
    ax.set_ylabel("true class", fontsize=12, color=INK2, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title("Stutter type-3 confusion - block / rep / pro\n"
                 "LOSO recall (%); macro-F1 0.38, κ 0.11",
                 fontsize=13, color=INK, pad=40, loc="left")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9, color=INK2, labelcolor=INK2, length=0)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    for i, c in enumerate(classes):
        print(f"  true {c:6s} recall={P[i,i]*100:5.1f}%  -> pred " +
              ", ".join(f"{classes[j]} {P[i,j]*100:.0f}%" for j in range(n)))


if __name__ == "__main__":
    main()
