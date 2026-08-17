#!/usr/bin/env python3
"""Composition of each disfluency type: silence / vowel / consonant / other.

Reads disfluency_composition.json (analyze_disfluency_composition). Stacked bars
show what a window of each type is made of, and why "silence" dominates disfluent
windows: it is concentrated in blocks (silent holds) and repetitions (inter-attempt
pauses); prolongations are sustained SOUND and stay low-silence -- proof the silence
is the dysfluent behavior itself, not a forced-alignment artifact.

RUN:  python -m artijepa.plot_disfluency_composition
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON = "/scratch1/hongn/artijepa/eval/stutter_type/disfluency_composition.json"
OUT_PNG = "/scratch1/hongn/artijepa/eval/stutter_type/disfluency_composition.png"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
C_SIL = "#9a9992"   # silence  (neutral)
C_VOW = "#2a78d6"   # vowel    (slot 1)
C_CON = "#eb6834"   # consonant(slot 2)
C_OTH = "#d9d8d2"   # other    (light neutral)

LABELS = {"block": "block\n(silent hold)", "rep": "rep\n(repeated attempts)",
          "pro": "pro\n(sustained sound)", "fluent": "fluent\n(negative)"}
ORDER = ["fluent", "pro", "rep", "block"]   # bottom->top = rising silence


def main():
    d = json.load(open(JSON))
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    y = np.arange(len(ORDER))
    for i, g in enumerate(ORDER):
        r = d[g]
        sil, vow, con = r["silence"] * 100, r["vowel"] * 100, r["consonant"] * 100
        oth = max(0.0, 100 - sil - vow - con)
        segs = [(sil, C_SIL, "silence"), (con, C_CON, "consonant"),
                (vow, C_VOW, "vowel"), (oth, C_OTH, "other")]
        left = 0.0
        for val, col, name in segs:
            ax.barh(i, val, left=left, color=col, zorder=3,
                    edgecolor=SURFACE, linewidth=1.5,
                    label=name if i == 0 else None)
            if val >= 7:
                tc = "#ffffff" if col in (C_SIL, C_CON, C_VOW) else INK2
                ax.text(left + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=10, color=tc, fontweight="bold")
            left += val
        ax.text(101, i, f"n={r['n']}\n{r['mean_dur']:.1f}s avg",
                va="center", fontsize=8.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[g] for g in ORDER], fontsize=10.5, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of window-time", fontsize=10.5, color=INK2)
    ax.set_title("What each disfluency type is made of - and why disfluent windows are silent\n"
                 "silence tracks the behavior: block (hold) > rep (pauses) > pro (sound) > fluent",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    ax.grid(axis="x", color="#e3e2dd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, length=0)

    # legend in fixed order
    from matplotlib.patches import Patch
    handles = [Patch(fc=c, label=l) for c, l in
               [(C_SIL, "silence"), (C_CON, "consonant"), (C_VOW, "vowel"),
                (C_OTH, "other")]]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=4, frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.text(0.01, -0.10,
             "Prolongations stay low-silence (25%) despite being highly non-canonical "
             "speech -> the <sil> is real held/paused articulation, not the aligner "
             "failing on stuttered speech. Note pro is the one type MORE consonant-heavy "
             "than fluent (44% vs 41%): you sustain a consonant.",
             fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
