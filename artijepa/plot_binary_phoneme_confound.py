#!/usr/bin/env python3
"""Two-panel summary of the stutter-binary phoneme/silence confound analysis.

Reads phoneme_confound.json (artijepa.analyze_binary_phoneme_confound).

Left  : phone-class occupancy (fraction of window-time), disfluent vs fluent.
        The user's hypothesis was "disfluent biased toward stop consonants" -- the
        bars show the opposite (disfluent windows are LESS consonant/vowel-heavy);
        the one class that explodes is SILENCE.
Right : how far a no-video classifier gets from each feature set, LOSO, vs the
        video encoder -- attributing the leakage to silence, not phoneme identity.

RUN:  python -m artijepa.plot_binary_phoneme_confound
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON = "/scratch1/hongn/artijepa/eval/stutter_binary/phoneme_confound.json"
OUT_PNG = "/scratch1/hongn/artijepa/eval/stutter_binary/phoneme_confound.png"

C_DIS = "#2a78d6"   # disfluent  (categorical slot 1)
C_FLU = "#eb6834"   # fluent     (categorical slot 2)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"

CLASSES = ["vowel", "stop", "affricate", "fricative", "nasal", "liquid", "glide",
           "silence"]


def main():
    d = json.load(open(JSON))
    op = d["occupancy_pos"]; on = d["occupancy_neg"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.8),
                                   gridspec_kw=dict(width_ratios=[1.55, 1.0]))
    fig.patch.set_facecolor(SURFACE)
    for a in (axL, axR):
        a.set_facecolor(SURFACE)

    # -------- left: grouped occupancy bars ---------------------------------- #
    x = np.arange(len(CLASSES)); w = 0.4
    dv = [op[c] * 100 for c in CLASSES]; fv = [on[c] * 100 for c in CLASSES]
    axL.bar(x - w / 2, dv, w, color=C_DIS, label="disfluent", zorder=3)
    axL.bar(x + w / 2, fv, w, color=C_FLU, label="fluent", zorder=3)
    for xi, v in zip(x - w / 2, dv):
        axL.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                 color=INK2)
    for xi, v in zip(x + w / 2, fv):
        axL.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                 color=INK2)
    axL.set_xticks(x)
    axL.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=10, color=INK2)
    axL.set_ylabel("% of window-time on this phone class", fontsize=10.5, color=INK2)
    axL.set_ylim(0, 60)
    axL.set_title("Phone-class occupancy: disfluent windows are silence-heavy,\n"
                  "not consonant-heavy (consonant time 27% vs 41%)",
                  fontsize=11.5, color=INK, loc="left", pad=8)
    axL.grid(axis="y", color="#e3e2dd", lw=0.8, zorder=0)
    axL.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axL.spines[s].set_color("#d5d4cf")
    axL.tick_params(colors=INK2, length=0)
    leg = axL.legend(loc="upper center", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    # -------- right: no-video leakage ladder -------------------------------- #
    bars = [
        ("phone identity\n(silence removed)", d["nosilence_macro_f1"], "#c9c8c2"),
        ("full phone+silence\nhistogram", d["phoneme_only_macro_f1"], "#9a9992"),
        ("SILENCE fraction\nalone (1 feature)", d["silence_only_macro_f1"], C_DIS),
        ("video encoder\n(V-JEPA2 T-SSL)", d["video_macro_f1"], "#123a6b"),
    ]
    yp = np.arange(len(bars))
    vals = [b[1] for b in bars]
    axR.barh(yp, vals, color=[b[2] for b in bars], zorder=3, height=0.62)
    for yi, v in zip(yp, vals):
        axR.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=11,
                 fontweight="bold", color=INK)
    axR.axvline(0.5, color="#e34948", lw=1.4, ls="--", zorder=4)
    axR.text(0.5, -0.62, "chance", color="#e34948", fontsize=9, ha="center", va="top")
    axR.set_yticks(yp)
    axR.set_yticklabels([b[0] for b in bars], fontsize=9.5, color=INK2)
    axR.set_xlim(0.45, 0.92)
    axR.set_xlabel("macro-F1 (LOSO, no video except last bar)", fontsize=10, color=INK2)
    axR.set_title("Silence alone reproduces 0.74 of the 0.83;\n"
                  "phoneme identity alone is chance (0.51)",
                  fontsize=11.5, color=INK, loc="left", pad=8)
    axR.grid(axis="x", color="#e3e2dd", lw=0.8, zorder=0)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axR.spines[s].set_color("#d5d4cf")
    axR.tick_params(colors=INK2, length=0)

    fig.suptitle("Does the stutter-binary 0.83 come from a phoneme confound?  "
                 "-- No: it's silence, not consonant identity.",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.03,
             f"phones-tier coverage {d['n_pos_cov']+d['n_neg_cov']}/{d['n_rows']} rows "
             "(65%, uneven per speaker). Silence-in-a-speech-window IS disfluency "
             "(blocks are silent holds), so this caps how much the 0.83 proves about "
             "learned articulatory *dynamics* rather than being an artifact.",
             fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
