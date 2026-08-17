#!/usr/bin/env python3
"""Macro-F1 vs temporal sampling (32f / dyn 25fps / dyn 50fps), artiJEPA vs V-JEPA2 stock.

Binary fluent-vs-disfluent, LOSO, full spatial grid, pooled macro-F1 over all 3901
held-out clips. Probe is `attentive` at fixed 32f and `seq_attentive_lstm` on the
dynamic path (variable length needs the recurrent head).

  artiJEPA T-SSL ViT-L (rt-MRI fine-tune, ckpt_215)  -- all three x positions
  V-JEPA2 ViT-L (FAIR stock)                         -- 32f ONLY

**The V-JEPA2 dynamic arm does not exist.** As of 2026-07-23 every dynamic-length
run and every ragged feature cache under feat_cache/stutter_binary/ is `tssl256*`;
there is no vjepa_pt dyn cache and no vjepa_pt seq_* result JSON. The two missing
points are drawn as explicit "not run" marks rather than interpolated, so the
figure shows the shape of the gap instead of hiding it.

Seed counts differ per point and are printed under each marker -- the dynamic runs
are single-seed (see RESULTS_stutter.md: frozen features make seeds cheap-ish but
the 62/110 GiB extractions were run once).

PROVENANCE -- read back from the result JSONs, not the markdown:
  artiJEPA 32f    stutter_binary_tssl256_215_b5da470386_attentive_loso_s{0,1}.json
  artiJEPA dyn25  ..._dyn_tssl256comb215_dyn_5bccb7ebec_seq_attentive_lstm_loso_s0.json
  artiJEPA dyn50  ..._dyn_tssl256comb215_dyn_3531522779_seq_attentive_lstm_loso_s0.json
  V-JEPA2  32f    stutter_binary_vjepa_pt_36c9b08905_attentive_loso_s{0,1,2}.json

RUN:  python -m artijepa.plot_stutter_f1_vs_sampling
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG = "/scratch1/hongn/artijepa/eval/stutter_binary/f1_vs_sampling_encoders.png"

XORDER = ["fixed 32f", "dynamic 25 fps", "dynamic 50 fps"]
XPOS = {k: i for i, k in enumerate(XORDER)}

C_ARTI = "#2a78d6"   # categorical slot 1
C_VJEP = "#eb6834"   # categorical slot 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"

# (x, mean macro-F1, sd, n_seeds)
ARTI = [("fixed 32f", 0.8301, 0.0185, 2),
        ("dynamic 25 fps", 0.7913, 0.0, 1),
        ("dynamic 50 fps", 0.7660, 0.0, 1)]
VJEP = [("fixed 32f", 0.7892, 0.0120, 3)]
VJEP_MISSING = ["dynamic 25 fps", "dynamic 50 fps"]


def _draw(ax, pts, color, label, dodge=0.0):
    # both encoders share the 32f tick; dodge so markers, whiskers and the value /
    # seed-count labels of the two series never land on top of each other
    xs = [XPOS[p[0]] + dodge for p in pts]
    ys = [p[1] for p in pts]
    errs = [p[2] for p in pts]
    # a one-point series still needs a visible legend swatch -> marker, not a bare line
    if len(pts) > 1:
        ax.plot(xs, ys, color=color, lw=2, zorder=2, label=label)
    else:
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=9, mec=SURFACE, mew=2,
                zorder=2, label=label, ls="none")
    ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=color, elinewidth=2,
                capsize=5, capthick=2, zorder=3)
    ax.plot(xs, ys, "o", ms=10, color=color, mfc=color, mec=SURFACE, mew=2,
            ls="none", zorder=4)
    for (x, y, sd, n) in pts:
        ax.annotate(f"{y:.3f}", (XPOS[x] + dodge, y + sd), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11,
                    fontweight="bold", color=INK)
        # anchor below the lower error cap so the label never sits inside the whisker
        ax.annotate(f"n={n} seed{'s' if n > 1 else ''}", (XPOS[x] + dodge, y - sd),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=8.5, color=MUTED)


def main():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    _draw(ax, ARTI, C_ARTI, "artiJEPA T-SSL ViT-L (rt-MRI fine-tune, ckpt_215)", dodge=-0.055)
    _draw(ax, VJEP, C_VJEP, "V-JEPA2 ViT-L (FAIR stock)", dodge=+0.055)

    # the two points that were never run
    for x in VJEP_MISSING:
        ax.annotate("not run", (XPOS[x], 0.7415), ha="center", fontsize=10.5,
                    color=C_VJEP, style="italic")
        ax.plot(XPOS[x], 0.7345, marker="x", ms=9, mew=2.2, color=C_VJEP, ls="none")

    # the 32f gap, the one place both encoders actually meet
    ax.annotate("", xy=(0.20, 0.8301), xytext=(0.20, 0.7892),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
    # opaque bbox: the artiJEPA trend line passes straight through this text
    ax.annotate("+0.041\nat the only\nmatched point", xy=(0.24, 0.8095),
                fontsize=9, color=INK2, va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.32", fc=SURFACE, ec="none"))

    ax.set_xticks(range(len(XORDER)))
    ax.set_xticklabels(XORDER, fontsize=11, color=INK2)
    ax.set_xlim(-0.42, len(XORDER) - 0.55)
    ax.set_ylim(0.725, 0.865)
    ax.set_ylabel("pooled macro-F1 (LOSO, 3901 held-out clips)", fontsize=11, color=INK2)
    ax.set_xlabel("temporal parameterization of the clip", fontsize=11, color=INK2)
    ax.set_title("Stutter binary: does the rt-MRI fine-tune's edge survive dynamic sampling?\n"
                 "full spatial grid, 256px - unanswerable past 32f until the V-JEPA2 dynamic arm is run",
                 fontsize=12.5, color=INK, pad=14, loc="left")

    ax.grid(axis="y", color="#e3e2dd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, length=0)

    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax.annotate("probe: `attentive` at fixed 32f, `seq_attentive_lstm` on the dynamic path "
                "(variable length requires the recurrent head)",
                xy=(0.0, -0.145), xycoords="axes fraction", fontsize=8.5, color=MUTED, va="top")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print("NOTE: V-JEPA2 stock has no dynamic-length run; 2 of its 3 points are 'not run', "
          "so the fixed-vs-dynamic comparison between encoders is not yet answerable.")


if __name__ == "__main__":
    main()
