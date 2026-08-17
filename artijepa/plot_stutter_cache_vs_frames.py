#!/usr/bin/env python3
"""Feature-cache cost vs temporal parameterization for the stutter binary probe.

X = how the clip is sampled in time (fixed 16f/32f/200f, or dynamic at a target
sample_fps); Y = the on-disk fp16 feature cache; each point is annotated with its
pooled LOSO macro-F1. Two series, because cache size is dominated by *what is
cached per frame*, not by frame count alone:

  full grid       -- the [T'*S', D] token grid is kept (probes `attentive` /
                     `seq_attentive_lstm`). ~256x bigger, and the axis on which
                     the fps sweep was actually run.
  spatially pooled-- S' is mean-pooled first, so only [T', D] lands on disk
                     (probes `pooled_attentive` / `seq_attentive`).

Plotting both on one cache axis without the distinction is the trap: 200f looks
"cheap" at 763 MB only because that row is spatially pooled; the same 200 frames
on the full grid is ~190 GiB (docs/STUTTERING.md sec 10.6 estimate, never run).

PROVENANCE -- every plotted point is read back from the result JSON / measured
with du, not transcribed from the markdown:
  full grid   32f      /scratch1/.../stutter_binary_tssl256_215_b5da470386_attentive_loso_s0.json
              dyn 25   ..._dyn_tssl256comb215_dyn_5bccb7ebec_seq_attentive_lstm_loso_s0.json
              dyn 50   ..._dyn_tssl256comb215_dyn_3531522779_seq_attentive_lstm_loso_s0.json
              dyn ~99  cache only -- run dropped (RESULTS_stutter.md), no F1
              200f     never run; cache is a sec-10.6 estimate
  pooled      32f      ..._tssl256_71bc4397eb_pooled_attentive_loso_s0.json
              dyn ~99  ..._dyn_tssl256_dyn_abc21683e1_seq_attentive_loso_s0.json
  16f         VideoMAE-L only (224px, different encoder family) -- drawn with a
              square marker because it is NOT the same encoder as the rest.

NOT PLOTTED (doc-only, unverifiable in the results dir -- see the caveat printed
by this script): docs/STUTTERING.md sec 10.8 lists a *pooled* dynamic 25 fps row
at 250 MB / 0.767, but the only pooled dynamic JSON on disk is `abc21683e1`,
which self-reports sample_fps=native at 0.797 GiB / 0.7291.

RUN:  python -m artijepa.plot_stutter_cache_vs_frames
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG = "/scratch1/hongn/artijepa/eval/stutter_binary/cache_vs_frames.png"

# categorical x, ordered by effective temporal coverage (dynamic p50 windows x 32f)
XORDER = ["16f", "32f", "dyn 25fps\n(~64f)", "dyn 50fps\n(~96f)", "dyn native\n(~192f)", "200f"]
XPOS = {k: i for i, k in enumerate(XORDER)}

C_GRID = "#2a78d6"   # categorical slot 1
C_POOL = "#eb6834"   # categorical slot 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"

# (x, cache GiB, macro-F1 or None, kind)
#   kind: "solid"=measured cache + measured F1, "nof1"=measured cache no F1,
#         "est"=estimated cache no F1, "other"=different encoder (VideoMAE-L)
GRID = [
    ("32f",                 30.477, 0.8432, "solid"),
    ("dyn 25fps\n(~64f)",   62.422, 0.7913, "solid"),
    ("dyn 50fps\n(~96f)",  110.227, 0.7660, "solid"),
    ("dyn native\n(~192f)", 203.899, None,  "nof1"),
    ("200f",               190.0,   None,   "est"),
    ("16f",                 11.667, 0.8043, "other"),
]
POOL = [
    ("32f",                  0.119, 0.8081, "solid"),
    ("dyn native\n(~192f)",  0.797, 0.7291, "solid"),
    ("16f",                  0.060, 0.8064, "other"),
]


def _draw(ax, pts, color, label):
    line = [p for p in pts if p[3] in ("solid", "nof1")]
    line.sort(key=lambda p: XPOS[p[0]])
    ax.plot([XPOS[p[0]] for p in line], [p[1] for p in line],
            color=color, lw=2, zorder=2, label=label)
    for x, gib, f1, kind in pts:
        marker = "s" if kind == "other" else "o"
        filled = kind in ("solid", "other")
        ax.plot(XPOS[x], gib, marker=marker, ms=9, zorder=3,
                color=color, mfc=color if filled else "#fcfcfb",
                mec=color, mew=2, ls="none")
        txt = f"F1 {f1:.3f}" if f1 is not None else ("est. cache\nnot run" if kind == "est" else "no F1\n(dropped)")
        ax.annotate(txt, (XPOS[x], gib), textcoords="offset points",
                    xytext=(0, 15 if kind != "est" else 14), ha="center",
                    fontsize=9, color=INK if f1 is not None else MUTED,
                    fontweight="bold" if f1 is not None else "normal")


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    _draw(ax, GRID, C_GRID, "full grid  [T'xS', D]")
    _draw(ax, POOL, C_POOL, "spatially pooled  [T', D]")

    ax.set_yscale("log")
    ax.set_xticks(range(len(XORDER)))
    ax.set_xticklabels(XORDER, fontsize=10, color=INK2)
    ax.set_xlim(-0.45, len(XORDER) - 0.4)
    ax.set_ylim(0.03, 700)
    ax.set_ylabel("feature cache on disk (GiB, log)", fontsize=11, color=INK2)
    ax.set_xlabel("temporal parameterization of the clip", fontsize=11, color=INK2)
    ax.set_title("Stutter binary probe: cache cost vs temporal sampling\n"
                 "frozen T-SSL ViT-L @256px (+ VideoMAE-L 16f), LOSO, seed 0 - annotated with pooled macro-F1",
                 fontsize=13, color=INK, pad=14, loc="left")

    ax.grid(axis="y", color="#e3e2dd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, length=0)

    leg = ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax.annotate("square = VideoMAE-L @224px (different encoder; only 16f arm that exists)\n"
                "hollow = no macro-F1 for that point",
                xy=(0.0, -0.16), xycoords="axes fraction", fontsize=9, color=MUTED, va="top")
    ax.annotate("200f full-grid never run (~190 GiB est.);\n"
                "the 763 MB 200f row in the docs is spatially pooled",
                xy=(XPOS["200f"], 190.0), textcoords="offset points", xytext=(0, -52),
                ha="center", fontsize=8.5, color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print("CAVEAT: docs/STUTTERING.md sec 10.8 lists a pooled dynamic 25fps row "
          "(250 MB / 0.767) with no matching JSON in the results dir; the only pooled "
          "dynamic result on disk self-reports sample_fps=native. Not plotted.")


if __name__ == "__main__":
    main()
