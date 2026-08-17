#!/usr/bin/env python3
"""Representation-collapse diagnostics over pretraining -- one panel per metric.

Reads `runs/<name>/diagnostics.jsonl` (written every epoch by `tssl_train.py:297`
via `collapse.feature_diagnostics`) for the three T-SSL runs and plots the three
label-free canaries against epoch:

    feature_std      (collapse -> 0)
    effective_rank   (collapse -> 0, out of dim=1024)
    mean_abs_cosine  (collapse -> 1)

Three panels, NOT three lines on one axis: the metrics live on different scales and
a shared y would be a dual-axis chart. x (epoch) is shared; the 50-epoch runs simply
end early. Dotted verticals mark the checkpoints that were phoneme-evaluated
(`plot_phoneme_ckpt_sweep.py`), so encoder quality can be read against the
diagnostics at the same epoch.

RUN:  PYTHONPATH=.:dev_artiJEPA python -m artijepa.plot_collapse_diag
OUT:  /scratch1/hongn/artijepa/eval/pretrain/collapse_diag.png (+ .csv of the series)
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = "/scratch1/hongn/artijepa/runs"
OUT_DIR = "/scratch1/hongn/artijepa/eval/pretrain"
OUT_PNG = os.path.join(OUT_DIR, "collapse_diag.png")
OUT_CSV = os.path.join(OUT_DIR, "collapse_diag.csv")

# fixed categorical order (dataviz reference palette slots 1-3) -- colour follows the
# run, never its rank, so adding a run never repaints the others.
# n_mon = clips the monitor actually sees = min(val rows, probe_max_batches x
# probe_batch_size); one centre clip per video (`sampling='crop'`). It sets the
# effective_rank CEILING at min(n_mon - 1, 1024), which differs across runs.
RUNS = [
    ("tssl_vitl_128", "128px / 75-spk", "#2a78d6", 279),           # split val 279, bs32
    ("tssl_vitl_256", "256px / 75-spk", "#eb6834", 128),           # alltrain val 128, bs8
    ("tssl_vitl_256_combined", "256px / +longitudinal", "#1baf7a", 128),  # comb val 128
]
# (key, title, subtitle, collapse direction, ylim)
PANELS = [
    ("feature_std", "feature_std", "mean per-dim std of pooled features",
     "collapse → 0", (0, None)),
    ("effective_rank", "effective_rank",
     "exp(entropy of singular values) — ceiling is min(n_monitor−1, 1024), NOT 1024",
     "collapse → 0", (0, None)),
    ("mean_abs_cosine", "mean_abs_cosine", "mean |cos| between held-out clips",
     "collapse → 1", (0, 1.05)),
]
CKPTS = [60, 100, 150, 200, 215]      # phoneme-evaluated checkpoints (combined run)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load(run):
    """-> [{'epoch': int, metric: float, ...}] sorted, last write per epoch wins."""
    path = os.path.join(RUNS_DIR, run, "diagnostics.jsonl")
    if not os.path.exists(path):
        return []
    by_epoch = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "feature_std" in r:                 # skip conditioned-JEPA diagnostics
                by_epoch[int(r["epoch"])] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def main():
    series = {run: load(run) for run, _, _, _ in RUNS}
    for run, _, _, n_mon in RUNS:
        print(f"{run}: {len(series[run])} epochs, n_monitor={n_mon}, "
              f"eff_rank ceiling={min(n_mon - 1, 1024)}")

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    for ax, (key, title, sub, direction, ylim) in zip(axes, PANELS):
        for x in CKPTS:                            # recessive: chrome, not data
            ax.axvline(x, color=GRID, lw=1, ls=":", zorder=0)
        if key == "effective_rank":                # the ceilings differ per run
            for n_mon in sorted({m for _, _, _, m in RUNS}):
                ceil = min(n_mon - 1, 1024)
                ax.axhline(ceil, color=GRID, lw=1.2, ls="--", zorder=1)
                ax.annotate(f"ceiling {ceil}  (n_monitor={n_mon})", (218, ceil),
                            xytext=(0, -3), textcoords="offset points",
                            fontsize=7.5, color=INK2, va="top", ha="right")
        for run, label, color, _ in RUNS:
            rows = series[run]
            if not rows:
                continue
            xs = [r["epoch"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, lw=2, color=color, label=label, zorder=3,
                    solid_capstyle="round")
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK2)               # direct label, text ink not series
        ax.set_ylim(*ylim)
        ax.set_ylabel(title, fontsize=10.5, color=INK)
        ax.set_title(f"{title} — {sub}   ({direction})", fontsize=10,
                     color=INK2, loc="left", pad=6)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    axes[-1].set_xlabel("pretraining epoch", fontsize=10.5, color=INK)
    axes[-1].set_xlim(0, 222)
    for x in CKPTS:
        axes[-1].annotate(f"ckpt {x}", (x, 0.02), rotation=90, fontsize=7.5,
                          color=INK2, ha="right", va="bottom")
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower right")
    fig.suptitle("Label-free representation-collapse diagnostics over T-SSL pretraining\n"
                 "held-out val — the monitor set differs per run (see n_monitor)",
                 fontsize=11.5, color=INK, x=0.02, ha="left", y=0.998,
                 va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "epoch", "feature_std", "effective_rank", "mean_abs_cosine"])
        for run, _, _, _ in RUNS:
            for r in series[run]:
                w.writerow([run, r["epoch"], r["feature_std"], r["effective_rank"],
                            r["mean_abs_cosine"]])
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
