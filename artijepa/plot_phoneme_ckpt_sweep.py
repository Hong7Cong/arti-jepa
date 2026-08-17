#!/usr/bin/env python3
"""Kappa-vs-checkpoint plot for the tssl_vitl_256_combined phoneme sweep.

Aggregates the Phase-1 headline probe (`attentive` x CE) result JSONs across the
pretraining checkpoints and plots Cohen's kappa vs pretraining epoch for the three
splits (val sub030, in-domain test sub043, x-domain test_lss usc_s1).

Points come from:
  ckpt 0   -> stock V-JEPA2 ViT-L baseline (`pretrained`); the T-SSL run is a
              domain-adaptive continuation FROM this init (tssl_vitl_256_combined.yaml
              `pretrained: true`), so epoch 0 == the pre-adaptation encoder. Same
              attentive x CE probe / 256px / manifest / 3 seeds as the rest.
  ckpt 100 -> tag tssl256comb100  (Phase 1)
  ckpt 215 -> tag tssl256comb215  (Phase 1/2, final)
  ckpt 60/150/200 -> tags tssl256comb{60,150,200}  (scripts/26_phoneme_ckpt_sweep.sbatch)

Each point = mean +/- std over available seeds. Only result files that carry a
`tests.test_lss` block (the combined cross-domain manifest) are used, so the stale
annot16-only comb100 hash (d36dd0c874) is ignored automatically.

RUN:  python -m artijepa.plot_phoneme_ckpt_sweep
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = "/scratch1/hongn/artijepa/eval"
EPOCHS = [0, 60, 100, 150, 200, 215]
OUT_PNG = "/scratch1/hongn/artijepa/eval/cksweep/phoneme_kappa_vs_ckpt.png"
# (json key path, legend label)
SPLITS = [
    ("val.kappa",             "val (sub030)"),
    ("test.kappa",            "in-domain test (sub043)"),
    ("tests.test_lss.kappa",  "x-domain test_lss (usc_s1)"),
]


def _get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def collect(epoch):
    """Return {split_key: [values over seeds]} for one ckpt epoch (test_lss files only).

    epoch 0 = the stock V-JEPA2 baseline the T-SSL run was initialized from
    (`pretrained` tag), so it globs the pretrained result files instead of a comb<E> tag.
    """
    stem = "pretrained" if epoch == 0 else f"tssl256comb{epoch}"
    pat = os.path.join(EVAL_DIR, f"phoneme_usc_lss_{stem}sp_*_attentive_ce_s*.json")
    vals = {k: [] for k, _ in SPLITS}
    for fp in sorted(glob.glob(pat)):
        d = json.load(open(fp))
        if _get(d, "tests.test_lss.kappa") is None:
            continue  # skip stale annot16-only manifest hash
        for k, _ in SPLITS:
            v = _get(d, k)
            if v is not None:
                vals[k].append(v)
    return vals


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rows = {e: collect(e) for e in EPOCHS}

    # ---- table ----
    print(f"{'ckpt':>6} {'nseed':>6} | " + " | ".join(f"{lbl:>26}" for _, lbl in SPLITS))
    for e in EPOCHS:
        v = rows[e]
        nseed = max((len(v[k]) for k, _ in SPLITS), default=0)
        cells = []
        for k, _ in SPLITS:
            xs = v[k]
            cells.append(f"{np.mean(xs):.3f} +/- {np.std(xs):.3f}" if xs else f"{'--':>13}")
        print(f"{e:>6} {nseed:>6} | " + " | ".join(f"{c:>26}" for c in cells))

    # ---- plot ----
    plt.figure(figsize=(7.2, 4.8))
    for k, lbl in SPLITS:
        xs, ys, es = [], [], []
        for e in EPOCHS:
            v = rows[e][k]
            if v:
                xs.append(e); ys.append(np.mean(v)); es.append(np.std(v))
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=1.8, label=lbl)
    plt.xlabel("pretraining checkpoint (epoch)")
    plt.ylabel("Cohen's $\\kappa$ (phoneme, frame-level)")
    plt.title("T-SSL 256 combined: phoneme $\\kappa$ vs pretraining checkpoint\n(attentive probe · CE · 256px)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
