"""Render the segment-level per-speaker×session confusion grid for any Test-4 task,
straight from a finished `glossphg_<enc>_s<seed>.json` -- NO GPU, NO re-extraction.

`eval_gloss_groups.py` only draws the per-speaker grid for the tasks in `--figs-tasks`
(default `manner,place`, since the 25x25 consonants grid x7 cells is unwieldy). This
helper reconstructs the in-memory `res` shape those plotters expect from the stored
per_group confusion matrices and reuses `plot_cm_per_speaker` verbatim, so you can add
the `vowels` / `consonants` grids after the fact without a SLURM job.

    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.plot_glossphg_perspk \
        --json /scratch1/hongn/artijepa/eval/gloss/glossphg_tssl256comb215_s0.json \
        --tasks vowels,consonants           # default: all four
"""
import argparse
import json
import os

import artijepa.eval_gloss_groups as GG


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="glossphg_<enc>_s<seed>.json")
    ap.add_argument("--tasks", default="vowels,consonants,manner,place")
    ap.add_argument("--out-dir", default=None,
                    help="default: the JSON's directory")
    args = ap.parse_args()

    d = json.load(open(args.json))
    enc, seed = d["encoder"], d["seed"]
    sfx = "" if d.get("speaker", "pooled") == "pooled" else f"_{d['speaker']}"
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))

    # res[cond]["_groups"] + res[cond][task]={"pooled":..,"per_group":..} is exactly
    # what _grid() and plot_cm_per_speaker() read.
    res = {}
    for cond in GG.CONDS:
        res[cond] = {"_groups": d["groups"][cond]}
        for task, t in d["conditions"][cond].items():
            if isinstance(t, dict) and "per_group" in t:
                res[cond][task] = t

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task not in d["conditions"]["pre"]:
            print(f"[plot] skip {task!r}: not in {os.path.basename(args.json)}")
            continue
        names = d["conditions"]["pre"][task]["pooled"]["class_names"]
        GG.plot_cm_per_speaker(res, task, names, out_dir, enc, seed, sfx)


if __name__ == "__main__":
    main()
