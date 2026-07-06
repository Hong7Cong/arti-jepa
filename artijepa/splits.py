"""Subject-disjoint train/val/test splits (Arti-JEPA A.8).

Splitting by *speaker* prevents anatomy leakage -- the headline requirement for
subject-independent evaluation. We also keep all repeats/utterances of a subject
on the same side of the split. Reads the manifest from build_manifest.py and
writes it back with an added `split` column.

Usage:
    python -m artijepa.splits \
        --manifest /data2/hongn/artijepa/manifest.csv \
        --out /data2/hongn/artijepa/manifest_split.csv \
        --val-frac 0.12 --test-frac 0.12 --seed 0
"""

import argparse
import csv
import random
from collections import defaultdict


def assign_splits(subjects, val_frac, test_frac, seed):
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    # frac > 0 always lands at least 1 subject in the split; frac == 0 yields a
    # true empty split (e.g. --val-frac 0 --test-frac 0 -> all data in train).
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
    test = set(subjects[:n_test])
    val = set(subjects[n_test:n_test + n_val])
    train = set(subjects[n_test + n_val:])
    mapping = {}
    for s in train:
        mapping[s] = "train"
    for s in val:
        mapping[s] = "val"
    for s in test:
        mapping[s] = "test"
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data2/hongn/artijepa/manifest.csv")
    ap.add_argument("--out", default="/data2/hongn/artijepa/manifest_split.csv")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--test-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    subjects = {r["subject"] for r in rows}
    mapping = assign_splits(subjects, args.val_frac, args.test_frac, args.seed)

    counts = defaultdict(lambda: [0, 0])  # split -> [n_subjects, n_videos]
    seen_subj = defaultdict(set)
    for r in rows:
        sp = mapping[r["subject"]]
        r["split"] = sp
        counts[sp][1] += 1
        seen_subj[sp].add(r["subject"])
    for sp in seen_subj:
        counts[sp][0] = len(seen_subj[sp])

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    for sp in ("train", "val", "test"):
        ns, nv = counts[sp]
        print(f"  {sp:5s}: {ns:3d} subjects, {nv:4d} videos")


if __name__ == "__main__":
    main()
