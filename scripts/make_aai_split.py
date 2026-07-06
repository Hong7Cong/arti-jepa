#!/usr/bin/env python
"""Carve a SPEAKER-DISJOINT train/val/test split for the AAI (audio->video) run.

Self-supervised => no labels; we only reassign the `split` column so val/test hold
ENTIRELY unseen speakers (honest audio->video-latent generalization). Writes a NEW
manifest so the ongoing T-SSL run (which reads manifest_combined.csv) is untouched.

Held-out speakers are drawn from BOTH corpora (speaker75 + longitudinal) so eval
covers both distributions. Deterministic (seed=0): sort subjects per corpus, shuffle
with a fixed RNG, take the first N_test as test, next N_val as val, rest train.

    python dev_artiJEPA/scripts/make_aai_split.py \
        --in  /scratch1/hongn/artijepa/manifest_combined.csv \
        --out /scratch1/hongn/artijepa/manifest_combined_aai.csv \
        --test-spk75 8 --val-spk75 4 --test-longit 3 --val-longit 1 --seed 0
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np


def corpus_of(path):
    return "spk75" if "speaker75" in path else "longit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/scratch1/hongn/artijepa/manifest_combined.csv")
    ap.add_argument("--out", default="/scratch1/hongn/artijepa/manifest_combined_aai.csv")
    ap.add_argument("--test-spk75", type=int, default=8)
    ap.add_argument("--val-spk75", type=int, default=4)
    ap.add_argument("--test-longit", type=int, default=3)
    ap.add_argument("--val-longit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp)))
    fields = rows[0].keys()
    assert "subject" in fields and "split" in fields, f"need subject+split cols, have {list(fields)}"

    # subjects per corpus (a subject lives in exactly one corpus)
    subj_corpus = {}
    for r in rows:
        subj_corpus[r["subject"]] = corpus_of(r["path"])
    by_corpus = defaultdict(list)
    for s, c in subj_corpus.items():
        by_corpus[c].append(s)

    rng = np.random.default_rng(args.seed)
    assign = {}                                  # subject -> split
    plan = {"spk75": (args.test_spk75, args.val_spk75),
            "longit": (args.test_longit, args.val_longit)}
    for c, subs in by_corpus.items():
        subs = sorted(subs)                      # deterministic base order
        order = rng.permutation(len(subs))
        n_test, n_val = plan[c]
        for rank, idx in enumerate(order):
            s = subs[idx]
            assign[s] = "test" if rank < n_test else ("val" if rank < n_test + n_val else "train")

    out_rows = []
    for r in rows:
        r = dict(r)
        r["split"] = assign[r["subject"]]
        out_rows.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(out_rows)

    # report
    vids = Counter(r["split"] for r in out_rows)
    spk = defaultdict(set)
    for r in out_rows:
        spk[r["split"]].add(r["subject"])
    print(f"wrote {args.out}  ({len(out_rows)} rows)")
    for sp in ("train", "val", "test"):
        by_c = Counter(subj_corpus[s] for s in spk[sp])
        print(f"  {sp:5s}: {vids[sp]:5d} videos | {len(spk[sp]):3d} speakers "
              f"(spk75={by_c['spk75']}, longit={by_c['longit']})")
    # sanity: splits are speaker-disjoint
    inter = (spk["train"] & spk["val"]) | (spk["train"] & spk["test"]) | (spk["val"] & spk["test"])
    print(f"  speaker-disjoint: {'OK' if not inter else 'VIOLATION ' + str(inter)}")
    print(f"  held-out test speakers: {sorted(spk['test'])}")
    print(f"  val speakers:           {sorted(spk['val'])}")


if __name__ == "__main__":
    main()
