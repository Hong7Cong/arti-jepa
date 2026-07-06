"""Concatenate several Arti-JEPA manifests into one combined training manifest.

Used to add extra pre-training corpora (e.g. the longitudinal set built by
``build_manifest_longitudinal.py``) on top of the 75-speaker manifest without
touching the per-corpus CSVs -- the dataloader reads a single manifest, so we
materialise the union on disk and point the config at it.

Columns are reconciled to the **union** across inputs (missing cells filled
blank), preserving first-seen order. As a leakage guard it checks that no
``subject`` appears in more than one input manifest -- subject-disjointness is
the headline requirement (A.8), so an overlap here would mean the same anatomy
sits in two corpora and is flagged loudly (``--strict`` makes it fatal).

Usage:
    python -m artijepa.merge_manifests \
        --inputs /data2/hongn/artijepa/manifest_alltrain.csv \
                 /data2/hongn/artijepa/manifest_longitudinal.csv \
        --out /data2/hongn/artijepa/manifest_combined.csv
"""

import argparse
import csv
from collections import Counter, defaultdict


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def merge(inputs, out, strict=False):
    if len(inputs) < 2:
        raise SystemExit("need at least two --inputs to merge")

    fields = []                                   # union, first-seen order
    per_source = []                               # [(path, rows)]
    subj_sources = defaultdict(set)               # subject -> {source paths}
    for path in inputs:
        rows = _read(path)
        if not rows:
            print(f"  [warn] {path} has no data rows")
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
            if r.get("subject"):
                subj_sources[r["subject"]].add(path)
        per_source.append((path, rows))

    # leakage guard: a subject must live in exactly one source manifest
    overlap = {s: sorted(src) for s, src in subj_sources.items() if len(src) > 1}
    if overlap:
        msg = (f"{len(overlap)} subject(s) appear in >1 input manifest "
               f"(possible anatomy leakage): "
               + ", ".join(f"{s}={src}" for s, src in list(overlap.items())[:10]))
        if strict:
            raise SystemExit("[merge] FATAL: " + msg)
        print("[merge] WARNING: " + msg)

    all_rows = []
    for path, rows in per_source:
        for r in rows:
            all_rows.append({k: r.get(k, "") for k in fields})
        sp = Counter(r.get("split", "") for r in rows)
        print(f"  {path}: {len(rows)} rows, splits={dict(sp)}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    total_split = Counter(r.get("split", "") for r in all_rows)
    n_subj = len({r.get("subject") for r in all_rows if r.get("subject")})
    print(f"Wrote {len(all_rows)} rows ({n_subj} subjects) -> {out}")
    print(f"  combined splits: {dict(total_split)}")
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="manifest CSVs to concatenate (>= 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="abort (not just warn) if a subject spans >1 input")
    args = ap.parse_args()
    merge(args.inputs, args.out, strict=args.strict)


if __name__ == "__main__":
    main()
