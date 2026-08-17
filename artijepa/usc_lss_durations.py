"""Segment-duration statistics of the usc_lss gold phoneme annotations.

Reads the hand-aligned ARPABET segments straight from `phoneme_manifest.csv` ->
per-utt phoneme JSON -- NO encoder / GPU, no video decode. Each gold segment is one
observation with duration `end - start` (seconds, as annotated); we report mean /
median / sd / p5 / p95 / total per phoneme and per manner class, plus the pooled
VOWEL (Vowel+Diphthong) and CONSONANT (Plosive+Fricative+Affricate+Nasal+Approximant)
aggregates. Classes come from `tsne_phonemes.PHON_CLASS`, so the vowel/consonant split
matches the phoneme-groups probe (`eval_phoneme_groups.py`) exactly.

Outputs (to eval/usc_lss/):
  - usc_lss_durations.csv     (long form: level,name,class,n,mean_ms,median_ms,sd_ms,p5_ms,p95_ms,total_s)
  - usc_lss_durations.json    (nested summary: pooled / by_class / by_phoneme)
  and a formatted table to stdout.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.usc_lss_durations
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np

import artijepa.phonemes as P
from artijepa.tsne_phonemes import PHON_CLASS, CLASS_ORDER

USC_ROOT = "/scratch1/hongn/usc_lss"
MANIFEST = os.path.join(USC_ROOT, "phoneme_manifest.csv")
OUT_DIR = "/scratch1/hongn/artijepa/eval/usc_lss"

VOWEL_CLASSES = {"Vowel", "Diphthong"}
CONS_CLASSES = {"Plosive", "Fricative", "Affricate", "Nasal", "Approximant"}


def collect(manifest=MANIFEST):
    """-> ({phoneme: [dur_s, ...]}, n_utt_read, n_utt_total)."""
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    per_phon = defaultdict(list)
    n_read = 0
    for r in rows:
        try:
            with open(r["phoneme_json"]) as f:
                segs = json.load(f)
        except FileNotFoundError:
            continue
        n_read += 1
        for s in segs:
            per_phon[s["phoneme"].lower()].append(float(s["end"]) - float(s["start"]))
    return per_phon, n_read, len(rows)


def stats(durs):
    """[dur_s] -> summary dict (ms for central tendency, s for total)."""
    d = np.asarray(durs, dtype=np.float64)
    return dict(n=int(d.size), mean_ms=float(d.mean() * 1e3),
                median_ms=float(np.median(d) * 1e3), sd_ms=float(d.std() * 1e3),
                p5_ms=float(np.percentile(d, 5) * 1e3),
                p95_ms=float(np.percentile(d, 95) * 1e3), total_s=float(d.sum()))


def _pool(per_phon, classes):
    return [x for p, ds in per_phon.items() if PHON_CLASS.get(p) in classes for x in ds]


def summarize(per_phon):
    """-> {'pooled': {...}, 'by_class': {...}, 'by_phoneme': {...}} of stats dicts."""
    pooled = {
        "VOWELS": _pool(per_phon, VOWEL_CLASSES),
        "CONSONANTS": _pool(per_phon, CONS_CLASSES),
        "SILENCE": _pool(per_phon, {"Silence"}),
    }
    by_class = defaultdict(list)
    for p, ds in per_phon.items():
        by_class[PHON_CLASS.get(p, "UNK")].extend(ds)
    return dict(
        pooled={k: stats(v) for k, v in pooled.items() if v},
        by_class={k: stats(v) for k, v in by_class.items() if v},
        by_phoneme={p: stats(ds) for p, ds in per_phon.items()},
    )


def _line(name, s):
    return (f"{name:16s} n={s['n']:6d}  mean={s['mean_ms']:7.1f} ms  "
            f"median={s['median_ms']:7.1f}  sd={s['sd_ms']:6.1f}  "
            f"p5={s['p5_ms']:6.1f}  p95={s['p95_ms']:7.1f}  total={s['total_s']:7.1f} s")


def main():
    per_phon, n_read, n_tot = collect()
    summ = summarize(per_phon)
    n_seg = sum(len(v) for v in per_phon.values())

    print(f"usc_lss: {n_read}/{n_tot} utterances read, {n_seg} gold segments\n")
    print("===== pooled =====")
    for k in ["VOWELS", "CONSONANTS", "SILENCE"]:
        if k in summ["pooled"]:
            print(_line(k, summ["pooled"][k]))
    print("\n===== by manner class =====")
    for k in list(CLASS_ORDER) + [c for c in summ["by_class"] if c not in CLASS_ORDER]:
        if k in summ["by_class"]:
            print(_line(k, summ["by_class"][k]))
    print("\n===== per phoneme (desc mean) =====")
    for p in sorted(summ["by_phoneme"], key=lambda p: -summ["by_phoneme"][p]["mean_ms"]):
        print(_line(f"{p} [{PHON_CLASS.get(p, '?')[:4]}]", summ["by_phoneme"][p]))

    unknown = sorted(p for p in per_phon if p not in P.PHON2IDX)
    if unknown:
        print(f"\n[warn] symbols outside ARPABET inventory: {unknown}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "usc_lss_durations.json"), "w") as f:
        json.dump(dict(n_utt=n_read, n_seg=n_seg, **summ), f, indent=2)
    cols = ["level", "name", "class", "n", "mean_ms", "median_ms", "sd_ms",
            "p5_ms", "p95_ms", "total_s"]
    with open(os.path.join(OUT_DIR, "usc_lss_durations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for level, key in [("pooled", "pooled"), ("class", "by_class"),
                           ("phoneme", "by_phoneme")]:
            for name, s in summ[key].items():
                w.writerow(dict(level=level, name=name,
                                **{"class": PHON_CLASS.get(name, "")}, **s))
    print(f"\nwrote {OUT_DIR}/usc_lss_durations.{{csv,json}}")


if __name__ == "__main__":
    main()
