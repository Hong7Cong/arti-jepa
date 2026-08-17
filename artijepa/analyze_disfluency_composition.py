#!/usr/bin/env python3
"""What is a disfluency event made of, and why is it silent? Per-type composition.

Follows up analyze_binary_phoneme_confound: that showed disfluent windows are ~52%
silence and silence alone reproduces most of the binary score. This asks WHY --
by breaking silence/vowel/consonant occupancy down by disfluency TYPE, on the same
rows (build_rows seed 0), phones-tier files only.

Expectation from the mechanism:
  block  -- silent articulatory HOLD (+ pauses between repeated attempts) -> high silence
  pro    -- sustained SOUND (a held phone) -> LOW silence
  rep    -- repeated gesture cycles with inter-attempt pauses -> medium

If blocks are high-silence and prolongations are low-silence, the silence is the
genuine dysfluent behavior (held postures / pauses), not a forced-alignment artifact
dumping speech into <sil>.

RUN:  python -m artijepa.analyze_disfluency_composition
"""
import json
import os
from collections import defaultdict

import numpy as np

from artijepa import stutter as S
from artijepa import stutter_binary as SB
from artijepa.stutter import FLUENT
from artijepa.analyze_binary_phoneme_confound import (
    window_phones, phone_class, CLASSES, CONSONANT)

OUT_JSON = "/scratch1/hongn/artijepa/eval/stutter_type/disfluency_composition.json"


def main():
    rows, _ = SB.build_rows(seed=0, neg_per_pos=1.0, min_dur=0.20, max_dur=8.0,
                            merge_gap=0.25, verbose=False)
    phone_cache = {}
    # per group: list of (silence_frac, vowel_frac, cons_frac, dur)
    by = defaultdict(list)

    for r in rows:
        spk, stem = r["speaker"], r["stem"]
        grp = FLUENT if r["bucket5"] == FLUENT else r["primary"]
        key = (spk, stem)
        if key not in phone_cache:
            tg = os.path.join(SB.ROOT, spk, "textgrid", stem + ".TextGrid")
            parsed = S.parse_textgrid(tg) if os.path.exists(tg) else {}
            phone_cache[key] = parsed.get("phones", [])
        phones = phone_cache[key]
        if not phones:
            continue
        occ, _ = window_phones(phones, r["xmin"], r["xmax"])
        tot = sum(occ.values())
        if tot <= 0:
            continue
        sil = occ.get("silence", 0.0) / tot
        vow = occ.get("vowel", 0.0) / tot
        con = sum(occ.get(c, 0.0) for c in CONSONANT) / tot
        by[grp].append((sil, vow, con, r["dur"]))

    order = ["block", "rep", "pro", FLUENT]
    order += [g for g in by if g not in order]
    print(f"{'group':<10}{'n':>6}{'dur_s':>8}{'silence%':>10}{'vowel%':>9}{'conson%':>9}")
    out = {}
    for g in order:
        if g not in by:
            continue
        arr = np.array(by[g])
        n = len(arr)
        sil, vow, con, dur = arr.mean(0)
        print(f"{g:<10}{n:>6}{dur:>8.2f}{sil*100:>9.1f}%{vow*100:>8.1f}%{con*100:>8.1f}%")
        out[g] = dict(n=int(n), mean_dur=float(dur), silence=float(sil),
                      vowel=float(vow), consonant=float(con),
                      silence_p50=float(np.median(arr[:, 0])))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_JSON}")
    print("\nReading: block silence >> pro silence confirms the silence is the held/"
          "paused disfluent behavior itself, not an alignment artifact (a prolongation "
          "is sustained sound, so it stays low-silence).")


if __name__ == "__main__":
    main()
