#!/usr/bin/env python3
"""Is the stutter BINARY probe's 0.83 a phoneme-distribution confound?

Hypothesis (user): disfluent windows may be phonetically distinct from the
duration-matched fluent negatives -- e.g. concentrated on stop consonants, the
classic stutter-prone class -- so a video encoder could score high by reading
articulatory posture / phoneme identity rather than disfluency *dynamics*.

We test it three ways on the EXACT rows the probe used (build_rows seed 0,
neg_per_pos 1.0, min/max 0.20/8.0, merge_gap 0.25 == cache tag b5da470386):

  1. Phone-class occupancy: fraction of each window's duration on vowel / stop /
     fricative / nasal / liquid / glide / affricate / silence, pos vs neg.
  2. Onset phone: the first non-silence phone at/after xmin (the "stuttered
     phoneme") -- class distribution pos vs neg, and a stop-consonant enrichment.
  3. THE DECISIVE TEST -- a phoneme-histogram-ONLY logistic regression, LOSO, no
     video at all. If it reaches ~0.83 the encoder could be riding the confound;
     if it sits near chance the video is doing real work the phonemes can't.

Only files with a `phones` tier (343/476) contribute; coverage per speaker is
printed so the reader sees what fraction of rows the phone analysis actually saw.

RUN:  python -m artijepa.analyze_binary_phoneme_confound
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

from artijepa import stutter as S
from artijepa import stutter_binary as SB
from artijepa.stutter import FLUENT

OUT_JSON = "/scratch1/hongn/artijepa/eval/stutter_binary/phoneme_confound.json"

# ARPABET (stress digits stripped) -> broad manner class
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
          "OW", "OY", "UH", "UW"}
STOPS = {"B", "D", "G", "P", "T", "K"}          # plosives -- the classic stutter class
AFFRIC = {"CH", "JH"}
FRIC = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
NASAL = {"M", "N", "NG"}
LIQUID = {"L", "R"}
GLIDE = {"W", "Y"}


def phone_class(lab):
    if lab == "" or lab is None:
        return "silence"
    base = lab.rstrip("0123456789")
    if base in VOWELS:  return "vowel"
    if base in STOPS:   return "stop"
    if base in AFFRIC:  return "affricate"
    if base in FRIC:    return "fricative"
    if base in NASAL:   return "nasal"
    if base in LIQUID:  return "liquid"
    if base in GLIDE:   return "glide"
    return "other"


CLASSES = ["vowel", "stop", "affricate", "fricative", "nasal", "liquid", "glide",
           "silence"]
CONSONANT = {"stop", "affricate", "fricative", "nasal", "liquid", "glide"}


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_phones(phones, x0, x1):
    """Time-in-class (s) over [x0,x1] and the onset phone base (first non-sil)."""
    occ = defaultdict(float)
    onset = None
    for p0, p1, lab in phones:
        ov = _overlap(x0, x1, p0, p1)
        if ov <= 0:
            continue
        occ[phone_class(lab)] += ov
        if onset is None and lab.strip():
            onset = lab.rstrip("0123456789")
    return occ, onset


def main():
    rows, stats = SB.build_rows(seed=0, neg_per_pos=1.0, min_dur=0.20, max_dur=8.0,
                                merge_gap=0.25, verbose=False)
    # phones per (speaker, stem), parsed once
    phone_cache = {}
    cov = defaultdict(lambda: [0, 0])   # speaker -> [rows_with_phones, rows_total]

    feats, labels, spks = [], [], []
    occ_pos = defaultdict(float); occ_neg = defaultdict(float)
    onset_pos = Counter(); onset_neg = Counter()
    n_pos_cov = n_neg_cov = 0

    for r in rows:
        spk, stem = r["speaker"], r["stem"]
        is_pos = r["bucket5"] != FLUENT
        cov[spk][1] += 1
        key = (spk, stem)
        if key not in phone_cache:
            tg = os.path.join(SB.ROOT, spk, "textgrid", stem + ".TextGrid")
            parsed = S.parse_textgrid(tg) if os.path.exists(tg) else {}
            phone_cache[key] = parsed.get("phones", [])
        phones = phone_cache[key]
        if not phones:
            continue
        occ, onset = window_phones(phones, r["xmin"], r["xmax"])
        if sum(occ.values()) <= 0:
            continue
        cov[spk][0] += 1
        tot = sum(occ.values())
        hist = np.array([occ.get(c, 0.0) / tot for c in CLASSES])   # class fractions
        feats.append(hist); labels.append(1 if is_pos else 0); spks.append(spk)
        tgt = (occ_pos if is_pos else occ_neg)
        for c in CLASSES:
            tgt[c] += occ.get(c, 0.0)
        if onset:
            (onset_pos if is_pos else onset_neg)[phone_class(onset)] += 1
        if is_pos: n_pos_cov += 1
        else:      n_neg_cov += 1

    X = np.array(feats); y = np.array(labels); sp = np.array(spks)
    print(f"rows total={len(rows)}  with phone coverage: pos={n_pos_cov} neg={n_neg_cov}")
    print("phones-tier row coverage per speaker (covered/total):")
    for s in sorted(cov):
        c, t = cov[s]
        print(f"   {s}: {c}/{t}  ({100*c/max(t,1):.0f}%)")

    # ---- 1. phone-class occupancy (time-weighted), pos vs neg -------------- #
    def norm(d):
        tot = sum(d.values()) or 1.0
        return {c: d.get(c, 0.0) / tot for c in CLASSES}
    fp, fn = norm(occ_pos), norm(occ_neg)
    print("\n=== phone-class occupancy (fraction of window-time) ===")
    print(f"{'class':<10}{'disfluent':>11}{'fluent':>10}{'ratio p/f':>11}")
    for c in CLASSES:
        rat = fp[c] / fn[c] if fn[c] > 1e-9 else float("inf")
        print(f"{c:<10}{fp[c]*100:>10.1f}%{fn[c]*100:>9.1f}%{rat:>11.2f}")
    cons_p = sum(fp[c] for c in CONSONANT); cons_n = sum(fn[c] for c in CONSONANT)
    print(f"{'CONSONANT':<10}{cons_p*100:>10.1f}%{cons_n*100:>9.1f}%"
          f"{cons_p/max(cons_n,1e-9):>11.2f}")

    # ---- 2. onset (stuttered) phone class --------------------------------- #
    def npct(cnt):
        tot = sum(cnt.values()) or 1
        return {c: 100 * cnt.get(c, 0) / tot for c in CLASSES}
    op, on = npct(onset_pos), npct(onset_neg)
    print("\n=== onset-phone class (first phone in the window) ===")
    print(f"{'class':<10}{'disfluent':>11}{'fluent':>10}")
    for c in CLASSES:
        print(f"{c:<10}{op[c]:>10.1f}%{on[c]:>9.1f}%")

    # ---- 3. phoneme-histogram-only logistic regression, LOSO -------------- #
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    def loso_lr(Xf):
        preds = np.zeros(len(y)); hard = np.zeros(len(y))
        for s in np.unique(sp):
            tr, te = sp != s, sp == s
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xf[tr], y[tr])
            preds[te] = clf.predict_proba(Xf[te])[:, 1]
            hard[te] = clf.predict(Xf[te])
        return f1_score(y, hard, average="macro"), roc_auc_score(y, preds)

    sil_idx = CLASSES.index("silence")
    spoken = [i for i in range(len(CLASSES)) if i != sil_idx]
    # renormalize the spoken-phone fractions to sum to 1 (removes the silence axis)
    Xsp = X[:, spoken].copy()
    Xsp = Xsp / np.clip(Xsp.sum(1, keepdims=True), 1e-9, None)

    macro_all, auc_all = loso_lr(X)                     # full 8-class histogram
    macro_sil, auc_sil = loso_lr(X[:, [sil_idx]])       # silence fraction ALONE
    macro_nosil, auc_nosil = loso_lr(Xsp)              # phone identity, silence removed

    print("\n=== phoneme/silence-histogram-only classifiers (no video), LOSO ===")
    print(f"   full 8-class histogram : macro-F1 {macro_all:.3f}  AUC {auc_all:.3f}")
    print(f"   SILENCE fraction alone : macro-F1 {macro_sil:.3f}  AUC {auc_sil:.3f}")
    print(f"   phone identity, no sil : macro-F1 {macro_nosil:.3f}  AUC {auc_nosil:.3f}")
    print(f"   video encoder (same rows/protocol)   macro-F1 0.830")
    print("   -> attributes the leakage: silence-in-a-speech-window vs phoneme identity.")
    macro, auc = macro_all, auc_all

    out = dict(
        n_rows=len(rows), n_pos_cov=int(n_pos_cov), n_neg_cov=int(n_neg_cov),
        coverage={s: cov[s] for s in cov},
        occupancy_pos=fp, occupancy_neg=fn,
        consonant_frac=dict(disfluent=cons_p, fluent=cons_n),
        onset_pos=op, onset_neg=on,
        phoneme_only_macro_f1=float(macro), phoneme_only_auc=float(auc),
        silence_only_macro_f1=float(macro_sil), silence_only_auc=float(auc_sil),
        nosilence_macro_f1=float(macro_nosil), nosilence_auc=float(auc_nosil),
        video_macro_f1=0.830)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
