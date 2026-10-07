#!/usr/bin/env python3
"""Label-distribution plots for the usc_lss (test_lss) phoneme dataset.

Five bar charts of the per-temporal-token label counts (the same tokens the
attentive_lstm probe classifies -> these are the row supports behind the confusion
matrices):
  * phonemes    -- every phoneme, colored by manner (sil shown separately)
  * groups      -- 7 manner-of-articulation classes
  * places      -- 8 consonant places of articulation
  * vowels      -- 15 vocalic identities
  * consonants  -- 25 consonant identities

Counts are token/frame-level (duration-weighted), read straight from the frozen
feature-cache labels.npy (identical across encoders), so no MRI decode needed.

Usage:
  source dev_artiJEPA/scripts/_env.sh
  python dev_artiJEPA/scratchpad/dist_usc_lss.py
"""
import argparse, os
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import artijepa.tsne_phonemes as TS
import artijepa.phonemes as P

CACHE = ("/scratch1/hongn/artijepa/feat_cache/phoneme/"
         "tssl256comb215sp_43c1fe20dd/{split}.labels.npy")


def bar(ax, labels, counts, colors, title, rotate=0):
    x = np.arange(len(labels))
    tot = max(1, sum(counts))
    ax.bar(x, counts, color=colors, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotate, fontsize=8,
                                         ha="right" if rotate else "center")
    for xi, c in zip(x, counts):
        ax.text(xi, c, f"{100*c/tot:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_title(title, fontsize=11); ax.set_ylabel("token count")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_lss")
    ap.add_argument("--out", default="/scratch1/hongn/artijepa/eval/attn_lstm_viz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lab = np.load(CACHE.format(split=args.split)).reshape(-1)
    lab = lab[lab != P.IGNORE_INDEX]                          # drop padded tokens
    n_sil = int((lab == P.SIL_IDX).sum())
    ph = np.array([P.IDX2PHON[int(i)] for i in lab], dtype=object)
    cnt = Counter(ph)
    total = len(lab)
    n_phon = total - n_sil
    print(f"[dist] {args.split}: {total} tokens ({n_sil} sil = {100*n_sil/total:.1f}%, "
          f"{n_phon} phonetic)")

    # ---- 1) by phoneme (colored by manner; sil last, grey) -------------------
    phons = [p for p in P.ARPABET if p in cnt and p != "sil"]
    phons.sort(key=lambda p: -cnt[p])
    labels = phons + (["sil"] if n_sil else [])
    counts = [cnt[p] for p in phons] + ([n_sil] if n_sil else [])
    cols = [TS.CLASS_COLOR.get(TS.PHON_CLASS.get(p), "#999") for p in phons] + \
           (["#bbbbbb"] if n_sil else [])
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 4.2))
    bar(ax, labels, counts, cols, f"usc_lss ({args.split}) — phoneme distribution "
        f"(% of {total} tokens; colored by manner)", rotate=60)
    fig.tight_layout(); fp = os.path.join(args.out, f"dist_phoneme_{args.split}.png")
    fig.savefig(fp, dpi=150); plt.close(fig); print("wrote", fp)

    # ---- 2) by group (manner) ------------------------------------------------
    gcnt = Counter()
    for p, c in cnt.items():
        g = TS.PHON_CLASS.get(p)
        if g:
            gcnt[g] += c
    labels = [g for g in TS.CLASS_ORDER if gcnt[g]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, labels, [gcnt[g] for g in labels], [TS.CLASS_COLOR[g] for g in labels],
        f"usc_lss ({args.split}) — manner-group distribution (% of {n_phon} phonetic tokens)",
        rotate=30)
    fig.tight_layout(); fp = os.path.join(args.out, f"dist_group_{args.split}.png")
    fig.savefig(fp, dpi=150); plt.close(fig); print("wrote", fp)

    # ---- 3) by place (consonants) --------------------------------------------
    pcnt = Counter()
    for p, c in cnt.items():
        pl = TS.PLACE_CLASS.get(p)
        if pl:
            pcnt[pl] += c
    n_cons = sum(pcnt.values())
    labels = [pl for pl in TS.PLACE_ORDER if pcnt[pl]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar(ax, labels, [pcnt[pl] for pl in labels], [TS.PLACE_COLOR[pl] for pl in labels],
        f"usc_lss ({args.split}) — consonant PLACE distribution (% of {n_cons} consonant tokens)",
        rotate=30)
    fig.tight_layout(); fp = os.path.join(args.out, f"dist_place_{args.split}.png")
    fig.savefig(fp, dpi=150); plt.close(fig); print("wrote", fp)

    # ---- 4) by vowel identity ------------------------------------------------
    vs = [p for p in TS.VOWEL_PHON if cnt[p]]
    vs.sort(key=lambda p: -cnt[p])
    n_vow = sum(cnt[p] for p in vs)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bar(ax, vs, [cnt[p] for p in vs], [TS.VOWEL_COLOR[p] for p in vs],
        f"usc_lss ({args.split}) — vowel identity distribution (% of {n_vow} vowel tokens)",
        rotate=0)
    fig.tight_layout(); fp = os.path.join(args.out, f"dist_vowel_{args.split}.png")
    fig.savefig(fp, dpi=150); plt.close(fig); print("wrote", fp)

    # ---- 5) by consonant identity (colored by manner) ------------------------
    cs = [p for p in TS.CONS_PHON if cnt[p]]
    cs.sort(key=lambda p: -cnt[p])
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cs)), 4))
    bar(ax, cs, [cnt[p] for p in cs],
        [TS.CLASS_COLOR.get(TS.PHON_CLASS.get(p), "#999") for p in cs],
        f"usc_lss ({args.split}) — consonant identity distribution (% of {n_cons} consonant tokens; "
        f"colored by manner)", rotate=60)
    fig.tight_layout(); fp = os.path.join(args.out, f"dist_consonant_{args.split}.png")
    fig.savefig(fp, dpi=150); plt.close(fig); print("wrote", fp)

    # ---- console summary -----------------------------------------------------
    print(f"\n[dist] manner groups (% of {n_phon} phonetic tokens):")
    for g in TS.CLASS_ORDER:
        if gcnt[g]:
            print(f"   {g:<12} {gcnt[g]:>7} {100*gcnt[g]/n_phon:>6.1f}%")
    print(f"[dist] vowels {n_vow} ({100*n_vow/n_phon:.1f}%) | consonants {n_cons} "
          f"({100*n_cons/n_phon:.1f}%)")


if __name__ == "__main__":
    main()
