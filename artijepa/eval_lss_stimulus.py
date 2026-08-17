"""Is the healthy-OOD (usc_lss) anchor depressed by UNSEEN STIMULUS rather than domain shift?

The Annot-16 probe's in-domain test (sub043) covers the **same 20 stimuli it trained on**
(vcv1-3, bVt, shibboleth, rainbow, grandfather1-2, northwind1-2, picture1-5, topic1-5),
whereas its healthy-OOD test (`test_lss` = usc_lss speaker s1, 684 utts / 71 items) is
mostly *different material*. So the in-domain -> healthy-OOD drop (κ 0.673 -> 0.412) mixes
**speaker/appearance shift** with **stimulus novelty**, and the two can be separated because
the two corpora happen to share one passage.

**Stimulus overlap (measured, not assumed).** Gold phoneme sequences (sil-stripped) of every
usc_lss item are matched against every Annot-16 stimulus. **6 of 71 items overlap**, and the
separation is unambiguous -- matched ratios 0.63-0.94 vs <=0.32 for all 65 others:

    item 39 -> grandfather1 0.921    item 56 -> grandfather1 0.943
    item 40 -> northwind1   0.930    item 57 -> northwind1   0.932
    item 38 -> rainbow      0.632    item 55 -> rainbow      0.784   (composite, see below)

All three passages are in the Annot-16 **train** split, so the probe genuinely saw this
material from its 14 training speakers.

!! `difflib.SequenceMatcher(...).ratio()` MUST be called with `autojunk=False` here. The
default autojunk heuristic discards any element occurring in >1% of a sequence of length
>=200 -- and in a ~40-symbol phoneme alphabet *every* symbol is that frequent, so matching
silently collapses. With autojunk on, northwind scored 0.372 and looked like a non-match
(it is a verbatim match); the correct ratio is 0.930.

Items 38 and 55 are **composites**: usc_lss item 55 opens with the shibboleth sentences
("she had your dark suit in greasy wash water...") and continues into other material, so its
ratio against any single Annot-16 stimulus understates the overlap. They are reported as a
separate `seen_partial` tier rather than pooled with the verbatim matches.

This script scores `test_lss` **per usc_lss item** with the same utterance-mode inference and
the same metric definitions used for the anchors, then reports:
  * `all`          -- all 684 utts (must reproduce the stored anchor κ)
  * `seen_pure`    -- ratio >= 0.90: items 39/56 grandfather + 40/57 northwind (31 utts)
  * `seen_partial` -- 0.60 <= ratio < 0.90: items 38/55, composite (28 utts)
  * `unseen`       -- the other 65 items (625 utts)
  * per-item κ/PER, so the matched items can be read against the *distribution* of unseen
    items rather than against a single pooled number (n is small; a lone number would not
    distinguish a real effect from item-level variance).

NO extraction and NO encoder: the `test_lss` features are already in the Annot-16 unpooled
cache (`feat_cache/phoneme/tssl256comb215sp_43c1fe20dd/test_lss.*`, 31 GB), so `extract`
cache-hits and the encoder argument is never touched.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_lss_stimulus \
        --encoder tssl256comb215 --head attentive_lstm --loss ce --seed 0   # + 1, 2
"""

import argparse
import collections
import copy
import difflib
import json
import os
import re

import numpy as np
import torch

import artijepa.phonemes as P
from artijepa.eval_phoneme import (
    load_config, extract, build_dataset, load_probe, _UttSpatialDS, _utt_loader)
from artijepa.eval_gloss import pick_probe, _utt_infer, _utt_metrics
from artijepa.confmat_phonemes import build_matrices, macro_recall

DEFAULT_CFG = "dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml"
SPLIT = "test_lss"
MATCH_MIN = 0.60          # ratio above which a usc_lss item shares stimulus material
PURE_MIN = 0.90           # ... and above which it is a verbatim match (not a composite)


# --------------------------------------------------------------------------- #
# stimulus matching (phoneme-sequence identity, no transcripts needed)
# --------------------------------------------------------------------------- #
def _seq(pj):
    return [p for p, _, _ in P.load_gold_segments(pj) if p != "sil"]


def stimulus_overlap(manifest, match_min=MATCH_MIN):
    """-> (matched {item: (annot16_stimulus, ratio)}, all_ratios {item: (name, ratio)}).

    Annot-16 reference = the longest exemplar of each stimulus. usc_lss item = its subparts
    concatenated in order (the corpus chunks each item into `_0.._N`)."""
    import csv
    rows = list(csv.DictReader(open(manifest)))
    ann = collections.defaultdict(list)
    for r in rows:
        if r["split"] in ("train", "val", "test"):
            m = re.match(r"sub\d+_2drt_(?:\d+_)?(.+?)(?:_r\d+)?$", r["utt_id"])
            if m:
                ann[m.group(1)].append(_seq(r["phoneme_json"]))
    ref = {k: max(v, key=len) for k, v in ann.items()}

    parts = collections.defaultdict(list)
    for r in rows:
        if r["split"] == SPLIT:
            _, _, it, pt = r["utt_id"].split("_")
            parts[int(it)].append((int(pt), r["phoneme_json"]))
    best = {}
    for it, ps in parts.items():
        ps.sort()
        cat = [p for _, pj in ps for p in _seq(pj)]
        # autojunk=False is REQUIRED: see the module docstring. With it on, every phoneme
        # counts as "popular junk" and verbatim passages score like unrelated ones.
        name, ratio = max(
            ((k, difflib.SequenceMatcher(None, cat, v, autojunk=False).ratio())
             for k, v in ref.items()), key=lambda kv: kv[1])
        best[it] = (name, round(ratio, 4))
    matched = {it: v for it, v in best.items() if v[1] >= match_min}
    return matched, best


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--encoder", default="tssl256comb215")
    ap.add_argument("--head", default="attentive_lstm")
    ap.add_argument("--loss", default="ce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-set", default="annot16")
    ap.add_argument("--arti-out", default="/scratch1/hongn/artijepa")
    args = ap.parse_args()

    eval_dir = os.path.join(args.arti_out, "eval")
    out_dir = os.path.join(eval_dir, "gloss"); os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    pt = pick_probe(args.encoder, args.seed, eval_dir, args.train_set,
                    head=args.head, loss=args.loss)
    probe, ck = load_probe(pt, device=device)
    stored = ((ck.get("metrics", {}).get("tests") or {}).get(SPLIT) or {})
    print(f"[lss-stim] probe {os.path.basename(pt)}")
    print(f"[lss-stim] stored {SPLIT} anchor: κ={stored.get('kappa')} "
          f"PERµ={stored.get('per_micro')} n_utt={stored.get('n_utt')}")

    cfg["encoder"]["spec"] = ck["encoder_spec"]
    cfg["data"]["pool_spatial"] = False          # spatial head -> un-pooled token grid
    cfg["meta"]["tag"] = args.encoder            # -> <enc>sp_<hash>, the existing cache

    # cache-hit: encoder is never used (pass None so a miss fails loudly instead of
    # silently re-extracting 31 GB against a rebuilt-in-place manifest)
    feats, labels, meta = extract(None, cfg, SPLIT, device, torch.float16)
    ds = build_dataset(cfg, SPLIT)[0]
    refs = ds.reference_sequences()
    num_classes, drop = ds.num_classes, tuple(ds.collapse_drop)

    uds = _UttSpatialDS(feats, labels, meta, refs)
    mt = cfg["probe"].get("utt_max_tokens", 1024)
    workers = cfg["probe"].get("utt_eval_workers", 2)
    preds, labs = _utt_infer(probe, uds, device, mt, workers, device.type == "cuda")
    utts = sorted(preds)

    item_of = {u: int(ds.rows[u]["utt_id"].split("_")[2]) for u in utts}
    matched, all_ratios = stimulus_overlap(cfg["data"]["manifest"])
    seen_items = sorted(matched)
    print(f"[lss-stim] stimulus overlap (ratio >= {MATCH_MIN}): "
          + ", ".join(f"item {i} -> {matched[i][0]} ({matched[i][1]})" for i in seen_items))

    def subset(us, label):
        m = _utt_metrics(preds, labs, refs, num_classes, drop, us)
        t = np.concatenate([labs[u] for u in us]); p = np.concatenate([preds[u] for u in us])
        k = t != P.IGNORE_INDEX
        m["macro_recall"] = {kk: round(macro_recall(v[0]), 4)
                             for kk, v in build_matrices(t[k], p[k]).items()}
        m["label"] = label
        return m

    pure_items = sorted(i for i in matched if matched[i][1] >= PURE_MIN)
    part_items = sorted(i for i in matched if matched[i][1] < PURE_MIN)
    tier = {}
    for u in utts:
        it = item_of[u]
        tier[u] = ("seen_pure" if it in pure_items else
                   "seen_partial" if it in part_items else "unseen")
    res = {"all": subset(utts, "all 684 utts (reproduces the stored anchor)")}
    for name, desc in (("seen_pure", f"verbatim-match items {pure_items}"),
                       ("seen_partial", f"composite items {part_items}"),
                       ("unseen", f"the other {71 - len(matched)} items")):
        us = [u for u in utts if tier[u] == name]
        if us:
            res[name] = subset(us, desc)

    per_item = {}
    for it in sorted(set(item_of.values())):
        us = [u for u in utts if item_of[u] == it]
        if us:
            per_item[it] = _utt_metrics(preds, labs, refs, num_classes, drop, us)
            per_item[it]["best_annot16_match"] = all_ratios[it]

    print(f"\n===== healthy-OOD (usc_lss) by stimulus novelty — {args.encoder} "
          f"{args.head} s{args.seed} =====")
    print(f"{'subset':<16}{'κ':>9}{'PERµ':>9}{'PER_M':>9}{'acc':>8}{'n_utt':>7}")
    for k in ("all", "seen_pure", "seen_partial", "unseen"):
        if k not in res:
            continue
        m = res[k]
        print(f"{k:<16}{m['kappa']:>9.4f}{m['per_micro']:>9.3f}{m['per_macro']:>9.3f}"
              f"{m['frame_acc']:>8.3f}{m['n_utt']:>7}")

    # read each matched item against the DISTRIBUTION of unseen items, per metric
    for metric in ("kappa", "per_micro"):
        vals = {it: per_item[it][metric] for it in per_item}
        un = [v for it, v in vals.items() if it not in matched]
        mu, sd = float(np.mean(un)), float(np.std(un, ddof=1))
        print(f"\nper-item {metric} over the {len(un)} UNSEEN items: "
              f"mean={mu:.4f} sd={sd:.4f} min={min(un):.4f} max={max(un):.4f}")
        for it in sorted(matched):
            z = (vals[it] - mu) / sd
            pct = 100.0 * sum(v < vals[it] for v in un) / len(un)
            t = "pure" if matched[it][1] >= PURE_MIN else "part"
            print(f"  item {it:>2} {matched[it][0]:<13}({t}) {metric}={vals[it]:.4f}  "
                  f"z={z:+.2f}  pct={pct:5.1f}%  n_utt={per_item[it]['n_utt']}")

    jp = os.path.join(out_dir, f"lss_stimulus_{args.encoder}_{args.head}_s{args.seed}.json")
    json.dump({"probe": os.path.basename(pt), "seed": args.seed,
               "stored_anchor": stored, "match_min": MATCH_MIN,
               "matched_items": {str(k): v for k, v in matched.items()},
               "subsets": res, "per_item": {str(k): v for k, v in per_item.items()},
               "all_ratios": {str(k): v for k, v in all_ratios.items()}},
              open(jp, "w"), indent=2)
    print(f"\n[lss-stim] wrote {jp}")


if __name__ == "__main__":
    main()
