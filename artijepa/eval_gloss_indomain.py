"""Glossectomy Test 2 — the in-domain phoneme-decodability ceiling (see docs/GLOSS.md).

The transfer probe (usc_lss-trained) drops from κ≈0.56 (usc_lss) to κ≈0.19 (gloss).
That single number fuses acquisition-domain shift + speaker-OOD + pathology. Test 2
answers the one cut that IS identifiable from data already on disk:

    (A) the phoneme signal is physically GONE from the gloss images (articulator limit)
        -- an in-domain probe trained AND tested on gloss also stays near the floor; vs
    (B) the signal is PRESENT but the usc_lss probe just doesn't transfer (different
        articulation / OOD) -- a fresh in-domain gloss probe recovers a high ceiling.

Why not the cheaper checks (documented in docs/GLOSS.md §4): raw-token separability
(t-SNE rep-B) is weak on usc_lss too, and rep-A is entangled with usc_lss-trained
queries. Only a probe RETRAINED in-domain removes that entanglement.

Design (fair by construction):
  * SEGMENT-pooled features -- for each contiguous run of equal per-token phoneme
    labels (one gold phone; ~5-7 tokens at 40 ms/token) mean-pool the frozen
    spatial-mean-pooled token vectors -> one [D] vector per phone instance. This
    matches "phoneme = 200-300 ms" and kills the 40 ms per-token noise.
  * MATCHED capacity -- a plain linear head (logreg) / LDA on those features; the
    same head is used for the usc_lss ceiling so the comparison is apples-to-apples.
  * NO leakage CV -- GroupKFold by utterance (primary, "LOUO ceiling"); Leave-One-
    Speaker-Out for gloss (conservative). usc_lss is one speaker -> LOUO only.
  * NO re-extraction -- reads the frozen caches eval_gloss / eval_phoneme already wrote.

Metric = frame/segment-level Cohen's κ (same definition as the transfer eval, sil
kept as a class) + accuracy + macro-recall (groups/vowels/consonants).

Usage:
    source dev_artiJEPA/scripts/_env.sh
    python -m artijepa.eval_gloss_indomain --encoder tssl256comb100
    python -m artijepa.eval_gloss_indomain --encoder tssl256comb100 --classifier lda --pca 128
"""

import argparse
import copy
import json
import os

import numpy as np

import artijepa.phonemes as P
from artijepa.eval_phoneme import load_config, load_probe, build_dataset, _tag
from artijepa.eval_gloss import _cond_manifest, GLOSS_ROOT, CONDS
from artijepa.tsne_phonemes import find_probe
from artijepa.confmat_phonemes import build_matrices, macro_recall


# --------------------------------------------------------------------------- #
# cache loading (no re-extraction) + segment pooling
# --------------------------------------------------------------------------- #
def _load_cache(cache_dir, split):
    """-> (feats mmap [N,T',S',D] or [N,T',D], labels [N,T'], meta [N,3])."""
    fp, lp, mp = (os.path.join(cache_dir, f"{split}.{x}.npy")
                  for x in ("feats", "labels", "meta"))
    for x in (fp, lp, mp):
        if not os.path.exists(x):
            raise SystemExit(
                f"[test2] missing cache {x}\n"
                "        run the transfer eval first (python -m artijepa.eval_gloss "
                "... for gloss; the usc_lss probe run for usc_lss).")
    return np.load(fp, mmap_mode="r"), np.load(lp), np.load(mp)


def segment_pool(feats, labels, meta, speaker_of, cond):
    """Reassemble per-utterance token streams (chunk order) and mean-pool the frozen
    features over each contiguous run of equal phoneme label -> one vector per phone.

    Returns (X [M,D] f32, y [M] i64 phoneme-idx, groups [M] utterance-id str,
             spk [M] speaker str). Spatial tokens (if present) are mean-pooled first.
    """
    # rows per utterance, in chunk order
    by_utt = {}
    for n, (utt, chunk, _) in enumerate(meta):
        by_utt.setdefault(int(utt), []).append((int(chunk), n))
    X, y, groups, spk = [], [], [], []
    for utt, rows in by_utt.items():
        rows.sort(key=lambda z: z[0])
        # concat feats/labels across chunks; spatial-pool [T',S',D]->[T',D] if 4-D
        fseq, lseq = [], []
        for _, n in rows:
            f = np.asarray(feats[n], dtype=np.float32)
            if f.ndim == 3:                      # [T',S',D] -> mean over spatial S'
                f = f.mean(1)
            fseq.append(f); lseq.append(labels[n])
        fseq = np.concatenate(fseq, 0)           # [T'*n_chunks, D]
        lseq = np.concatenate(lseq, 0)           # [T'*n_chunks]
        # contiguous runs of equal, non-IGNORE label -> one pooled segment each
        i, T = 0, len(lseq)
        gid = f"{cond}:{utt}"
        sp = speaker_of(utt)
        while i < T:
            lab = int(lseq[i]); j = i + 1
            while j < T and int(lseq[j]) == lab:
                j += 1
            if lab != P.IGNORE_INDEX:
                X.append(fseq[i:j].mean(0)); y.append(lab)
                groups.append(gid); spk.append(sp)
            i = j
    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(groups), np.asarray(spk))


# --------------------------------------------------------------------------- #
# cross-validated in-domain probe
# --------------------------------------------------------------------------- #
def _make_clf(kind, pca, seed):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [("sc", StandardScaler())]
    if pca and pca > 0:
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    if kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        steps.append(("clf", LinearDiscriminantAnalysis()))
    else:
        from sklearn.linear_model import LogisticRegression
        # multinomial is the default/only multiclass mode in recent sklearn
        steps.append(("clf", LogisticRegression(max_iter=2000, C=1.0)))
    return Pipeline(steps)


def cv_predict(X, y, groups, splitter, kind, pca, seed):
    """Out-of-fold predictions over all segments (no utterance/speaker leakage)."""
    pred = np.full(len(y), -1, np.int64)
    n_splits = 0
    for tr, te in splitter.split(X, y, groups):
        clf = _make_clf(kind, pca, seed)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        n_splits += 1
    return pred, n_splits


def score(y_true, y_pred):
    kappa = P.cohen_kappa(y_true, y_pred, P.NUM_PHONEMES)
    acc = float((y_true == y_pred).mean())
    mats = build_matrices(y_true, y_pred)
    mr = {k: macro_recall(mats[k][0]) for k in ("groups", "vowels", "consonants")}
    return {"kappa": round(float(kappa), 4), "acc": round(acc, 4),
            "macro_recall": {k: round(v, 4) for k, v in mr.items()},
            "n_seg": int(len(y_true))}


# --------------------------------------------------------------------------- #
def _gloss_segments(cfg, encoder, speaker, scratch, cache_root):
    """Segment-pool gloss pre+post from their on-disk caches -> stacked arrays."""
    parts = []
    for cond in CONDS:
        manifest, n = _cond_manifest(cond, speaker, scratch)
        c2 = copy.deepcopy(cfg)
        c2["data"]["manifest"] = manifest
        c2["data"]["pool_spatial"] = False
        c2["meta"]["tag"] = f"gloss_{cond}" + (f"_{speaker}" if speaker else "") + \
            f"_{encoder}"
        name, split = _tag(c2, "test")
        cache_dir = os.path.join(cache_root, name)
        ds = build_dataset(c2, "test")[0]
        speaker_of = lambda utt, ds=ds: ds.rows[int(utt)].get("speaker", "gloss")
        feats, labels, meta = _load_cache(cache_dir, split)
        parts.append(segment_pool(feats, labels, meta, speaker_of, cond))
        print(f"[test2] gloss {cond}: {n} utts -> {len(parts[-1][1])} phone segments "
              f"<- {os.path.basename(cache_dir)}")
    return [np.concatenate([p[i] for p in parts], 0) for i in range(4)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="tssl256comb100")
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_gloss.yaml")
    ap.add_argument("--seed", type=int, default=0, help="probe seed (cache/probe)")
    ap.add_argument("--speaker", default=None,
                    help="restrict gloss to spk1|spk2|spk3 (default pooled)")
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "lda"])
    ap.add_argument("--pca", type=int, default=0, help="PCA dims before the head (0=off)")
    ap.add_argument("--folds", type=int, default=5, help="GroupKFold splits (LOUO ceiling)")
    ap.add_argument("--arti-out", default="/scratch1/hongn/artijepa")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCHPAD", "/tmp"))
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    eval_dir = os.path.join(args.arti_out, "eval")
    out_dir = os.path.join(eval_dir, "gloss"); os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(args.config)
    cache_root = cfg["meta"]["cache_dir"]

    # locate the transfer probe -> its encoder_spec/feature_tag + reference numbers
    pt = find_probe(args.encoder, args.seed, eval_dir)
    _, ck = load_probe(pt, device="cpu")
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    usc_kappa = (ck.get("metrics", {}).get("test", {}) or {}).get("kappa")
    print(f"[test2] probe {os.path.basename(pt)}  usc_lss transfer/headline κ={usc_kappa}")

    # ---- usc_lss in-domain ceiling (test split, LOUO) ----------------------
    usc_dir = os.path.join(cache_root, ck["feature_tag"])
    uf, ul, um = _load_cache(usc_dir, "test")
    uX, uy, ug, _ = segment_pool(uf, ul, um, lambda utt: "usc_s1", "usc")
    print(f"[test2] usc_lss test: {len(set(ug))} utts -> {len(uy)} phone segments "
          f"<- {ck['feature_tag']}")
    up, uk = cv_predict(uX, uy, ug, GroupKFold(min(args.folds, len(set(ug)))),
                        args.classifier, args.pca, args.seed)
    usc = score(uy, up); usc["cv"] = f"GroupKFold-{uk}(utt)"

    # ---- gloss in-domain ceiling -------------------------------------------
    gX, gy, gg, gspk = _gloss_segments(cfg, args.encoder, args.speaker,
                                       args.scratch, cache_root)
    gp, gk = cv_predict(gX, gy, gg, GroupKFold(min(args.folds, len(set(gg)))),
                        args.classifier, args.pca, args.seed)
    gloss_louo = score(gy, gp); gloss_louo["cv"] = f"GroupKFold-{gk}(utt)"

    result = {"encoder": args.encoder, "seed": args.seed,
              "speaker": args.speaker or "pooled", "classifier": args.classifier,
              "pca": args.pca, "usc_lss_transfer_kappa": usc_kappa,
              "usc_lss_indomain": usc, "gloss_indomain_louo": gloss_louo}

    # gloss Leave-One-Speaker-Out (only meaningful pooled across speakers)
    if not args.speaker and len(set(gspk)) > 1:
        sp, sk = cv_predict(gX, gy, gspk, LeaveOneGroupOut(),
                            args.classifier, args.pca, args.seed)
        result["gloss_indomain_loso"] = score(gy, sp)
        result["gloss_indomain_loso"]["cv"] = f"LOSO-{sk}({sorted(set(gspk))})"

    # ---- verdict + report --------------------------------------------------
    # A-vs-B is decided on the SAME scale: gloss in-domain LOUO vs the usc_lss
    # in-domain ceiling measured with the identical segment/linear/CV pipeline.
    # (The κ≈0.19 transfer floor is a token-level frame κ from eval_gloss -- a
    # DIFFERENT scale -- so it is reported only as context, never in the ratio.)
    ceil = usc["kappa"]
    g = gloss_louo["kappa"]
    ratio = g / ceil if ceil > 0 else float("nan")
    if ratio >= 0.8:
        verdict = ("(B) signal PRESENT: gloss phoneme content is as decodable "
                   "in-domain as usc_lss -> the transfer drop is OOD/representation, "
                   "NOT a physical articulator limit")
    elif ratio <= 0.4:
        verdict = ("(A) signal WEAK even in-domain -> consistent with a physical "
                   "articulator limit (information loss)")
    else:
        verdict = "MIXED -> inspect per-articulator macro-recall (docs/GLOSS.md §7)"
    result["indomain_ratio_gloss_over_usc"] = round(float(ratio), 3)
    result["transfer_floor_token_kappa"] = 0.19    # context only, different scale
    result["verdict"] = verdict
    # cross-speaker caveat: LOUO shares speakers train/test; LOSO (3 speakers) does not
    loso = result.get("gloss_indomain_loso")
    if loso is not None and loso["kappa"] < 0.5 * g:
        result["speaker_caveat"] = (
            f"cross-speaker (LOSO) collapses to κ={loso['kappa']:.3f}: the recoverable "
            "signal is largely speaker-specific (only 3 speakers), so speaker-OOD "
            "dominates the transfer gap; pathology vs acquisition-domain stay "
            "unseparable without a healthy same-domain control (docs/GLOSS.md §8).")

    print(f"\n===== Test 2: in-domain phoneme-decodability ceiling "
          f"({args.encoder}, {args.classifier}"
          f"{f', pca{args.pca}' if args.pca else ''}, "
          f"{args.speaker or 'pooled'}) =====")
    print(f"{'setting':<26}{'κ':>8}{'acc':>8}{'grp':>7}{'vow':>7}{'con':>7}{'n_seg':>8}")

    def _row(name, s):
        mr = s["macro_recall"]
        print(f"{name:<26}{s['kappa']:>8.3f}{s['acc']:>8.3f}"
              f"{mr['groups']:>7.3f}{mr['vowels']:>7.3f}{mr['consonants']:>7.3f}"
              f"{s['n_seg']:>8}")
    _row("usc_lss in-domain", usc)
    _row("gloss in-domain (LOUO)", gloss_louo)
    if "gloss_indomain_loso" in result:
        _row("gloss in-domain (LOSO)", result["gloss_indomain_loso"])
    print(f"\n  usc_lss in-domain ceiling κ={ceil:.3f}  |  gloss/usc ratio={ratio:.2f}"
          f"  |  (token-level transfer floor κ≈0.19, different scale)")
    print(f"  VERDICT: {verdict}")
    if "speaker_caveat" in result:
        print(f"  CAVEAT: {result['speaker_caveat']}")

    sfx = f"_{args.speaker}" if args.speaker else ""
    pcs = f"_pca{args.pca}" if args.pca else ""
    op = os.path.join(out_dir, f"gloss_indomain_{args.encoder}"
                      f"_{args.classifier}{pcs}{sfx}_s{args.seed}.json")
    json.dump(result, open(op, "w"), indent=2)
    print(f"[test2] wrote {op}")


if __name__ == "__main__":
    main()
