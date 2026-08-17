"""Pre/post-glossectomy phoneme decoding + viz (Arti-JEPA eval Task 9).

**Transfer** eval: take a frozen-encoder **attentive** phoneme probe (default the winning
`tssl256comb100`) and apply it -- with NO retraining -- to gloss features, separately
for **pre-** and **post-**glossectomy. One probe on both conditions ⇒ a clean pre→post
Δ in kappa / PER and directly comparable t-SNE + confusion matrices.

Parallel key: `--encoder tssl256comb215` transfers the SAME-geometry probe trained on the
FINAL epoch-215 checkpoint (its `phoneme_usc_lss_tssl256comb215sp_*_attentive_ce_s*.pt`
probes exist from `phoneme_xmodel.sbatch tssl256comb215`; `pick_probe`'s glob resolves the
key automatically, own feature cache). ckpt_100 is left as the default so existing gloss
rows stay reproducible; cross-domain phoneme showed ckpt_215 ≈ ckpt_100 (RESULTS_phonepred 1b).

Two probes exist for `tssl256comb100` (identical frozen encoder `ckpt_100.pt`, so the
gloss feature cache is shared — only the head/predictions differ). Pick with `--train-set`:
  - `annot16` (**default**): the 75-speaker Annot-16 *combined* probe (`docs/phonePred.md`
    Phase-1). It has already been measured OOD on a *healthy* speaker in a new acquisition
    domain (`test_lss` = usc_lss, κ≈0.334) — the ideal "appearance/domain shift only"
    anchor. Transferring the SAME probe to gloss adds pathology on top, so the extra drop
    below that healthy-OOD anchor is the identifiable pathology component.
  - `usc_lss`: the older single-speaker usc_lss-only probe (`docs/GLOSS.md` §1-9).

Pipeline (all reused, no new model code):
  1. `pick_probe` locates the saved probe `.pt` for the chosen train-set (by inspecting
     `ck['manifest']`) -> its `encoder_spec` names the frozen encoder checkpoint to rebuild.
  2. for each condition: `eval_phoneme.extract` the un-pooled [N,T',S',D] gloss
     features (cached), then `predict` with the transferred probe -> per-token phonemes.
  3. `eval_phoneme.evaluate` -> kappa / PER, overall AND **per gloss speaker** (subset the
     pooled extraction by `ds.rows` speaker, no re-extraction); `confmat_phonemes.build_matrices`
     + `tsne_phonemes.build_reps` -> the pre-vs-post panels.

Gloss data (manifests + usc_lss-format JSON) come from `artijepa.gloss`; run that first
(or pass --build). Outputs -> `/scratch1/hongn/artijepa/eval/gloss/`.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    python -m artijepa.gloss                       # build gloss_{pre,post}_manifest.csv
    python -m artijepa.eval_gloss --encoder tssl256comb100          # annot16 probe, pooled + per-speaker
    python -m artijepa.eval_gloss --train-set usc_lss              # old usc_lss-only probe
    python -m artijepa.eval_gloss --speaker spk1                    # restrict extraction to one speaker
"""

import argparse
import copy
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import artijepa.phonemes as P
from artijepa.eval_phoneme import (
    load_config, load_frozen_encoder, extract, build_dataset, predict,
    evaluate, load_probe, _UttSpatialDS, _utt_loader)
from artijepa.tsne_phonemes import (
    build_reps, pca_reduce, embed, scatter, legend_for, color_scheme,
    REP_LABEL, DEFAULT_ARTI_OUT, FOCUS_IDX)
from artijepa.confmat_phonemes import (
    build_matrices, plot_confmat, macro_recall, PANEL)

CONDS = ["pre", "post"]
GLOSS_ROOT = "/scratch1/hongn/gloss"

# substring that identifies each probe's training manifest (ck['manifest'])
TRAINSET_MANIFEST = {"annot16": "annot16", "usc_lss": "usc_lss"}


def pick_probe(encoder, seed, eval_dir, train_set, head="attentive", loss="ce"):
    """Locate the saved probe .pt for an encoder key + training corpus.

    Several probes can share the `<encoder>sp_<hash>` stem (same frozen encoder, different
    *training* manifest — e.g. tssl256comb100 has an Annot-16 combined probe and an older
    usc_lss-only one). `find_probe` would return whichever sorts first; here we
    disambiguate on the checkpoint's own `manifest` field so the choice is explicit and
    order-independent. `head`/`loss` select the probe family: `attentive`/`ce` = the
    Phase-1 chunk-mode probe (default), `attentive_lstm`/`ce` = the Phase-2 utterance-mode
    bi-LSTM probe (RESULTS_phonepred.md Phase 2). Returns the matching .pt path."""
    pat = os.path.join(eval_dir,
                       f"phoneme_usc_lss_{encoder}sp_*_{head}_{loss}_s{seed}.pt")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"[gloss] no {head}/{loss} probe for encoder={encoder!r} "
                         f"seed={seed} at\n  {pat}")
    want = TRAINSET_MANIFEST.get(train_set, train_set)
    matched = []
    for h in hits:
        ck = torch.load(h, map_location="cpu", weights_only=False)
        if want in str(ck.get("manifest", "")):
            matched.append((h, str(ck.get("manifest"))))
    if not matched:
        avail = {os.path.basename(h): torch.load(h, map_location="cpu",
                 weights_only=False).get("manifest") for h in hits}
        raise SystemExit(f"[gloss] no {encoder} probe trained on {train_set!r} "
                         f"(want '{want}' in manifest); available:\n  "
                         + "\n  ".join(f"{k}: {v}" for k, v in avail.items()))
    if len(matched) > 1:
        print(f"[gloss] WARNING {len(matched)} probes match train_set={train_set!r}; "
              f"using {os.path.basename(matched[0][0])}")
    print(f"[gloss] probe train_set={train_set!r} -> {os.path.basename(matched[0][0])} "
          f"(trained on {matched[0][1]})")
    return matched[0][0]


def _cond_manifest(cond, speaker, scratch):
    """Path to the (optionally speaker-filtered) manifest for a condition."""
    src = os.path.join(GLOSS_ROOT, f"gloss_{cond}_manifest.csv")
    if not os.path.exists(src):
        raise SystemExit(f"[gloss] missing {src}; run `python -m artijepa.gloss` first")
    if not speaker:
        return src, sum(1 for _ in csv.DictReader(open(src)))
    rows = [r for r in csv.DictReader(open(src)) if r["speaker"] == speaker]
    dst = os.path.join(scratch, f"gloss_{cond}_{speaker}.csv")
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=next(iter(rows)).keys() if rows else
                           ["utt_id"])
        w.writeheader(); w.writerows(rows)
    return dst, len(rows)


def _speaker_groups(rows, cond):
    """Per-utt group label for the per-speaker breakdown: the speaker, OR
    ``speaker/<session>`` when one speaker was recorded in multiple acquisition
    sessions within this condition. Only spk2 post has this (two post-op sessions
    `post1`/`post2`, distinct video subfolders) — everyone else stays a bare speaker.
    Session comes from the manifest's `session` column (pre|post|post1|post2); it collapses
    to the speaker whenever it equals the condition (single-session case).

    That column is authoritative and must NOT be re-derived from `path`: manifests now point
    at the native source tree `<spk>/<session>/{video,avi}/<utt>.avi`, whose parent dir is the
    media subdir, so path-sniffing would label every clip `video`/`avi` and silently merge
    spk2's post1+post2. Older manifests lacking the column fall back to the path (they pointed
    at `resampled_video/<spk>/<session>/`, where the parent IS the session)."""
    labels = []
    for r in rows:
        sess = r.get("session") or os.path.basename(os.path.dirname(r.get("path", "")))
        spk = r.get("speaker", "?")
        labels.append(f"{spk}/{sess}" if sess and sess != cond else spk)
    return labels


def run_condition(encoder, base_cfg, cond, manifest, tag, probe, device, dtype):
    """Extract gloss features for one condition + apply the transferred probe."""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["manifest"] = manifest
    cfg["data"]["pool_spatial"] = False               # attentive -> un-pooled grid
    cfg["meta"]["tag"] = tag
    feats, labels, meta = extract(encoder, cfg, "test", device, dtype)
    ds = build_dataset(cfg, "test")[0]
    refs = ds.reference_sequences()
    num_classes, drop = ds.num_classes, tuple(ds.collapse_drop)
    spk_of_utt = _speaker_groups(ds.rows, cond)               # utt row idx -> group label
    pred = predict(probe, feats, device)
    m = evaluate(pred, labels, meta, refs, num_classes, drop)
    flat_t = labels.reshape(-1)
    keep = flat_t != P.IGNORE_INDEX
    mats = build_matrices(flat_t[keep], pred.reshape(-1)[keep])
    m["macro_recall"] = {k: macro_recall(mats[k][0]) for k in mats}
    # per-speaker breakdown: subset the pooled extraction by each utt's speaker
    # (meta[:,0] = utt row idx) and re-evaluate -- no re-extraction.
    per_spk, per_spk_mats = {}, {}
    spk_of_row = np.array([spk_of_utt[int(u)] for u in meta[:, 0]], dtype=object)
    for spk in sorted(set(spk_of_utt)):
        rmask = spk_of_row == spk
        if not rmask.any():
            continue
        per_spk[spk] = evaluate(pred[rmask], labels[rmask], meta[rmask], refs,
                                num_classes, drop)
        st, sp = labels[rmask].reshape(-1), pred[rmask].reshape(-1)
        sk = st != P.IGNORE_INDEX
        per_spk_mats[spk] = build_matrices(st[sk], sp[sk])
    m["per_speaker"] = per_spk
    print(f"[gloss] {cond}: κ={m['kappa']} PERµ={m['per_micro']} "
          f"frame_acc={m['frame_acc']}  macro-recall "
          + " ".join(f"{k}={m['macro_recall'][k]:.3f}" for k in
                     ("groups", "vowels", "consonants")))
    for spk, sm in per_spk.items():
        print(f"[gloss]   {spk}: κ={sm['kappa']} PERµ={sm['per_micro']} "
              f"n_utt={sm['n_utt']}")
    return {"feats": feats, "labels": labels, "pred": pred, "metrics": m,
            "mats": mats, "per_spk_mats": per_spk_mats, "spk_of_row": spk_of_row}


# --------------------------------------------------------------------------- #
# utterance-mode transfer (attentive_lstm head, RESULTS_phonepred.md Phase 2)
#
# The bi-LSTM head models the WHOLE utterance (~800 tokens), so it cannot be scored
# with the chunk-mode `predict` (that would run the recurrence over 16-token chunks
# and discard the utterance context the head was trained on). We mirror the training-
# time eval path (`eval_phoneme._UttSpatialDS` + `_utt_loader`): one item = one whole
# utterance read contiguously from the shared spatial cache, run through the LSTM with
# its true length, giving per-frame phoneme predictions we then score exactly like the
# chunk path (frame kappa/acc + per-utterance PER), overall and per gloss speaker.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _utt_infer(clf, uds, device, max_tokens, workers, amp):
    """-> {utt: pred_frames}, {utt: true_frames} over the non-pad tokens of each utt."""
    clf.eval()
    preds, labs = {}, {}
    for x, y, in_lens, _tg, _tl, utts in _utt_loader(uds, max_tokens, workers, False,
                                                     prefetch_factor=1):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = clf(x, lens=in_lens)                 # [B,T,C]; lens masks the pad
        ids = logits.float().argmax(-1).cpu().numpy()     # [B,T]
        for b, u in enumerate(utts):
            n = int(in_lens[b])
            preds[int(u)] = ids[b][:n]
            labs[int(u)] = y[b][:n].numpy()
    return preds, labs


def _utt_metrics(preds, labs, refs, num_classes, drop, utts):
    """Frame kappa/acc + per-utterance PER over a chosen subset of utterances
    (same metric definitions as eval_phoneme._utt_eval / evaluate)."""
    ps, ts, pers = [], [], []
    tot_err = tot_ref = 0
    for u in utts:
        seq, lab = preds[u], labs[u]
        ps.append(seq); ts.append(lab)
        hyp = P.collapse_sequence(seq, drop=tuple(drop))
        ref = refs.get(u, [])
        _, nref = P.phoneme_error_rate(ref, hyp)
        if nref > 0:
            e = P.edit_distance(ref, hyp)
            tot_err += e; tot_ref += nref; pers.append(e / nref)
    fp = np.concatenate(ps); ft = np.concatenate(ts)
    return {"kappa": round(P.cohen_kappa(ft, fp, num_classes), 4),
            "frame_acc": round(P.frame_accuracy(ft, fp), 4),
            "per_micro": round(tot_err / max(1, tot_ref), 4),
            "per_macro": round(float(np.mean(pers)) if pers else 1.0, 4),
            "n_utt": len(pers)}


def run_condition_utt(encoder, base_cfg, cond, manifest, tag, probe, device, dtype):
    """Utterance-mode transfer for the attentive_lstm head (extract + LSTM inference)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["manifest"] = manifest
    cfg["data"]["pool_spatial"] = False               # spatial head -> un-pooled grid
    cfg["meta"]["tag"] = tag
    feats, labels, meta = extract(encoder, cfg, "test", device, dtype)
    ds = build_dataset(cfg, "test")[0]
    refs = ds.reference_sequences()
    num_classes, drop = ds.num_classes, tuple(ds.collapse_drop)
    spk_of_utt = _speaker_groups(ds.rows, cond)               # utt row idx -> group label

    uds = _UttSpatialDS(feats, labels, meta, refs)            # whole-utterance items
    mt = cfg["probe"].get("utt_max_tokens", 1024)
    workers = cfg["probe"].get("utt_eval_workers", 2)
    amp = (device.type == "cuda")
    preds, labs = _utt_infer(probe, uds, device, mt, workers, amp)
    utts = sorted(preds)                                      # utts with a non-empty ref

    m = _utt_metrics(preds, labs, refs, num_classes, drop, utts)
    flat_p = np.concatenate([preds[u] for u in utts])
    flat_t = np.concatenate([labs[u] for u in utts])
    keep = flat_t != P.IGNORE_INDEX
    mats = build_matrices(flat_t[keep], flat_p[keep])
    m["macro_recall"] = {k: macro_recall(mats[k][0]) for k in mats}
    per_spk, per_spk_mats = {}, {}
    for spk in sorted(set(spk_of_utt)):
        us = [u for u in utts if spk_of_utt[u] == spk]
        if not us:
            continue
        per_spk[spk] = _utt_metrics(preds, labs, refs, num_classes, drop, us)
        st = np.concatenate([labs[u] for u in us])
        sp = np.concatenate([preds[u] for u in us])
        sk = st != P.IGNORE_INDEX
        per_spk_mats[spk] = build_matrices(st[sk], sp[sk])
    m["per_speaker"] = per_spk
    # feats-row -> speaker group, so the per-speaker t-SNE can mask the same cache
    spk_of_row = np.array([spk_of_utt[int(u)] for u in meta[:, 0]], dtype=object)
    print(f"[gloss] {cond}: κ={m['kappa']} PERµ={m['per_micro']} "
          f"frame_acc={m['frame_acc']}  macro-recall "
          + " ".join(f"{k}={m['macro_recall'][k]:.3f}" for k in
                     ("groups", "vowels", "consonants")))
    for spk, sm in per_spk.items():
        print(f"[gloss]   {spk}: κ={sm['kappa']} PERµ={sm['per_micro']} "
              f"n_utt={sm['n_utt']}")
    return {"feats": feats, "labels": labels, "pred": None, "metrics": m,
            "mats": mats, "per_spk_mats": per_spk_mats, "spk_of_row": spk_of_row}


# --------------------------------------------------------------------------- #
# plotting: pre | post, side by side
# --------------------------------------------------------------------------- #
def plot_confmats(res, out_dir, enc, seed, sfx, kappas):
    for mtype in ("groups", "vowels", "consonants"):
        pw, ph, fscale = PANEL[mtype]
        fig, axes = plt.subplots(1, 2, figsize=(pw * 2, ph), squeeze=False)
        im = None
        for j, cond in enumerate(CONDS):
            M, rows, cols = res[cond]["mats"][mtype]
            mr = res[cond]["metrics"]["macro_recall"][mtype]
            im = plot_confmat(axes[0][j], M, rows, cols,
                              f"{cond}  (κ={kappas[cond]:.3f}, macro-rec={mr:.3f})",
                              fscale)
        d = (res["post"]["metrics"]["macro_recall"][mtype]
             - res["pre"]["metrics"]["macro_recall"][mtype])
        fig.suptitle(f"{enc} — {mtype} confusion, pre vs post glossectomy "
                     f"(row-normalized recall %)   Δmacro-rec(post−pre)={d:+.3f}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.97, 0.94])
        cax = fig.add_axes([0.975, 0.12, 0.008, 0.72])
        fig.colorbar(im, cax=cax, label="recall")
        fp = os.path.join(out_dir, f"confmat_gloss_{mtype}_{enc}{sfx}_s{seed}.png")
        fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"[gloss] wrote {fp}")


# --------------------------------------------------------------------------- #
# per-speaker panels: one row per gloss speaker, one column per session
#
# The pooled pre|post figures average over speakers, but the per-speaker kappa table
# shows the speakers behave differently (spk3 in particular, RESULTS_gloss.md §Test-3).
# These lay the SAME matrices/embeddings out per speaker x session so a per-speaker
# place-of-articulation deficit (if any) is visible instead of averaged away.
# `spk2` has two post-op sessions, so the column is the session (pre|post1|post2),
# not the condition.
# --------------------------------------------------------------------------- #
def _panel_grid(res):
    """-> (row speakers, column sessions, {(spk, col): (cond, group_label)})."""
    cells = {}
    for cond in CONDS:
        for g in res[cond].get("per_spk_mats", {}):
            base, _, sess = g.partition("/")
            cells[(base, sess or cond)] = (cond, g)
    rows = sorted({r for r, _ in cells})
    cols = [c for c in ("pre", "post", "post1", "post2")
            if any((r, c) in cells for r in rows)]
    return rows, cols, cells


def _cell_kappa(res, cond, group):
    return res[cond]["metrics"].get("per_speaker", {}).get(group, {}).get("kappa")


def plot_confmats_per_speaker(res, out_dir, enc, seed, sfx, mtypes):
    """Speaker x session grid of confusion matrices (default: consonant PLACE)."""
    rows, cols, cells = _panel_grid(res)
    for mtype in mtypes:
        pw, ph, fscale = PANEL[mtype]
        fig, axes = plt.subplots(len(rows), len(cols),
                                 figsize=(pw * len(cols), ph * len(rows)),
                                 squeeze=False)
        im = None
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                ax = axes[i][j]
                if (r, c) not in cells:
                    ax.axis("off")
                    continue
                cond, g = cells[(r, c)]
                M, rlab, clab = res[cond]["per_spk_mats"][g][mtype]
                mr = macro_recall(M)
                k = _cell_kappa(res, cond, g)
                ktxt = f"κ={k:.3f}, " if k is not None else ""
                im = plot_confmat(ax, M, rlab, clab,
                                  f"{g}  ({ktxt}macro-rec={mr:.3f}, n={int(M.sum())})",
                                  fscale)
        fig.suptitle(f"{enc} — {mtype} confusion per gloss speaker × session "
                     f"(row-normalized recall %)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.97, 0.96])
        cax = fig.add_axes([0.975, 0.12, 0.008, 0.72])
        fig.colorbar(im, cax=cax, label="recall")
        fp = os.path.join(out_dir,
                          f"confmat_gloss_{mtype}_perspk_{enc}{sfx}_s{seed}.png")
        fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"[gloss] wrote {fp}")


def plot_tsne_per_speaker(res, out_dir, enc, seed, sfx, reps, method, cap,
                          perplexity, pca):
    """Speaker x session grid of consonant-only t-SNE, colored by PLACE.

    Each panel embeds only that speaker/session's consonant tokens (vowels + sil +
    other speakers masked out of `labels`, which is all `build_reps` subsamples on),
    so the geometry is within-speaker -- not the cross-speaker identity split that
    dominates the pooled view (RESULTS_gloss.md §Per-vowel cross-speaker t-SNE)."""
    rows, cols, cells = _panel_grid(res)
    cons = np.array(sorted(FOCUS_IDX["consonant"]))
    for rep in reps:
        fig, axes = plt.subplots(len(rows), len(cols),
                                 figsize=(4.3 * len(cols), 4.5 * len(rows)),
                                 squeeze=False)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                ax = axes[i][j]
                ax.set_xticks([]); ax.set_yticks([])
                if (r, c) not in cells:
                    ax.axis("off")
                    continue
                cond, g = cells[(r, c)]
                R = res[cond]
                lab = R["labels"].copy()
                lab[R["spk_of_row"] != g] = P.IGNORE_INDEX      # other speakers out
                lab[~np.isin(lab, cons)] = P.IGNORE_INDEX       # vowels + sil out
                n_tok = int((lab != P.IGNORE_INDEX).sum())
                if n_tok < 20:
                    ax.set_title(f"{g}  (only {n_tok} consonant tokens)", fontsize=9)
                    ax.axis("off")
                    continue
                X, pid = build_reps(R["feats"], lab, [rep],
                                    res["_pooler"] if rep == "A" else None, seed, cap)
                Xr = pca_reduce(X[rep], pca) if pca and pca > 0 else X[rep]
                k = _cell_kappa(res, cond, g)
                ktxt = f"κ={k:.3f}, " if k is not None else ""
                scatter(ax, embed(Xr, method, seed, perplexity), pid,
                        f"{g}  ({ktxt}{len(pid)} tokens)", "place")
        fig.suptitle(f"{enc} — {REP_LABEL[rep]} — {method}, consonants per gloss "
                     f"speaker × session (colored by place of articulation)", fontsize=12)
        legend_for(fig, "place", ncol=min(len(color_scheme("place")[1]), 8))
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fp = os.path.join(out_dir,
                          f"tsne_gloss_place_perspk_rep{rep}_{method}_{enc}{sfx}_s{seed}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[gloss] wrote {fp}")


def plot_tsne(res, out_dir, enc, seed, sfx, kappas, reps, method, cap, perplexity,
              pca, color="manner"):
    embeds = {}                                        # (cond, rep) -> (xy, pid)
    for cond in CONDS:
        r = res[cond]
        X, pid = build_reps(r["feats"], r["labels"], reps,
                            res["_pooler"] if "A" in reps else None, seed, cap)
        for rep in reps:
            Xr = pca_reduce(X[rep], pca) if pca and pca > 0 else X[rep]
            embeds[(cond, rep)] = (embed(Xr, method, seed, perplexity), pid)
    for rep in reps:
        fig, axes = plt.subplots(1, 2, figsize=(4.6 * 2, 4.8), squeeze=False)
        for j, cond in enumerate(CONDS):
            xy, pid = embeds[(cond, rep)]
            scatter(axes[0][j], xy, pid, f"{cond}  (κ={kappas[cond]:.3f})", color)
        fig.suptitle(f"{enc} — {REP_LABEL[rep]} — {method}, pre vs post glossectomy "
                     f"(colored by {color}, sil dropped)", fontsize=11)
        legend_for(fig, color, ncol=min(len(color_scheme(color)[1]), 8))
        fig.tight_layout(rect=[0, 0.07, 1, 0.94])
        fp = os.path.join(out_dir, f"tsne_gloss_rep{rep}_{method}_{enc}{sfx}_s{seed}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[gloss] wrote {fp}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="tssl256comb100",
                    help="encoder key whose attentive probe to transfer")
    ap.add_argument("--head", default="attentive",
                    choices=["attentive", "attentive_lstm"],
                    help="probe head to transfer: attentive (Phase-1 chunk-mode; default) "
                         "or attentive_lstm (Phase-2 utterance-mode bi-LSTM)")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="which trained-probe loss variant to transfer (default ce)")
    ap.add_argument("--train-set", default="annot16",
                    choices=["annot16", "usc_lss"],
                    help="which trained probe to transfer: annot16 (75-spk combined; "
                         "default, has a healthy-OOD anchor) or usc_lss (single-spk)")
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_gloss.yaml")
    ap.add_argument("--seed", type=int, default=0, help="which probe seed")
    ap.add_argument("--speaker", default=None, help="restrict to spk1|spk2|spk3 "
                    "(matched pre/post Δ; default = pooled across speakers)")
    ap.add_argument("--build", action="store_true", help="(re)build gloss manifests first")
    ap.add_argument("--per-speaker-figs", action="store_true",
                    help="also emit speaker × session grids: confusion matrices "
                         "(--per-speaker-mtypes) + consonant t-SNE colored by place")
    ap.add_argument("--per-speaker-mtypes", default="place,consonants",
                    help="comma list from groups,vowels,consonants,place for the "
                         "per-speaker confusion grid (default place,consonants)")
    ap.add_argument("--rep", default="both", choices=["both", "A", "B"])
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--arti-out", default=DEFAULT_ARTI_OUT)
    ap.add_argument("--scratch", default=os.environ.get("SCRATCHPAD", "/tmp"))
    args = ap.parse_args()

    if args.build:
        from artijepa.gloss import build_manifest
        build_manifest(speaker=args.speaker)

    eval_dir = os.path.join(args.arti_out, "eval")
    out_dir = os.path.join(eval_dir, "gloss"); os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg["meta"].get("dtype", "float16").lower(), torch.float32)

    # locate + load the trained probe; its encoder_spec = the frozen encoder to rebuild.
    pt = pick_probe(args.encoder, args.seed, eval_dir, args.train_set,
                    head=args.head, loss=args.loss)
    probe, ck = load_probe(pt, device=device)
    utt_mode = ck.get("probe_kind") == "attentive_lstm"
    ckm = ck.get("metrics", {}) or {}
    # OOD anchors from the probe's own held-out metrics:
    #  indomain = the probe's primary in-domain test (usc_lss test / annot16 sub043);
    #  healthy_ood = usc_lss cross-domain (annot16 probe only) = "appearance/domain shift,
    #  no pathology" reference the gloss (pathological) transfer is read against.
    indomain_kappa = (ckm.get("test", {}) or {}).get("kappa")
    healthy_ood_kappa = ((ckm.get("tests", {}) or {}).get("test_lss", {}) or {}).get("kappa")
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    print(f"[gloss] transfer probe {os.path.basename(pt)}  (train_set={args.train_set}, "
          f"in-domain test κ={indomain_kappa}"
          + (f", healthy-OOD usc_lss κ={healthy_ood_kappa}" if healthy_ood_kappa else "")
          + f")\n[gloss] frozen encoder <- {ck['encoder_spec']}  dtype={dtype} device={device}")

    encoder = load_frozen_encoder(cfg, device)
    res = {}
    for cond in CONDS:
        manifest, n = _cond_manifest(cond, args.speaker, args.scratch)
        tag = f"gloss_{cond}" + (f"_{args.speaker}" if args.speaker else "") + \
              f"_{args.encoder}"
        print(f"[gloss] === {cond} ({n} utts) tag={tag} head={args.head} ===")
        if utt_mode:
            res[cond] = run_condition_utt(encoder, cfg, cond, manifest, tag, probe,
                                          device, dtype)
        else:
            res[cond] = run_condition(encoder, cfg, cond, manifest, tag, probe,
                                      device, dtype)
    del encoder; torch.cuda.empty_cache()
    probe.to("cpu"); res["_pooler"] = probe.pooler   # t-SNE rep-A pools on CPU tensors

    kappas = {c: res[c]["metrics"]["kappa"] for c in CONDS}
    # keep annot16/usc_lss AND per-head outputs apart; the default attentive head keeps its
    # original stem (no suffix) so existing Phase-1 gloss files stay byte-identical.
    head_sfx = "" if args.head == "attentive" else f"_{args.head}"
    out_enc = f"{args.encoder}_{args.train_set}{head_sfx}"
    sfx = f"_{args.speaker}" if args.speaker else ""

    # -- pre vs post summary + Δ --------------------------------------------
    def dl(a, b):
        return f"{b - a:+.3f}"
    print(f"\n===== pre vs post glossectomy (transfer probe: {args.encoder}, "
          f"train_set={args.train_set}, in-domain κ={indomain_kappa}"
          + (f", healthy-OOD usc_lss κ={healthy_ood_kappa}" if healthy_ood_kappa else "")
          + f") {'['+args.speaker+']' if args.speaker else '[pooled]'} =====")
    mp, mo = res["pre"]["metrics"], res["post"]["metrics"]
    print(f"{'metric':<16}{'pre':>10}{'post':>10}{'Δ(post-pre)':>14}")
    print(f"{'kappa':<16}{mp['kappa']:>10.3f}{mo['kappa']:>10.3f}{dl(mp['kappa'], mo['kappa']):>14}")
    print(f"{'PER_micro':<16}{mp['per_micro']:>10.3f}{mo['per_micro']:>10.3f}{dl(mp['per_micro'], mo['per_micro']):>14}")
    print(f"{'frame_acc':<16}{mp['frame_acc']:>10.3f}{mo['frame_acc']:>10.3f}{dl(mp['frame_acc'], mo['frame_acc']):>14}")
    for k in ("groups", "vowels", "consonants"):
        a, b = mp["macro_recall"][k], mo["macro_recall"][k]
        print(f"{'macro-rec '+k:<16}{a:>10.3f}{b:>10.3f}{dl(a, b):>14}")

    # -- per-speaker × condition table (κ / PERµ) ---------------------------
    speakers = sorted(set(mp.get("per_speaker", {})) | set(mo.get("per_speaker", {})))
    print(f"\n----- per gloss speaker × condition (κ / PERµ / n_utt) -----")
    print(f"{'speaker':<10}{'pre_κ':>8}{'pre_PER':>9}{'pre_n':>7}"
          f"{'post_κ':>9}{'post_PER':>10}{'post_n':>8}")
    def _cell(m, spk):
        s = m.get("per_speaker", {}).get(spk)
        return s if s else {"kappa": float("nan"), "per_micro": float("nan"), "n_utt": 0}
    for spk in speakers:
        a, b = _cell(mp, spk), _cell(mo, spk)
        print(f"{spk:<10}{a['kappa']:>8.3f}{a['per_micro']:>9.3f}{a['n_utt']:>7}"
              f"{b['kappa']:>9.3f}{b['per_micro']:>10.3f}{b['n_utt']:>8}")
    print(f"{'pooled':<10}{mp['kappa']:>8.3f}{mp['per_micro']:>9.3f}{mp['n_utt']:>7}"
          f"{mo['kappa']:>9.3f}{mo['per_micro']:>10.3f}{mo['n_utt']:>8}")

    summary = {"encoder": args.encoder, "train_set": args.train_set,
               "head": args.head, "loss": args.loss,
               "probe_ckpt": os.path.basename(pt), "seed": args.seed,
               "speaker": args.speaker or "pooled",
               "anchors": {"indomain_test_kappa": indomain_kappa,
                           "healthy_ood_usc_lss_kappa": healthy_ood_kappa},
               "pre": mp, "post": mo}
    sp = os.path.join(out_dir, f"gloss_transfer_{out_enc}{sfx}_s{args.seed}.json")
    json.dump(summary, open(sp, "w"), indent=2)
    print(f"[gloss] wrote {sp}")

    # -- figures -------------------------------------------------------------
    plot_confmats(res, out_dir, out_enc, args.seed, sfx, kappas)
    reps = ["A", "B"] if args.rep == "both" else [args.rep]
    plot_tsne(res, out_dir, out_enc, args.seed, sfx, kappas, reps,
              args.method, args.cap, args.perplexity, args.pca)
    if args.per_speaker_figs:
        mtypes = [m.strip() for m in args.per_speaker_mtypes.split(",") if m.strip()]
        bad = [m for m in mtypes if m not in PANEL]
        if bad:
            raise SystemExit(f"[gloss] unknown --per-speaker-mtypes {bad}; "
                             f"choose from {sorted(PANEL)}")
        plot_confmats_per_speaker(res, out_dir, out_enc, args.seed, sfx, mtypes)
        plot_tsne_per_speaker(res, out_dir, out_enc, args.seed, sfx, reps,
                              args.method, args.cap, args.perplexity, args.pca)
        # per-speaker place matrices as numbers, for the writeup tables
        dump = {}
        for cond in CONDS:
            for g, mm in res[cond]["per_spk_mats"].items():
                for mtype in mtypes:
                    M, rl, cl = mm[mtype]
                    supp = M.sum(1, keepdims=True)
                    dump.setdefault(f"{cond}/{g}", {})[mtype] = {
                        "rows": rl, "cols": cl,
                        "counts": M.astype(int).tolist(),
                        "recall": (M / np.where(supp == 0, 1, supp)).tolist(),
                        "macro_recall": macro_recall(M),
                        "kappa": _cell_kappa(res, cond, g)}
        jp = os.path.join(out_dir,
                          f"confmat_gloss_perspk_values_{out_enc}{sfx}_s{args.seed}.json")
        json.dump(dump, open(jp, "w"), indent=2)
        print(f"[gloss] wrote {jp}")
    print(f"[gloss] done -> {out_dir}")


if __name__ == "__main__":
    main()
