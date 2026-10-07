#!/usr/bin/env python3
"""t-SNE + confusion matrices for the BEST phoneme model: tssl256comb215 x attentive_lstm.

Unlike the cross-model tools (`tsne_phonemes` / `confmat_phonemes`, which assume the
chunk-mode `attentive` probe and score via `predict()`), the attentive_lstm probe was
trained/evaluated in UTTERANCE mode (~800-token bi-LSTM context). Chunk-mode scoring
would give the wrong predictions, so this script drives the probe utterance-by-utterance
exactly like `eval_phoneme._utt_eval` and collects, per temporal token:
  * true phoneme id, argmax prediction  -> confusion matrices
  * the POST-LSTM penultimate vector h   -> rep A t-SNE (the real decision space)
  * (rep B = the encoder's mean-over-space geometry, probe-independent, via build_reps)

Reuses the phonetic-class maps + renderers from the two cross-model tools so the
groupings (manner / vowel / consonant / place) match the rest of the docs.

Usage (compute/interactive node, NOT login -- multi-cache torch/sklearn load):
  source dev_artiJEPA/scripts/_env.sh; export OMP_NUM_THREADS=4
  python dev_artiJEPA/scratchpad/viz_attentive_lstm.py --split test_lss
"""
import argparse, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import artijepa.eval_phoneme as E
import artijepa.tsne_phonemes as TS
import artijepa.confmat_phonemes as CM
import artijepa.phonemes as P

DEF_PROBE = ("/scratch1/hongn/artijepa/eval/"
             "phoneme_usc_lss_tssl256comb215sp_43c1fe20dd_attentive_lstm_ce_s0.pt")


@torch.no_grad()
def utt_forward(clf, ds, device):
    """Drive the attentive_lstm probe over whole utterances (mirrors _utt_eval's loop
    + forward). Returns flat per-token (true_ids, pred_ids, H[M,2Hs]) over all non-pad
    tokens, in utterance order."""
    clf.eval()
    amp = device.type == "cuda"
    T_ids, P_ids, Hs = [], [], []
    for x, y, in_lens, _tg, _tl, _utts in E._utt_loader(ds, 1024, 2, False, prefetch_factor=1):
        x = x.to(device, non_blocking=True)
        B, T, S, D = x.shape
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            q = clf.pooler(x.reshape(B * T, S, D)).squeeze(1)      # [B*T, D]  spatial pool
            h = clf.norm(q.reshape(B, T, D))                       # [B, T, D]
            pk = torch.nn.utils.rnn.pack_padded_sequence(
                h, in_lens.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = clf.rnn(pk)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(
                h, batch_first=True, total_length=T)               # [B, T, 2Hs] penultimate
            logits = clf.head(h)                                   # [B, T, C]
        pred = logits.float().argmax(-1).cpu().numpy()
        hf = h.float().cpu().numpy()
        for b in range(B):
            n = int(in_lens[b])
            T_ids.append(y[b][:n].numpy()); P_ids.append(pred[b][:n]); Hs.append(hf[b][:n])
    return (np.concatenate(T_ids), np.concatenate(P_ids), np.concatenate(Hs, 0))


def subsample(phon_ids, seed, cap):
    """Balanced <=cap tokens/phoneme over valid (non-pad, non-sil) tokens -> index array."""
    valid = (phon_ids != P.IGNORE_INDEX) & (phon_ids != P.SIL_IDX)
    idx = np.where(valid)[0]
    rng = np.random.default_rng(seed); sel = []
    for p in np.unique(phon_ids[idx]):
        ip = idx[phon_ids[idx] == p]
        sel.append(rng.choice(ip, cap, replace=False) if len(ip) > cap else ip)
    return np.sort(np.concatenate(sel))


def tsne_panel(X, phon_ids, view, seed, perplexity, pca):
    """One t-SNE view: focus-mask -> PCA -> embed. Returns (xy, pids)."""
    foc = TS.VIEWS[view]["focus"]
    mask = (np.ones(len(phon_ids), bool) if foc is None
            else np.isin(phon_ids, list(TS.FOCUS_IDX[foc])))
    Xr = X[mask]
    Xr = TS.pca_reduce(Xr, pca) if pca and pca > 0 else Xr
    return TS.embed(Xr, "tsne", seed, perplexity), phon_ids[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=DEF_PROBE)
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml")
    ap.add_argument("--split", default="test_lss", choices=["test", "test_lss"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--out", default="/scratch1/hongn/artijepa/eval/attn_lstm_viz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clf, ck = E.load_probe(args.probe, device=device)
    assert ck["probe_kind"] == "attentive_lstm", f"expected attentive_lstm, got {ck['probe_kind']}"
    kap = ck["metrics"].get("tests", {}).get("test_lss", {}).get("kappa") \
        if args.split == "test_lss" else ck["metrics"]["test"]["kappa"]
    print(f"[viz] probe={os.path.basename(args.probe)} tag={ck['feature_tag']} "
          f"{args.split} kappa={kap}")

    # config -> restore encoder.spec + tag so extract() cache-hits the RIGHT cache
    cfg = E.load_config(args.config)
    cfg["probe"]["type"] = "attentive_lstm"; cfg["probe"]["loss"] = ck["loss"]
    cfg["data"]["pool_spatial"] = False
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    cfg["meta"]["tag"] = ck["feature_tag"].rsplit("_", 1)[0].removesuffix("sp")
    assert E._tag(cfg, "train")[0] == ck["feature_tag"], "feature-cache tag mismatch"

    feats, labels, meta = E.extract(None, cfg, args.split, device, torch.float16)  # cache-hit
    split_ds = E.build_dataset(cfg, args.split)[0]
    refs = split_ds.reference_sequences()
    ds = E._UttSpatialDS(feats, labels, meta, refs)

    print("[viz] utterance-mode forward (pooler -> bi-LSTM -> head) ...")
    true_ids, pred_ids, H = utt_forward(clf, ds, device)
    print(f"[viz]   {len(true_ids)} tokens; frame-acc(all non-sil)="
          f"{(pred_ids[(true_ids!=P.IGNORE_INDEX)&(true_ids!=P.SIL_IDX)] == true_ids[(true_ids!=P.IGNORE_INDEX)&(true_ids!=P.SIL_IDX)]).mean():.3f}")

    tag = f"tssl215_attentive_lstm_{ck['loss']}_{args.split}_s{args.seed}"

    # ===== CONFUSION MATRICES (utterance-mode predictions) =====================
    keep = true_ids != P.IGNORE_INDEX
    mats = CM.build_matrices(true_ids[keep], pred_ids[keep])
    rec = {m: CM.macro_recall(mats[m][0]) for m in mats}
    print("[viz] macro-recall  " + "  ".join(f"{m}={rec[m]:.3f}" for m in CM.MTYPES))
    # combined row: the 4 granularities side by side
    row = list(CM.MTYPES)
    fig, axes = plt.subplots(1, len(row), squeeze=False,
                             figsize=(sum(CM.PANEL[m][0] for m in row) + 1, 4.6))
    im = None
    for j, m in enumerate(row):
        M, rows, cols = mats[m]
        im = CM.plot_confmat(axes[0][j], M, rows, cols,
                             f"{m}\nmacro-recall={rec[m]:.3f}", CM.PANEL[m][2])
    fig.suptitle(f"tssl256comb215 x attentive_lstm x {ck['loss']}  -  confusion (rows=true, "
                 f"cols=pred, row-norm recall %)  -  usc_lss {args.split}  (k={kap:.3f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.93])
    cax = fig.add_axes([0.985, 0.12, 0.006, 0.72]); fig.colorbar(im, cax=cax, label="recall")
    fp = os.path.join(args.out, f"confmat_{tag}.png")
    fig.savefig(fp, dpi=170, bbox_inches="tight"); plt.close(fig); print(f"[viz] wrote {fp}")

    # ===== t-SNE: rep A (post-LSTM decision space) + rep B (encoder geometry) ===
    # rep B is probe-independent: build_reps samples its own balanced token set.
    repB, pidB = TS.build_reps(feats, labels, ["B"], None, args.seed, args.cap)
    repB = repB["B"]
    selA = subsample(true_ids, args.seed, args.cap)
    XA, pidA = H[selA], true_ids[selA]

    reps = {"A": ("A: post-LSTM decision space", XA, pidA),
            "B": ("B: raw-pooled encoder geometry", repB, pidB)}
    for rk, (rlabel, X, pid) in reps.items():
        for view in TS.VIEW_ORDER:
            color = TS.VIEWS[view]["color"]
            xy, pv = tsne_panel(X, pid, view, args.seed, args.perplexity, args.pca)
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            TS.scatter(ax, xy, pv, f"tssl256comb215 x attentive_lstm x {ck['loss']}\n{TS.VIEWS[view]['title']}"
                       f"  -  {rlabel}  ({len(pv)} tok)", color)
            TS.legend_for(fig, color, ncol=min(len(TS.color_scheme(color)[1]), 6))
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            fp = os.path.join(args.out, f"tsne_{view}_rep{rk}_{tag}.png")
            fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[viz] wrote {fp}")

    print(f"\n[viz] DONE -> {args.out}")


if __name__ == "__main__":
    main()
