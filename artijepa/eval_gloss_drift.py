"""Task 1 — where do a speaker's phonemes move after glossectomy? (docs/GLOSS.md §10)

We do NOT train a pre-vs-post classifier (it is lazy: it latches onto the big static
post-surgery shift, not the per-phoneme change). Instead, per speaker, we train ONE
attentive probe over all 41 phonemes on that speaker's pre+post phones, read out its
penultimate `q` per phone, and study where each phoneme goes in that phoneme-organized
space: a t-SNE (colored by phoneme, ○ pre / △ post) plus a **drift matrix** — for each
phoneme, what fraction of its POST phones land nearest each phoneme's PRE centroid. The
diagonal is "stayed itself"; a big off-diagonal i→j is phoneme i merging into j (a
neutralization, e.g. /s/→/sh/). A pre→pre leave-one-utterance-out baseline separates a
real post change from a phoneme that was already confusable.

Reads the speaker's cached pre/post features (run `eval_gloss.py --speaker <spk>` first
to extract them if the cache is missing). Trains a tiny probe -> needs a GPU ideally.

Usage:
    source dev_artiJEPA/scripts/_env.sh
    python -m artijepa.eval_gloss_drift --encoder tssl256comb100 --speaker spk1
"""

import argparse
import copy
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import artijepa.phonemes as P
from artijepa.eval_phoneme import (load_config, load_probe, build_dataset, _tag,
                                   TokenProbe, _FeatDS)
from artijepa.eval_gloss import _cond_manifest, CONDS
from artijepa.eval_gloss_indomain import _load_cache
from artijepa.tsne_phonemes import find_probe, pca_reduce, embed

LABIAL = {"p", "b", "m", "f", "v", "w"}                  # minimally tongue-dependent
TONGUE = {"t", "d", "s", "z", "l", "r", "n", "k", "g",   # lingual (coronal + velar + ...)
          "sh", "zh", "ch", "jh", "th", "dh", "ng", "y"}


# --------------------------------------------------------------------------- #
# train one attentive 41-way phoneme probe on the speaker's pre+post
# --------------------------------------------------------------------------- #
def train_phoneme_probe(pre, post, device, num_classes, epochs, lr, seed):
    torch.manual_seed(seed)
    dim = pre[0].shape[-1]
    clf = TokenProbe(dim, num_classes, kind="attentive", heads=8).to(device)
    ds = torch.utils.data.ConcatDataset([_FeatDS(pre[0], pre[1]),
                                         _FeatDS(post[0], post[1])])
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True,
                                         num_workers=2, drop_last=False)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=P.IGNORE_INDEX)
    acc = 0.0
    for ep in range(epochs):
        cur = lr * (ep + 1) / 5 if ep < 5 else \
            0.5 * lr * (1 + math.cos(math.pi * (ep - 5) / max(1, epochs - 5)))
        for g in opt.param_groups:
            g["lr"] = cur
        clf.train(); correct = tot = 0
        for x, y in loader:
            x = x.float().to(device); y = y.to(device)
            logits = clf(x)                               # [B,T',C]
            loss = lossf(logits.reshape(-1, num_classes), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                m = y != P.IGNORE_INDEX
                correct += int((logits.argmax(-1)[m] == y[m]).sum()); tot += int(m.sum())
        acc = correct / max(1, tot)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[drift]   ep {ep:2d} lr {cur:.2e} train-tok-acc {acc:.3f}")
    return clf.eval(), acc


# --------------------------------------------------------------------------- #
# per-phone q (penultimate vector), segment-pooled, sil dropped
# --------------------------------------------------------------------------- #
@torch.no_grad()
def phone_q(clf, feats, labels, meta, device):
    by_utt = {}
    for n, (u, c, _) in enumerate(meta):
        by_utt.setdefault(int(u), []).append((int(c), n))
    X, ph, utt = [], [], []
    for u, rows in by_utt.items():
        rows.sort(key=lambda z: z[0])
        qs, ls = [], []
        for _, n in rows:
            x = torch.from_numpy(np.asarray(feats[n], dtype=np.float32)).to(device)
            q = clf.pooler(x).squeeze(1).cpu().numpy()    # [T',S',D]->[T',1,D]->[T',D]
            qs.append(q); ls.append(labels[n])
        qseq = np.concatenate(qs, 0); lseq = np.concatenate(ls, 0)
        i, T = 0, len(lseq)
        while i < T:
            lab = int(lseq[i]); j = i + 1
            while j < T and int(lseq[j]) == lab:
                j += 1
            if lab != P.IGNORE_INDEX and lab != P.SIL_IDX:
                X.append(qseq[i:j].mean(0)); ph.append(lab); utt.append(u)
            i = j
    return np.asarray(X, np.float32), np.asarray(ph), np.asarray(utt)


# --------------------------------------------------------------------------- #
# drift matrices
# --------------------------------------------------------------------------- #
def _centroids(q, ph, phonemes):
    return np.stack([q[ph == p].mean(0) for p in phonemes])


def assign_to(q_src, ph_src, cent, phonemes):
    """row-normalized [P,P]: fraction of source phones of i nearest to centroid j."""
    idx = {p: k for k, p in enumerate(phonemes)}
    Pn = len(phonemes); M = np.zeros((Pn, Pn)); cnt = np.zeros(Pn)
    for x, p in zip(q_src, ph_src):
        if p not in idx:
            continue
        j = ((cent - x) ** 2).sum(1).argmin()
        M[idx[p]][j] += 1; cnt[idx[p]] += 1
    return M / np.maximum(1, cnt)[:, None]


def baseline_louo(q, ph, utt, phonemes):
    """pre->pre with leave-one-utterance-out centroids (no self-match)."""
    idx = {p: k for k, p in enumerate(phonemes)}
    Pn = len(phonemes); M = np.zeros((Pn, Pn)); cnt = np.zeros(Pn)
    for u in np.unique(utt):
        te = utt == u; tr = ~te
        valid = [p for p in phonemes if (tr & (ph == p)).any()]
        if not valid:
            continue
        cent = np.stack([q[tr & (ph == p)].mean(0) for p in valid])
        for x, p in zip(q[te], ph[te]):
            if p not in idx:
                continue
            j = valid[((cent - x) ** 2).sum(1).argmin()]
            M[idx[p]][idx[j]] += 1; cnt[idx[p]] += 1
    return M / np.maximum(1, cnt)[:, None]


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_tsne(qp, php, qo, pho, phonemes, names, method, seed, pca, perp, cap, rng,
              fp, title):
    X = np.concatenate([qp, qo]); ph = np.concatenate([php, pho])
    cond = np.array([0] * len(qp) + [1] * len(qo))
    keep = []
    for p in phonemes:
        for cc in (0, 1):
            si = np.where((ph == p) & (cond == cc))[0]
            if len(si) > cap:
                si = rng.choice(si, cap, replace=False)
            keep.append(si)
    keep = np.concatenate(keep)
    xy = embed(pca_reduce(X[keep], pca) if pca > 0 else X[keep], method, seed, perp)
    cmap = matplotlib.colormaps["tab20"]
    fig, ax = plt.subplots(figsize=(12, 10))
    for k, p in enumerate(phonemes):
        for cc, mk in ((0, "o"), (1, "^")):
            m = (ph[keep] == p) & (cond[keep] == cc)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=11, alpha=0.5, linewidths=0,
                           color=cmap(k % 20), marker=mk)
        mp = (ph[keep] == p) & (cond[keep] == 0)
        if mp.any():
            ax.text(xy[mp, 0].mean(), xy[mp, 1].mean(), names[p], fontsize=10,
                    fontweight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.scatter([], [], marker="o", color="gray", label="pre")
    ax.scatter([], [], marker="^", color="gray", label="post")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.set_title(title, fontsize=12)
    fig.tight_layout(); fig.savefig(fp, dpi=155, bbox_inches="tight"); plt.close(fig)
    print(f"[drift] wrote {fp}")


def plot_drift(base, drift, phonemes, names, fp, title):
    labels = [names[p] for p in phonemes]
    fig, axes = plt.subplots(1, 2, figsize=(1.0 + 0.42 * len(phonemes) * 2,
                                            1.0 + 0.42 * len(phonemes)), squeeze=False)
    for ax, M, sub in ((axes[0][0], base, "baseline: pre→pre (LOUO)"),
                       (axes[0][1], drift, "drift: post→pre")):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("nearest pre-centroid"); ax.set_ylabel("phoneme (source)")
        ax.set_title(sub, fontsize=10)
    fig.colorbar(im, ax=axes[0][1], fraction=0.046, label="fraction")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fp, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[drift] wrote {fp}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="tssl256comb100")
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_gloss.yaml")
    ap.add_argument("--speaker", default="spk1", help="one speaker (spk1|spk2|spk3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-per-cond", type=int, default=5,
                    help="keep a phoneme only if >= this many instances in BOTH conditions")
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--cap", type=int, default=200, help="max points per (phoneme,cond) in t-SNE")
    ap.add_argument("--arti-out", default="/scratch1/hongn/artijepa")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCHPAD", "/tmp"))
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dir = os.path.join(args.arti_out, "eval")
    out_dir = os.path.join(eval_dir, "gloss"); os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(args.config)
    cache_root = cfg["meta"]["cache_dir"]

    # usc probe only supplies encoder_spec so the cache hash matches the extraction
    pt = find_probe(args.encoder, args.seed, eval_dir)
    _, ck = load_probe(pt, device="cpu")
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    print(f"[drift] speaker={args.speaker} device={device}  encoder<-{ck['encoder_spec']}")

    # load the speaker's pre/post caches (no re-extraction)
    data = {}
    for cond in CONDS:
        manifest, n = _cond_manifest(cond, args.speaker, args.scratch)
        c2 = copy.deepcopy(cfg)
        c2["data"]["manifest"] = manifest
        c2["data"]["pool_spatial"] = False
        c2["meta"]["tag"] = f"gloss_{cond}_{args.speaker}_{args.encoder}"
        name, split = _tag(c2, "test")
        feats, labels, meta = _load_cache(os.path.join(cache_root, name), split)
        data[cond] = (feats, labels, meta)
        print(f"[drift] {cond}: {n} utts, {len(labels)} clips <- {name}")

    # train ONE 41-way attentive phoneme probe on pre+post
    clf, tacc = train_phoneme_probe(
        (data["pre"][0], data["pre"][1]), (data["post"][0], data["post"][1]),
        device, P.NUM_PHONEMES, args.epochs, args.lr, args.seed)

    # per-phone q for each condition
    qp, php, up = phone_q(clf, *data["pre"], device)
    qo, pho, uo = phone_q(clf, *data["post"], device)
    print(f"[drift] phones: pre {len(php)}, post {len(pho)}")

    # phonemes present with >= min in BOTH conditions
    phonemes = [p for p in range(P.NUM_PHONEMES)
                if p != P.SIL_IDX
                and (php == p).sum() >= args.min_per_cond
                and (pho == p).sum() >= args.min_per_cond]
    names = {p: P.IDX2PHON[p] for p in phonemes}
    print(f"[drift] {len(phonemes)} phonemes kept: {[names[p] for p in phonemes]}")

    cent_pre = _centroids(qp, php, phonemes)
    drift = assign_to(qo, pho, cent_pre, phonemes)          # post -> pre
    base = baseline_louo(qp, php, up, phonemes)             # pre  -> pre (LOUO)

    # per-phoneme summary: self-retention + top merge target (excluding self)
    idx = {p: k for k, p in enumerate(phonemes)}
    per_ph = {}
    mergers = []
    for p in phonemes:
        r = drift[idx[p]].copy(); self_ret = float(r[idx[p]])
        r[idx[p]] = -1
        tgt = phonemes[int(r.argmax())]
        per_ph[names[p]] = {
            "retention_post": round(self_ret, 3),
            "baseline_retention": round(float(base[idx[p]][idx[p]]), 3),
            "top_merge_into": names[tgt],
            "top_merge_frac": round(float(drift[idx[p]][idx[tgt]]), 3),
            "n_pre": int((php == p).sum()), "n_post": int((pho == p).sum()),
            "class": "tongue" if names[p] in TONGUE else
                     ("labial" if names[p] in LABIAL else "vowel/other")}
        mergers.append((names[p], names[tgt], per_ph[names[p]]["top_merge_frac"]))

    def _mean_ret(group):
        v = [per_ph[n]["retention_post"] for n in per_ph if per_ph[n]["class"] == group]
        return round(float(np.mean(v)), 3) if v else None

    mergers.sort(key=lambda z: -z[2])
    result = {
        "encoder": args.encoder, "speaker": args.speaker, "seed": args.seed,
        "probe_train_tok_acc": round(tacc, 3), "n_phonemes": len(phonemes),
        "mean_retention_tongue": _mean_ret("tongue"),
        "mean_retention_labial": _mean_ret("labial"),
        "top_mergers": [{"from": a, "into": b, "frac": f} for a, b, f in mergers[:10]],
        "per_phoneme": per_ph}

    import json
    sfx = f"_{args.speaker}"
    jp = os.path.join(out_dir, f"gloss_drift_{args.encoder}{sfx}_s{args.seed}.json")
    json.dump(result, open(jp, "w"), indent=2)

    print("\n===== phoneme drift (post → nearest pre-centroid), "
          f"{args.speaker}, {args.encoder} (train acc {tacc:.2f}) =====")
    print(f"{'phon':>5}{'ret_post':>10}{'ret_base':>10}{'→merge':>8}{'frac':>7}{'class':>13}")
    for p in sorted(phonemes, key=lambda p: per_ph[names[p]]["retention_post"]):
        d = per_ph[names[p]]
        print(f"{names[p]:>5}{d['retention_post']:>10.2f}{d['baseline_retention']:>10.2f}"
              f"{d['top_merge_into']:>8}{d['top_merge_frac']:>7.2f}{d['class']:>13}")
    print(f"\n  mean post-retention  tongue={result['mean_retention_tongue']}  "
          f"labial={result['mean_retention_labial']}  "
          f"(lower tongue ⇒ tongue phonemes moved more — supports H2)")
    print(f"  top mergers: " + ", ".join(f"{a}→{b}({f:.2f})" for a, b, f in mergers[:6]))
    print(f"[drift] wrote {jp}")

    # figures
    rng = np.random.default_rng(args.seed)
    ttl = f"{args.encoder} — {args.speaker} — phoneme q (○ pre / △ post), {args.method}"
    plot_tsne(qp, php, qo, pho, phonemes, names, args.method, args.seed, args.pca,
              args.perplexity, args.cap, rng,
              os.path.join(out_dir, f"gloss_drift_tsne_{args.encoder}{sfx}_s{args.seed}.png"),
              ttl)
    plot_drift(base, drift, phonemes, names,
               os.path.join(out_dir, f"gloss_drift_mat_{args.encoder}{sfx}_s{args.seed}.png"),
               f"{args.encoder} — {args.speaker} — phoneme drift (row-normalized)")
    print(f"[drift] done -> {out_dir}")


if __name__ == "__main__":
    main()
