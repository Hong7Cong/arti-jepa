"""Per-class t-SNE across glossectomy speakers + the usc_lss healthy reference speaker.

One panel PER CLASS; each point is one instance of that class, COLORED BY SPEAKER
(spk1/spk2/spk3 gloss + usc_s1), MARKER = condition (○ pre, △ post, □ usc healthy).
If a class's instances split into per-speaker clusters, its features are speaker-specific
-- the visual form of the Test-2 LOSO collapse (docs/GLOSS.md §9); if the gloss points
land on the usc_s1 (healthy) cloud, that class transfers speaker/appearance-invariantly.

Three label spaces, selected with ``--label-space`` (comma-list, all three by default —
they share one feature-collection pass, so asking for all three costs the same as one):

  * **vowels** (15 panels) -- vowel + diphthong phoneme identity, one panel per phoneme.
  * **place**  (8 panels)  -- consonant PLACE of articulation (Bilabial…Glottal).
  * **manner** (5 panels)  -- consonant MANNER (Plosive/Fricative/Affricate/Nasal/
                              Approximant).

These are the same label spaces as the segment-level Phase-3 ladder in RESULTS_gloss.md,
so the panels are the geometry behind those macro-F1 / κ numbers.

Panel captions stay minimal -- class name + n only. Two per-panel diagnostics go to STDOUT
(so the job log keeps them without cluttering the figure):
  * **sil(spk)** -- silhouette of that panel's points under the *speaker* labeling (cosine,
    on the PCA-reduced vectors, computed only when >=2 speakers have >=2 points). Positive =
    the class separates BY SPEAKER, i.e. speaker-specific features; ~0 or negative = speakers
    are interleaved, i.e. the class transfers.
  * **n per (class, speaker)** -- the balance behind each panel's n, after the --cap.

Representation:
  - **rep B (raw)**: the encoder's spatial-mean token vector -- probe-independent, so all
    four speakers sit on equal footing.
  - **rep A (trained-q, default)**: the attentive probe's AttentivePooler output (biased
    toward the probe's training corpus; pick which probe with --train-set).

usc_s1 comes from the SAME frozen encoder: for the annot16 probe it is the `test_lss`
split of its feature cache (all 684 usc_lss utts); for the usc_lss probe it is `test`.
Reads the frozen gloss caches eval_gloss already wrote (no re-extraction).

Usage:
    source dev_artiJEPA/scripts/_env.sh
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    # headline: all three label spaces, ckpt-215 attentive_lstm probe space, segment level
    python -m artijepa.tsne_gloss_vowels --encoder tssl256comb215 \
          --head attentive_lstm --rep A --per segment --include-usc
    python -m artijepa.tsne_gloss_vowels --include-usc --rep B --per segment  # raw geometry
    python -m artijepa.tsne_gloss_vowels --include-usc --label-space vowels   # vowels only
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

import artijepa.phonemes as P
from artijepa.eval_phoneme import load_config, load_probe, build_dataset, _tag
from artijepa.eval_gloss import _cond_manifest, CONDS, pick_probe
from artijepa.eval_gloss_indomain import _load_cache
from artijepa.tsne_phonemes import (pca_reduce, embed, PHON_CLASS, PLACE_CLASS,
                                    PLACE_ORDER, VOWEL_PHON, CONS_PHON,
                                    CONS_MANNER_ORDER)

# stable per-speaker colors + per-condition markers
SPK_COLOR = {"spk1": "#1f77b4", "spk2": "#ff7f0e", "spk3": "#2ca02c",
             "usc_s1": "#7f7f7f"}
COND_MARKER = {"pre": "o", "post": "^", "usc": "s"}

# --------------------------------------------------------------------------- #
# label spaces: (ordered panel labels, phoneme -> panel label, panel-title prefix)
# --------------------------------------------------------------------------- #
LABEL_SPACES = {
    "vowels": (list(VOWEL_PHON), lambda p: p if p in set(VOWEL_PHON) else None, "/{}/"),
    "place":  (list(PLACE_ORDER), lambda p: PLACE_CLASS.get(p), "{}"),
    "manner": (list(CONS_MANNER_ORDER),
               lambda p: PHON_CLASS.get(p) if p in set(CONS_PHON) else None, "{}"),
}
SPACE_ORDER = ["vowels", "place", "manner"]


def space_members(space):
    """Phoneme indices that belong to any panel of `space`."""
    _, group_of, _ = LABEL_SPACES[space]
    return {P.PHON2IDX[p] for p in P.ARPABET if group_of(p) is not None}


# --------------------------------------------------------------------------- #
# rep-A / rep-B token vectors, tagged by phoneme + speaker + condition
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _row_rep(fn, tok_idx, rep, pooler):
    """rep vectors [k,D] for temporal tokens tok_idx of one clip fn [T',S',D]."""
    sub = fn[tok_idx]                                   # [k,S',D]
    if rep == "B":
        return sub.mean(1)                             # [k,D] spatial mean
    q = pooler(torch.from_numpy(sub)).squeeze(1)        # [k,S',D]->[k,1,D]->[k,D]
    return q.numpy()


def collect(feats, labels, meta, speaker_of, cond, rep, pooler, per, vset):
    """-> list of (vec[D], phon_idx, speaker, cond) for in-`vset` tokens/segments."""
    recs = []
    by_utt = {}
    for n, (utt, chunk, _) in enumerate(meta):
        by_utt.setdefault(int(utt), []).append((int(chunk), n))
    for utt, rows in by_utt.items():
        rows.sort(key=lambda z: z[0])
        spk = speaker_of(utt)
        # concat rep-vectors + labels across chunks (token order)
        vseq, lseq = [], []
        for _, n in rows:
            fn = np.asarray(feats[n], dtype=np.float32)   # [T',S',D]
            lab = labels[n]
            tk = np.arange(len(lab))
            vseq.append(_row_rep(fn, tk, rep, pooler)); lseq.append(lab)
        vseq = np.concatenate(vseq, 0); lseq = np.concatenate(lseq, 0)
        if per == "token":
            for i in range(len(lseq)):
                if int(lseq[i]) in vset:
                    recs.append((vseq[i], int(lseq[i]), spk, cond))
        else:                                             # segment: pool equal-label runs
            i, T = 0, len(lseq)
            while i < T:
                lab = int(lseq[i]); j = i + 1
                while j < T and int(lseq[j]) == lab:
                    j += 1
                if lab in vset:
                    recs.append((vseq[i:j].mean(0), lab, spk, cond))
                i = j
    return recs


def gloss_recs(cfg, encoder, speaker, scratch, cache_root, rep, pooler, per, vset):
    recs = []
    for cond in CONDS:
        manifest, n = _cond_manifest(cond, speaker, scratch)
        c2 = copy.deepcopy(cfg)
        c2["data"]["manifest"] = manifest
        c2["data"]["pool_spatial"] = False
        c2["meta"]["tag"] = f"gloss_{cond}" + (f"_{speaker}" if speaker else "") + \
            f"_{encoder}"
        name, split = _tag(c2, "test")
        ds = build_dataset(c2, "test")[0]
        speaker_of = lambda utt, ds=ds: ds.rows[int(utt)].get("speaker", "gloss")
        feats, labels, meta = _load_cache(os.path.join(cache_root, name), split)
        r = collect(feats, labels, meta, speaker_of, cond, rep, pooler, per, vset)
        recs += r
        print(f"[vtsne] gloss {cond}: {n} utts -> {len(r)} {per}s")
    return recs


# --------------------------------------------------------------------------- #
# per-panel speaker silhouette (does this class split BY SPEAKER?)
# --------------------------------------------------------------------------- #
def _spk_silhouette(Xr, spk_lbl):
    """silhouette(speaker) on cosine geometry, or None if not computable."""
    uniq, cnt = np.unique(spk_lbl, return_counts=True)
    if (cnt >= 2).sum() < 2:
        return None
    m = np.isin(spk_lbl, uniq[cnt >= 2])
    if m.sum() < 3:
        return None
    from sklearn.metrics import silhouette_score
    try:
        return float(silhouette_score(Xr[m], spk_lbl[m], metric="cosine"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
def _figure(space, args, X, ph, spk, cnd, speakers, rng, out_dir, sfx):
    """One figure for one label space. Returns the written path."""
    panel_order, group_of, title_fmt = LABEL_SPACES[space]
    grp = np.array([group_of(P.IDX2PHON[int(i)]) for i in ph], dtype=object)
    panels = [c for c in panel_order if (grp == c).sum() >= args.min_per_class]
    if not panels:
        print(f"[vtsne] {space}: no class reaches --min-per-class {args.min_per_class}")
        return None
    print(f"[vtsne] {space}: {len(panels)} panels {panels}")

    ncol = min(4, len(panels)); nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.5 * nrow),
                             squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    sils, n_by_panel = {}, {}
    for k, c in enumerate(panels):
        ax = axes[k // ncol][k % ncol]; ax.axis("on")
        # balance per (class, speaker) so no speaker dominates the panel
        keep = []
        for s in speakers:
            si = np.where((grp == c) & (spk == s))[0]
            if len(si) > args.cap:
                si = rng.choice(si, args.cap, replace=False)
            keep.append(si)
        keep = np.concatenate(keep)
        Xr = pca_reduce(X[keep], args.pca) if args.pca > 0 else X[keep]
        xy = embed(Xr, args.method, args.seed, args.perplexity)
        sil = _spk_silhouette(Xr, spk[keep])
        sils[c] = sil
        for s in speakers:
            for cd in ("pre", "post", "usc"):
                m = (spk[keep] == s) & (cnd[keep] == cd)
                if m.any():
                    ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.6, linewidths=0,
                               color=SPK_COLOR[s], marker=COND_MARKER[cd])
        # caption stays minimal -- class + n. The per-speaker breakdown and sil(spk) go to
        # stdout (see below), so the figure reads cleanly and the numbers stay recoverable.
        ax.set_title(f"{title_fmt.format(c)}  n={len(keep)}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        n_by_panel[c] = {s: int((spk[keep] == s).sum()) for s in speakers}

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=SPK_COLOR[s], label=s)
               for s in speakers]
    handles += [plt.Line2D([0], [0], marker=COND_MARKER[c], ls="", color="k", label=c)
                for c in ("pre", "post", "usc") if (cnd == c).any()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{args.encoder} — per-{space[:-1] if space=='vowels' else space} "
                 f"{args.method} across speakers "
                 f"(rep {args.rep}: {args.train_set+'-trained q' if args.rep=='A' else 'raw'}, "
                 f"{args.per}-level)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fp = os.path.join(out_dir, f"tsne_gloss_{space}_rep{args.rep}_{args.per}_"
                      f"{args.method}_{args.encoder}{sfx}_s{args.seed}.png")
    fig.savefig(fp, dpi=155, bbox_inches="tight"); plt.close(fig)
    # everything the captions no longer show, on stdout so the job log keeps it
    have = [v for v in sils.values() if v is not None]
    print(f"[vtsne] {space} sil(spk): " +
          " ".join(f"{c}={'na' if sils[c] is None else format(sils[c], '+.3f')}"
                   for c in panels) +
          (f"  | mean {np.mean(have):+.3f} median {np.median(have):+.3f}"
           if have else ""))
    print(f"[vtsne] {space} n per (class, speaker): " +
          " ".join(f"{c}[" + "/".join(str(n_by_panel[c][s]) for s in speakers) + "]"
                   for c in panels) + f"   (order: {'/'.join(speakers)})")
    print(f"[vtsne] wrote {fp}")
    return fp


def _figure_bygroup(space, args, X, ph, spk, cnd, speakers, rng, out_dir, sfx):
    """ONE plot, all speakers pooled, colored by class label."""
    panel_order, group_of, title_fmt = LABEL_SPACES[space]
    grp = np.array([group_of(P.IDX2PHON[int(i)]) for i in ph], dtype=object)
    panels = [c for c in panel_order if (grp == c).sum() >= args.min_per_class]
    keep = []
    for c in panels:
        for s in speakers:
            si = np.where((grp == c) & (spk == s))[0]
            if len(si) > args.cap:
                si = rng.choice(si, args.cap, replace=False)
            keep.append(si)
    keep = np.concatenate(keep)
    Xr = pca_reduce(X[keep], args.pca) if args.pca > 0 else X[keep]
    xy = embed(Xr, args.method, args.seed, args.perplexity)
    cmap = matplotlib.colormaps["tab20"]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    for i, c in enumerate(panels):
        m = grp[keep] == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, linewidths=0,
                       color=cmap(i % 20),
                       label=f"{title_fmt.format(c)}  ({int(m.sum())})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              frameon=False, title=space, markerscale=1.6)
    ax.set_title(f"{args.encoder} — {space} {args.method}, all speakers pooled, "
                 f"colored by class (rep {args.rep}: "
                 f"{args.train_set+'-trained q' if args.rep=='A' else 'raw'}, "
                 f"{args.per}-level, n={len(keep)})", fontsize=12)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"tsne_gloss_{space}_byclass_rep{args.rep}_"
                      f"{args.per}_{args.method}_{args.encoder}{sfx}_s{args.seed}.png")
    fig.savefig(fp, dpi=155, bbox_inches="tight"); plt.close(fig)
    print(f"[vtsne] wrote {fp}")
    return fp


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="tssl256comb100")
    ap.add_argument("--train-set", default="annot16", choices=["annot16", "usc_lss"],
                    help="which trained probe defines rep-A q space + the usc_lss source "
                         "(annot16 -> test_lss = full usc_lss; usc_lss -> test)")
    ap.add_argument("--head", default="attentive",
                    choices=["attentive", "attentive_lstm"],
                    help="probe head whose AttentivePooler defines rep-A q (attentive_lstm "
                         "shares the same spatial pooler as attentive; rep B is head-agnostic)")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="which trained-probe loss variant to load (default ce)")
    ap.add_argument("--config", default="dev_artiJEPA/configs/eval_gloss.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speaker", default=None, help="restrict gloss to spk1|spk2|spk3")
    ap.add_argument("--rep", default="A", choices=["A", "B"],
                    help="A=trained attentive-pooler q (per --train-set); B=raw encoder "
                         "(probe-independent, recommended for cross-speaker comparison)")
    ap.add_argument("--per", default="token", choices=["token", "segment"])
    ap.add_argument("--label-space", default="vowels,place,manner",
                    help="comma-list of " + "|".join(SPACE_ORDER) + " (default: all "
                         "three; they share one feature-collection pass)")
    ap.add_argument("--mode", default="by_speaker",
                    choices=["by_speaker", "by_vowel", "by_class"],
                    help="by_speaker: one panel/class colored by speaker; "
                    "by_class: ONE plot, all speakers pooled, colored by class "
                    "(by_vowel = legacy alias of by_class)")
    ap.add_argument("--include-usc", action="store_true",
                    help="overlay the usc_lss training speaker (usc_s1) as a reference")
    ap.add_argument("--cap", type=int, default=200,
                    help="max instances per (class, speaker) for balance")
    ap.add_argument("--min-per-class", "--min-per-vowel", dest="min_per_class",
                    type=int, default=12,
                    help="drop classes with fewer than this many instances overall")
    ap.add_argument("--fig-suffix", default="",
                    help="extra tag appended to the output filename before _s<seed>, so a "
                         "restyled figure lands beside the previous one instead of "
                         "overwriting it (e.g. --fig-suffix mincap)")
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--arti-out", default="/scratch1/hongn/artijepa")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCHPAD", "/tmp"))
    args = ap.parse_args()

    spaces = [s.strip() for s in args.label_space.split(",") if s.strip()]
    bad = [s for s in spaces if s not in LABEL_SPACES]
    if bad:
        raise SystemExit(f"[vtsne] unknown --label-space {bad}; pick from {SPACE_ORDER}")
    spaces = [s for s in SPACE_ORDER if s in spaces]

    eval_dir = os.path.join(args.arti_out, "eval")
    out_dir = os.path.join(eval_dir, "gloss"); os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(args.config)
    cache_root = cfg["meta"]["cache_dir"]
    # ONE collection pass over the union of every requested space's members
    vset = set().union(*(space_members(s) for s in spaces))

    pt = pick_probe(args.encoder, args.seed, eval_dir, args.train_set,
                    head=args.head, loss=args.loss)
    probe, ck = load_probe(pt, device="cpu")
    pooler = probe.pooler if args.rep == "A" else None
    cfg["encoder"]["spec"] = ck["encoder_spec"]
    print(f"[vtsne] rep {args.rep} <- {os.path.basename(pt)}  per={args.per}  "
          f"spaces={spaces}  ({len(vset)} phonemes)")

    recs = gloss_recs(cfg, args.encoder, args.speaker, args.scratch, cache_root,
                      args.rep, pooler, args.per, vset)
    if args.include_usc:
        # usc_s1 lives in the probe's own feature cache: as the `test_lss` split for the
        # annot16 combined probe (all 684 usc_lss utts), else `test` (usc_lss-only probe).
        usc_dir = os.path.join(cache_root, ck["feature_tag"])
        usc_split = ("test_lss"
                     if os.path.exists(os.path.join(usc_dir, "test_lss.feats.npy"))
                     else "test")
        uf, ul, um = _load_cache(usc_dir, usc_split)
        ur = collect(uf, ul, um, lambda utt: "usc_s1", "usc", args.rep, pooler,
                     args.per, vset)
        recs += ur
        print(f"[vtsne] usc_s1 <- {os.path.basename(usc_dir)}/{usc_split}: "
              f"{len(ur)} {args.per}s")

    X = np.stack([r[0] for r in recs]).astype(np.float32)
    ph = np.array([r[1] for r in recs]); spk = np.array([r[2] for r in recs])
    cnd = np.array([r[3] for r in recs])
    rng = np.random.default_rng(args.seed)
    speakers = [s for s in SPK_COLOR if (spk == s).any()]
    print(f"[vtsne] N={len(recs)} {args.per}s, speakers={speakers}")

    sfx = (f"_{args.speaker}" if args.speaker else "") + \
          (f"_{args.head}" if args.head != "attentive" else "") + \
          ("_uscref" if args.include_usc else "") + \
          (f"_{args.fig_suffix.lstrip('_')}" if args.fig_suffix else "")

    fn = _figure if args.mode == "by_speaker" else _figure_bygroup
    for space in spaces:
        fn(space, args, X, ph, spk, cnd, speakers, rng, out_dir, sfx)


if __name__ == "__main__":
    main()
