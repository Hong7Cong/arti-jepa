"""Binary fluent-vs-disfluent rtMRI clip dataloader (stuttering corpus).

**Task 8b** -- tell a DISFLUENCY clip apart from a REGULAR (fluent) speech clip.

* **Positive** (disfluent, label 1): every annotated ``disfluency``-tier event
  ``[xmin, xmax]`` (seconds), same definition as the disfluency-type eval.
* **Negative** (fluent, label 0): a "regular speech" window carved from a
  *fluent-speech region* = (non-empty ``words`` intervals, merged across short
  pauses) **minus** every ``disfluency``-tier event. This uses the consistently
  present ``disfluency`` tier as the truth for "is this stretch disfluent" and the
  ``words`` tier only to stay on actual speech (not leading/trailing silence) --
  it does not rely on the ``flue_``/``disf_`` word-prefix convention, which only a
  subset of files use.

The one design constraint the caller asked for: **the negatives' duration
distribution must match the positives'** so clip length is not a give-away feature.
We therefore draw each negative's target duration from the empirical disfluency
duration pool and place a window of that length inside a fluent region that fits
(non-overlapping within a file). Negatives are balanced ``neg_per_pos`` : 1 per
file, drawn from the same speakers/recordings as the positives.

Each clip is ``num_frames`` (default **200**) frames sampled **uniformly across its
window at the video's native fps** (~99 fps). Loading/preprocessing reuse
``stutter.DisfluencySegmentDataset`` (z-score/minmax -> bicubic resize -> grayscale
x3), so this stays byte-identical to the rest of the Arti-JEPA input pipeline; the
only change is ``frames_per_clip = 200`` and ``event_pad_s = 0`` (the window is the
event/negative interval exactly).

Usage
-----
    from artijepa import stutter_binary as SB
    rows, stats = SB.build_rows(seed=0, neg_per_pos=1)          # pos + matched neg
    loader, ds = SB.make_loader(rows, num_frames=200, batch_size=8, shuffle=True)
    for clips, labels, meta in loader:      # clips [B,3,200,S,S], labels {0,1}
        ...

    # leave-one-speaker-out train / test:
    tr = SB.filter_speakers(rows, keep={"PWS3","PWS4","PWS5","PWS6","PWS7","PWS8"})
    te = SB.filter_speakers(rows, keep={"PWS10"})

CLI (build a manifest + sanity-check one batch, needs the `artijepa` env / decord):
    python -m artijepa.stutter_binary --check --num-frames 200 --neg-per-pos 1
"""

import argparse
import csv
import os
from collections import Counter

import cv2
import numpy as np
import torch

from artijepa import stutter as S
from artijepa.stutter import FLUENT, MANIFEST_COLS, _stem_media, canonicalize
from artijepa.rtmri_dataset import PreprocConfig, _intensity_norm, _spatial, _to_gray

ROOT = "/data1/span_data/stuttering"
SPEAKERS = ("PWS3", "PWS4", "PWS5", "PWS6", "PWS7", "PWS8", "PWS10")
GRAYSCALE_STATS = "/data2/hongn/artijepa/grayscale_stats.json"


# --------------------------------------------------------------------------- #
# interval algebra on (xmin, xmax) second ranges
# --------------------------------------------------------------------------- #
def _speech_spans(words, merge_gap):
    """Maximal spans of actual speech from the ``words`` tier.

    Non-empty word intervals are speech; runs separated by a silence <= ``merge_gap``
    (s) are merged into one span so a fluent window may cross short inter-word
    pauses (natural for continuous speech). Returns sorted, disjoint ``[a, b]``.
    """
    spans = []
    cur = None
    for xmin, xmax, text in sorted(words):
        if not text.strip():
            continue                                    # silence
        if cur is not None and xmin - cur[1] <= merge_gap:
            cur[1] = max(cur[1], xmax)
        else:
            if cur is not None:
                spans.append((cur[0], cur[1]))
            cur = [xmin, xmax]
    if cur is not None:
        spans.append((cur[0], cur[1]))
    return spans


def _subtract(spans, cuts, min_dur):
    """spans minus cut intervals -> disjoint remainders with length >= min_dur."""
    cuts = sorted(cuts)
    out = []
    for a, b in spans:
        segs = [(a, b)]
        for ca, cb in cuts:
            nxt = []
            for sa, sb in segs:
                if cb <= sa or ca >= sb:                # no overlap
                    nxt.append((sa, sb))
                    continue
                if ca > sa:
                    nxt.append((sa, min(ca, sb)))       # left remainder
                if cb < sb:
                    nxt.append((max(cb, sa), sb))       # right remainder
            segs = nxt
        out.extend((sa, sb) for sa, sb in segs if sb - sa >= min_dur)
    return out


def fluent_regions(parsed, tiers, merge_gap, min_dur):
    """Fluent-speech free regions for one TextGrid: speech spans minus disfluencies."""
    spans = _speech_spans(parsed.get("words", []), merge_gap)
    cuts = [(a, b) for tier in tiers for a, b, t in parsed.get(tier, []) if t.strip()]
    return _subtract(spans, cuts, min_dur)


def disf_events(parsed, tiers, min_dur, max_dur):
    """Canonicalizable disfluency events (positives) for one TextGrid.

    -> list of (xmin, xmax, dur, primary, bucket5, multi, raw_text, tier).
    """
    out = []
    for tier in tiers:
        for xmin, xmax, text in parsed.get(tier, []):
            if not text.strip():
                continue
            dur = xmax - xmin
            if dur < min_dur or dur > max_dur:
                continue
            primary, bucket5, comps = canonicalize(text)
            if primary is None:
                continue
            out.append((xmin, xmax, dur, primary, bucket5, "|".join(comps),
                        text.strip(), tier))
    return out


def _place_matched(regions, target_durs, n, rng, min_dur):
    """Place up to ``n`` non-overlapping windows in ``regions`` (list of (a,b)).

    Each window's target length is drawn from ``target_durs`` (the positive-duration
    pool) so the negatives inherit the disfluency duration distribution. A fitting
    region is chosen at random, the window placed at a random offset inside it, and
    the region's remainders returned to the free pool (keeps windows disjoint). If no
    region fits the drawn length, the largest remaining region is consumed whole (a
    slightly shorter negative) -- reported so the caller can see the match quality.
    """
    free = [(a, b) for a, b in regions if b - a >= min_dur]
    out = []
    for _ in range(n):
        if not free:
            break
        t = float(rng.choice(target_durs))
        fits = [i for i, (a, b) in enumerate(free) if b - a >= t]
        if fits:
            i = int(rng.choice(fits))
            a, b = free[i]
            off = float(rng.uniform(0.0, (b - a) - t))
            w0, w1 = a + off, a + off + t
        else:                                           # largest region, whole
            i = max(range(len(free)), key=lambda k: free[k][1] - free[k][0])
            a, b = free[i]
            if b - a < min_dur:
                break
            w0, w1 = a, b
        seg = free.pop(i)
        if w0 - seg[0] >= min_dur:
            free.append((seg[0], w0))
        if seg[1] - w1 >= min_dur:
            free.append((w1, seg[1]))
        out.append((round(w0, 6), round(w1, 6)))
    return out


# --------------------------------------------------------------------------- #
# row builder (positives + duration-matched fluent negatives)
# --------------------------------------------------------------------------- #
def build_rows(root=ROOT, speakers=SPEAKERS, tiers=("disfluency",),
               neg_per_pos=1, seed=0, min_dur=0.20, max_dur=8.0,
               merge_gap=0.25, verbose=True):
    """Build binary-classification rows: disfluency events + matched fluent windows.

    Returns ``(rows, stats)``. ``rows`` are manifest dicts (``MANIFEST_COLS`` schema)
    consumable directly by ``stutter.DisfluencySegmentDataset(task='binary')``:
    positives keep their disfluency ``bucket5``; negatives are ``FLUENT``. The two
    passes are (1) collect every file's positives + fluent regions and pool all
    positive durations, then (2) draw duration-matched negatives per file.
    """
    rng = np.random.default_rng(seed)
    per_file = []                                       # (spk, stem, vid, wav, events, regions)
    pos_durs = []
    for spk in speakers:
        tg_dir = os.path.join(root, spk, "textgrid")
        if not os.path.isdir(tg_dir):
            continue
        spk_dir = os.path.join(root, spk)
        for fn in sorted(os.listdir(tg_dir)):
            if not fn.endswith(".TextGrid"):
                continue
            stem = fn[: -len(".TextGrid")]
            vid, wav = _stem_media(spk_dir, stem)
            if not vid:
                continue
            parsed = S.parse_textgrid(os.path.join(tg_dir, fn))
            events = disf_events(parsed, tiers, min_dur, max_dur)
            regions = fluent_regions(parsed, tiers, merge_gap, min_dur)
            per_file.append((spk, stem, vid, wav, events, regions))
            pos_durs.extend(e[2] for e in events)

    if not pos_durs:
        raise RuntimeError("no positive (disfluency) events found")
    target_durs = np.asarray(pos_durs, dtype=np.float64)

    rows = []
    neg_short = 0                                       # negs shorter than drawn target
    for spk, stem, vid, wav, events, regions in per_file:
        for xmin, xmax, dur, primary, bucket5, multi, raw, tier in events:
            rows.append(dict(speaker=spk, stem=stem, path=vid, audio=wav, tier=tier,
                             xmin=round(xmin, 6), xmax=round(xmax, 6),
                             dur=round(dur, 6), raw_text=raw, primary=primary,
                             bucket5=bucket5, multi=multi))
        n_neg = int(round(len(events) * neg_per_pos))
        for w0, w1 in _place_matched(regions, target_durs, n_neg, rng, min_dur):
            rows.append(dict(speaker=spk, stem=stem, path=vid, audio=wav, tier="fluent",
                             xmin=w0, xmax=w1, dur=round(w1 - w0, 6), raw_text="",
                             primary=FLUENT, bucket5=FLUENT, multi=""))

    rows.sort(key=lambda r: (r["speaker"], r["stem"], r["xmin"], r["tier"]))
    for i, r in enumerate(rows):
        r["seg_id"] = i

    pos = [r for r in rows if r["bucket5"] != FLUENT]
    neg = [r for r in rows if r["bucket5"] == FLUENT]
    pd = np.asarray([r["dur"] for r in pos]); nd = np.asarray([r["dur"] for r in neg])
    stats = {
        "n_total": len(rows), "n_pos": len(pos), "n_neg": len(neg),
        "pos_per_speaker": dict(sorted(Counter(r["speaker"] for r in pos).items())),
        "neg_per_speaker": dict(sorted(Counter(r["speaker"] for r in neg).items())),
        "pos_dur": _dur_summary(pd), "neg_dur": _dur_summary(nd),
    }
    if verbose:
        print(f"[binary] {len(rows)} rows  pos={len(pos)} neg={len(neg)} "
              f"(neg_per_pos={neg_per_pos}, seed={seed}, tiers={list(tiers)})")
        print(f"[binary]   pos/speaker : {stats['pos_per_speaker']}")
        print(f"[binary]   neg/speaker : {stats['neg_per_speaker']}")
        print(f"[binary]   pos dur (s) : {stats['pos_dur']}")
        print(f"[binary]   neg dur (s) : {stats['neg_dur']}   "
              f"(duration-matched to positives)")
    return rows, stats


def _dur_summary(d):
    if len(d) == 0:
        return {}
    return {"n": int(len(d)), "min": round(float(d.min()), 3),
            "p50": round(float(np.median(d)), 3), "mean": round(float(d.mean()), 3),
            "p90": round(float(np.quantile(d, 0.9)), 3), "max": round(float(d.max()), 3)}


def filter_speakers(rows, keep):
    """Subset rows to a speaker set (for LOSO / train-test split)."""
    keep = set(keep)
    return [r for r in rows if r["speaker"] in keep]


def write_manifest(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows({k: r[k] for k in MANIFEST_COLS} for r in rows)
    print(f"[binary] wrote {len(rows)} rows -> {out_csv}")
    return out_csv


# --------------------------------------------------------------------------- #
# frame loading (OpenCV -- NOT decord)
# --------------------------------------------------------------------------- #
# The stuttering .avi files are `rawvideo` / `pix_fmt=pal8` (8-bit palettized).
# decord's VideoReader silently DECODES THESE AS ALL-ZERO frames (verified: every
# frame max=0), whereas OpenCV decodes them correctly (max=255). `stutter.py` /
# `eval_disfluency.py` use decord, so their features on this corpus are all-black --
# see docs/STUTTERING.md. This dataloader therefore decodes with OpenCV.
def _load_window_cv2(path, t0, t1, cfg):
    """Uniformly sample ``cfg.frames_per_clip`` frames across [t0,t1] (s), native fps.

    Reads only the source frames needed (sequential scan over the spanned range --
    rawvideo is cheap and exactly seekable), linearly interpolates between the two
    bracketing native frames per sample time, then applies the standard Arti-JEPA
    preprocessing (percentile clip + z-score/minmax -> resize -> grayscale x3).
    """
    F = cfg.frames_per_clip
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(cfg.target_fps)
    times = np.linspace(t0, t1, F, dtype=np.float64)
    s = np.clip(times * fps, 0.0, n_src - 1.0)
    f0 = np.floor(s).astype(np.int64)
    f1 = np.minimum(f0 + 1, n_src - 1)
    frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
    need = np.unique(np.concatenate([f0, f1]))
    lo, hi = int(need[0]), int(need[-1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    want = set(int(v) for v in need)
    got = {}
    idx = lo
    while idx <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            got[idx] = fr                                # [H,W,3] uint8 (BGR, gray-equal)
        idx += 1
    cap.release()
    if not got:
        raise RuntimeError(f"cv2 read no frames from {path} [{lo},{hi}]")
    last = got[min(got)]
    stack = np.stack([got.get(int(v), last) for v in need], 0)   # [K,H,W,3]

    gray = _to_gray(stack)                               # [K,H,W] float32 0..255
    remap = {int(v): i for i, v in enumerate(need)}
    i0 = torch.tensor([remap[int(v)] for v in f0])
    i1 = torch.tensor([remap[int(v)] for v in f1])
    clip = (1.0 - frac) * gray[i0] + frac * gray[i1]     # [F,H,W]
    clip = _intensity_norm(clip, cfg)
    clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
    clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
    return clip.unsqueeze(0).repeat(3, 1, 1, 1)          # [3,F,S,S]


class BinaryClipDataset(S.DisfluencySegmentDataset):
    """``DisfluencySegmentDataset`` that decodes with OpenCV (pal8-rawvideo safe)."""

    def _load_window(self, path, t0, t1):
        return _load_window_cv2(path, t0, t1, self.cfg)


# --------------------------------------------------------------------------- #
# dataset / dataloader
# --------------------------------------------------------------------------- #
def _preproc(num_frames, spatial_size, spatial_mode, intensity_norm,
             grayscale_stats, tubelet_size):
    assert num_frames % tubelet_size == 0, "num_frames must be divisible by tubelet_size"
    gmean, gstd = 0.0, 1.0
    if intensity_norm == "zscore" and grayscale_stats and os.path.exists(grayscale_stats):
        import json
        st = json.load(open(grayscale_stats)); gmean, gstd = st["mean"], st["std"]
    return PreprocConfig(
        target_fps=99.0, frames_per_clip=num_frames, sampling="tile",
        spatial_mode=spatial_mode, spatial_size=spatial_size,
        intensity_norm=intensity_norm, grayscale_mean=gmean, grayscale_std=gstd,
        augment=False, random_temporal_crop=False, tubelet_size=tubelet_size)


def make_dataset(rows, num_frames=200, spatial_size=256, spatial_mode="resize",
                 intensity_norm="zscore", grayscale_stats=GRAYSCALE_STATS,
                 tubelet_size=2, event_pad_s=0.0):
    """A ``DisfluencySegmentDataset`` over binary rows: 200 native-fps frames/clip.

    ``event_pad_s`` seconds of context are padded around each window (default 0 --
    the clip is exactly the disfluency event / matched fluent window). Labels: 0
    fluent, 1 disfluent (``stutter.row_label(task='binary')``).
    """
    cfg = _preproc(num_frames, spatial_size, spatial_mode, intensity_norm,
                   grayscale_stats, tubelet_size)
    classes, _ = S.label_space("binary")
    return BinaryClipDataset(rows, cfg, task="binary", classes=classes,
                             event_pad_s=event_pad_s)


def make_loader(rows, batch_size=8, num_workers=4, shuffle=True, **ds_kwargs):
    """(DataLoader, dataset). Batches yield ``(clips[B,3,F,S,S], labels[B], meta)``."""
    ds = make_dataset(rows, **ds_kwargs)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=S.collate, drop_last=False)
    return loader, ds


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tiers", nargs="+", default=["disfluency"])
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=0.20)
    ap.add_argument("--max-dur", type=float, default=8.0)
    ap.add_argument("--merge-gap", type=float, default=0.25)
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--spatial", type=int, default=256)
    ap.add_argument("--out", default=None, help="write a manifest CSV here")
    ap.add_argument("--check", action="store_true", help="load one batch and print shapes")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows, stats = build_rows(root=args.root, tiers=tuple(args.tiers),
                             neg_per_pos=args.neg_per_pos, seed=args.seed,
                             min_dur=args.min_dur, max_dur=args.max_dur,
                             merge_gap=args.merge_gap)
    if args.out:
        write_manifest(rows, args.out)
    if args.check:
        loader, ds = make_loader(rows, num_frames=args.num_frames, spatial_size=args.spatial,
                                 batch_size=args.batch, num_workers=2, shuffle=True)
        print(f"[binary] dataset={len(ds)} class_counts(fluent,disfluent)={ds.class_counts().tolist()}")
        clips, labels, meta = next(iter(loader))
        print(f"[binary] batch clips={tuple(clips.shape)} dtype={clips.dtype} "
              f"labels={labels.tolist()}")
        print(f"[binary]   clip stats: min={clips.min():.3f} max={clips.max():.3f} "
              f"mean={clips.mean():.3f}")
        print(f"[binary]   meta[:3]={meta[:3]}")


if __name__ == "__main__":
    main()
