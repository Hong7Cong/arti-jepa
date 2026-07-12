"""Stuttering corpus: TextGrid parsing, disfluency-type canonicalization,
manifest building, and a segment-level dataset (Arti-JEPA eval Task 8).

`/data1/span_data/stuttering/PWS{3,4,5,6,7,8,10}` -- 7 persons-who-stutter (PWS),
104x104 rtMRI @ ~99 fps (same geometry as usc_lss), with `.TextGrid` annotations
paired 1:1 to `avi/<stem>.avi` and `wav/<stem>.wav`.

Two disfluency tiers hold events as `<phoneme>_<type>` interval labels:
  * **disfluency**  (primary): ~2100 events. Dominant types block / rep / pro,
    with rarer osci / revert / filler / abandon and many compound `a+b` labels.
  * **disfluency2** (secondary/overlapping): ~126 events, ~99% rep.

Unlike the usc_lss per-token phoneme task, this is **segment classification**: each
labeled interval `[xmin, xmax]` (seconds) is one example; we pool the frozen
encoder's tokens over that window into a single vector and predict the disfluency
type. Alignment is in **seconds**, so the 99-fps video needs no special-casing --
the loader uniformly samples `frames_per_clip` frames spanning the (padded) event.

The canonical eval setup mirrors the phoneme eval: **attentive probe, 256px**.
"""

import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from decord import VideoReader, cpu

from artijepa.rtmri_dataset import _intensity_norm, _spatial, _to_gray, PreprocConfig


# --------------------------------------------------------------------------- #
# disfluency-type canonicalization
# --------------------------------------------------------------------------- #
# The seven annotated disfluency types (after typo repair). `block/rep/pro` are
# the canonical majority; the rest are rare.
CANON_TYPES = ["block", "rep", "pro", "osci", "revert", "filler", "abandon"]

# Observed typos / spelling variants -> canonical single component.
_SYNONYMS = {
    "blcok": "block", "blok": "block",
    "repo": "rep", "red": "rep", "ep": "rep", "wordrep": "rep",
    "fille": "filler",
}

# 5-way bucket used for the primary head (rare types fold into "other").
BUCKET5 = {
    "block": "block", "rep": "rep", "pro": "pro", "osci": "osci",
    "revert": "other", "filler": "other", "abandon": "other",
}
FLUENT = "fluent"          # label for sampled non-disfluent (empty) intervals


def _norm_component(c):
    """One `+`-separated component -> canonical type str, or None if unknown."""
    c = c.strip().lower().rstrip("?").strip()
    c = _SYNONYMS.get(c, c)
    return c if c in CANON_TYPES else None


def canonicalize(text):
    """`<phoneme>_<type>` label text -> (primary, bucket5, components).

    ``primary``   : first canonical component (7-type space), or None.
    ``bucket5``   : primary folded into {block,rep,pro,osci,other}, or None.
    ``components``: ordered unique canonical components (the multi-label set).

    Type = substring after the first `_` (TODO_eval spec); compound `a+b` keeps
    its primary component for the single-label head and the full set for the
    multi-label variant. Returns (None, None, []) when nothing canonicalizes.
    """
    text = (text or "").strip()
    if not text:
        return None, None, []
    typ = text.split("_", 1)[1] if "_" in text else text
    comps, seen = [], set()
    for raw in typ.split("+"):
        c = _norm_component(raw)
        if c and c not in seen:
            seen.add(c)
            comps.append(c)
    if not comps:
        return None, None, []
    primary = comps[0]
    return primary, BUCKET5[primary], comps


# --------------------------------------------------------------------------- #
# TextGrid parsing (no praatio/textgrid dependency -- Praat "long" text format)
# --------------------------------------------------------------------------- #
_ITEM_SPLIT = re.compile(r"item \[\d+\]:")
_NAME = re.compile(r'name = "((?:[^"\\]|\\.)*)"')
_INTERVAL = re.compile(
    r"intervals \[\d+\]:\s*"
    r"xmin = ([\d.eE+-]+)\s*"
    r"xmax = ([\d.eE+-]+)\s*"
    r'text = "((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def parse_textgrid(path):
    """Praat long-format TextGrid -> {tier_name: [(xmin, xmax, text), ...]}.

    Only IntervalTiers are returned; each interval keeps its raw (untrimmed) text.
    """
    txt = open(path, encoding="utf-8", errors="replace").read()
    tiers = {}
    for block in _ITEM_SPLIT.split(txt)[1:]:
        m = _NAME.search(block)
        if not m:
            continue
        name = m.group(1)
        ivs = [(float(a), float(b), t) for a, b, t in _INTERVAL.findall(block)]
        # a tier name can appear once; if duplicated, concatenate deterministically
        tiers.setdefault(name, []).extend(ivs)
    return tiers


def _stem_media(spk_dir, stem):
    """(video, audio) absolute paths for a TextGrid stem, '' if absent."""
    vid = os.path.join(spk_dir, "avi", stem + ".avi")
    wav = os.path.join(spk_dir, "wav", stem + ".wav")
    return (vid if os.path.exists(vid) else ""), (wav if os.path.exists(wav) else "")


# --------------------------------------------------------------------------- #
# manifest builder
# --------------------------------------------------------------------------- #
MANIFEST_COLS = [
    "seg_id", "speaker", "stem", "path", "audio", "tier",
    "xmin", "xmax", "dur", "raw_text", "primary", "bucket5", "multi",
]


def build_manifest(root="/data1/span_data/stuttering", out_csv=None,
                   speakers=("PWS3", "PWS4", "PWS5", "PWS6", "PWS7", "PWS8", "PWS10"),
                   tiers=("disfluency", "disfluency2"),
                   fluent_per_file=0, min_dur=0.10, max_dur=8.0, verbose=True):
    """Scan every TextGrid, emit one manifest row per labeled disfluency event.

    ``fluent_per_file > 0`` also samples that many empty (non-disfluent) intervals
    per file from the primary ``disfluency`` tier -- labeled ``fluent`` -- to seed
    the binary fluent-vs-disfluent baseline. ``min_dur``/``max_dur`` (seconds)
    filter out zero-length and pathologically long intervals.
    """
    out_csv = out_csv or os.path.join(root, "disfluency_manifest.csv")
    rows, dropped = [], Counter()
    for spk in speakers:
        spk_dir = os.path.join(root, spk)
        tg_dir = os.path.join(spk_dir, "textgrid")
        if not os.path.isdir(tg_dir):
            if verbose:
                print(f"[stutter] skip {spk}: no textgrid/ dir")
            continue
        for fn in sorted(os.listdir(tg_dir)):
            if not fn.endswith(".TextGrid"):
                continue
            stem = fn[: -len(".TextGrid")]
            vid, wav = _stem_media(spk_dir, stem)
            if not vid:
                dropped["no_video"] += 1
                continue
            parsed = parse_textgrid(os.path.join(tg_dir, fn))
            for tier in tiers:
                for xmin, xmax, text in parsed.get(tier, []):
                    dur = xmax - xmin
                    if not text.strip():
                        continue
                    if dur < min_dur or dur > max_dur:
                        dropped["bad_dur"] += 1
                        continue
                    primary, bucket5, comps = canonicalize(text)
                    if primary is None:
                        dropped["uncanon"] += 1
                        continue
                    rows.append({
                        "speaker": spk, "stem": stem, "path": vid, "audio": wav,
                        "tier": tier, "xmin": round(xmin, 6), "xmax": round(xmax, 6),
                        "dur": round(dur, 6), "raw_text": text.strip(),
                        "primary": primary, "bucket5": bucket5,
                        "multi": "|".join(comps),
                    })
            if fluent_per_file > 0:
                rows.extend(_sample_fluent(parsed.get("disfluency", []), spk, stem,
                                           vid, wav, fluent_per_file, min_dur, max_dur))
    rows.sort(key=lambda r: (r["speaker"], r["stem"], r["xmin"], r["tier"]))
    for i, r in enumerate(rows):
        r["seg_id"] = i
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    if verbose:
        by_spk = Counter(r["speaker"] for r in rows)
        by_b5 = Counter(r["bucket5"] for r in rows)
        print(f"[stutter] wrote {len(rows)} segments -> {out_csv}")
        print(f"[stutter]   per-speaker : {dict(sorted(by_spk.items()))}")
        print(f"[stutter]   per-bucket5 : {dict(by_b5.most_common())}")
        print(f"[stutter]   dropped     : {dict(dropped)}")
    return out_csv


def _sample_fluent(disf_intervals, spk, stem, vid, wav, k, min_dur, max_dur):
    """Deterministically pick up to k empty disfluency intervals as fluent negs."""
    cands = [(a, b) for a, b, t in disf_intervals
             if not t.strip() and min_dur <= (b - a) <= max_dur]
    cands.sort(key=lambda z: (z[1] - z[0]), reverse=True)   # longest, stable
    out = []
    for a, b in cands[:k]:
        out.append({
            "speaker": spk, "stem": stem, "path": vid, "audio": wav,
            "tier": "disfluency", "xmin": round(a, 6), "xmax": round(b, 6),
            "dur": round(b - a, 6), "raw_text": "", "primary": FLUENT,
            "bucket5": FLUENT, "multi": "",
        })
    return out


# --------------------------------------------------------------------------- #
# label spaces
# --------------------------------------------------------------------------- #
def label_space(task):
    """(classes, field) for a task name.

    ``type5``  : block/rep/pro/osci/other        (bucket5)
    ``type3``  : block/rep/pro                    (bucket5, rare dropped)
    ``binary`` : fluent/disfluent                 (derived; needs fluent rows)
    """
    if task == "type5":
        return ["block", "rep", "pro", "osci", "other"], "bucket5"
    if task == "type3":
        return ["block", "rep", "pro"], "bucket5"
    if task == "binary":
        return ["fluent", "disfluent"], "bucket5"
    raise ValueError(f"unknown disfluency task {task!r}")


def row_label(row, task, classes):
    """Map a manifest row to a class index for ``task``, or None to drop it."""
    if task == "binary":
        return 0 if row["bucket5"] == FLUENT else 1
    val = row["bucket5"]
    if val == FLUENT or val not in classes:      # fluent negs unused for type tasks
        return None
    return classes.index(val)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class DisfluencySegmentDataset(torch.utils.data.Dataset):
    """One labeled disfluency interval -> a fixed-length rtMRI clip + class label.

    Frames are sampled uniformly across the padded event window `[xmin - pad,
    xmax + pad]` (seconds) and linearly interpolated onto `frames_per_clip`
    positions, so events of any duration map to a canonical clip. Preprocessing is
    the standard Arti-JEPA path (z-score / minmax -> bicubic resize -> grayscale
    x3), identical to `usc_lss`.
    """

    def __init__(self, rows, cfg: PreprocConfig, task, classes, event_pad_s=0.15):
        super().__init__()
        assert cfg.tubelet_size and cfg.frames_per_clip % cfg.tubelet_size == 0
        self.cfg = cfg
        self.task = task
        self.classes = list(classes)
        self.event_pad_s = float(event_pad_s)
        self.rows, self.labels = [], []
        for r in rows:
            y = row_label(r, task, self.classes)
            if y is None:
                continue
            self.rows.append(r)
            self.labels.append(y)
        if not self.rows:
            raise ValueError(f"No disfluency rows for task={task!r}")
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        return np.bincount(self.labels, minlength=self.num_classes)

    def _load_window(self, path, t0, t1):
        """Uniformly sample F frames across [t0, t1] (s) with linear interp."""
        cfg = self.cfg
        F = cfg.frames_per_clip
        vr = VideoReader(path, num_threads=2, ctx=cpu(0))
        n_src = len(vr)
        fps = float(vr.get_avg_fps())
        # source frame positions (float) for the F uniformly spaced sample times
        times = np.linspace(t0, t1, F, dtype=np.float64)
        s = np.clip(times * fps, 0.0, n_src - 1.0)
        f0 = np.floor(s).astype(np.int64)
        f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
        need = np.unique(np.concatenate([f0, f1]))
        remap = {int(v): i for i, v in enumerate(need)}
        gray = _to_gray(vr.get_batch(need).asnumpy())           # [K,H,W]
        i0 = torch.tensor([remap[int(v)] for v in f0])
        i1 = torch.tensor([remap[int(v)] for v in f1])
        clip = (1.0 - frac) * gray[i0] + frac * gray[i1]        # [F,H,W]
        clip = _intensity_norm(clip, cfg)
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        return clip.unsqueeze(0).repeat(3, 1, 1, 1)             # [3,F,S,S]

    def __getitem__(self, i):
        row = self.rows[i]
        t0 = max(0.0, float(row["xmin"]) - self.event_pad_s)
        t1 = max(t0 + 1e-3, float(row["xmax"]) + self.event_pad_s)
        clip = self._load_window(row["path"], t0, t1)
        return {"clip": clip, "label": int(self.labels[i]),
                "seg": int(row["seg_id"]), "speaker": row["speaker"]}


def collate(batch):
    clips = torch.stack([b["clip"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    meta = [(b["seg"], b["speaker"]) for b in batch]
    return clips, labels, meta


def read_manifest(path, speakers=None, tiers=None):
    """Load manifest rows, optionally filtered by speaker set / tier set."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if speakers is not None:
        rows = [r for r in rows if r["speaker"] in set(speakers)]
    if tiers is not None:
        rows = [r for r in rows if r["tier"] in set(tiers)]
    return rows


# --------------------------------------------------------------------------- #
# metrics (no sklearn dependency)
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(y_true, y_pred, num_classes, class_names=None):
    """macro-F1, balanced accuracy, accuracy, per-class P/R/F1, confusion matrix."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)          # true count per class
    pred_pos = cm.sum(0).astype(np.float64)         # predicted count per class
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(pred_pos > 0, tp / pred_pos, np.nan)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    present = support > 0                            # classes with >=1 true example
    macro_f1 = float(np.nanmean(np.where(present, f1, np.nan))) if present.any() else 0.0
    bal_acc = float(np.nanmean(recall[present])) if present.any() else 0.0
    acc = float(tp.sum() / max(1, cm.sum()))
    names = class_names or [str(i) for i in range(num_classes)]
    per_class = {
        names[i]: {
            "precision": round(float(precision[i]), 4) if not np.isnan(precision[i]) else None,
            "recall": round(float(recall[i]), 4) if not np.isnan(recall[i]) else None,
            "f1": round(float(f1[i]), 4), "support": int(support[i]),
        } for i in range(num_classes)
    }
    return {
        "macro_f1": round(macro_f1, 4), "balanced_acc": round(bal_acc, 4),
        "accuracy": round(acc, 4), "n": int(cm.sum()),
        "per_class": per_class, "confusion": cm.tolist(),
    }
