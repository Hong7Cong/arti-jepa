"""Frozen-encoder PHONEME-CLIP classification eval (Arti-JEPA downstream, Phase 3).

Phases 1-2 slid a fixed window over each utterance and predicted a phoneme label
*per temporal token* (frame-level sequence task, kappa / PER). **Phase 3 is a
different task:** one sample = one whole phoneme, cut from the video by its
start-end alignment, classified with a **single** label (clip-level). We ask how
well a frozen encoder separates phoneme categories at five granularities:

    * vowcons     -- 2-way  vowel vs consonant     (the coarsest cut; all non-sil)
    * vowels      -- 15-way vowel identity        (monophthongs + diphthongs)
    * consonants  -- 25-way consonant identity
    * manner      -- 5-way  manner of articulation (consonants only)
    * place       -- 8-way  place of articulation  (consonants only)

`vowcons` is the only task that mixes both phoneme families in one label space, so
it is the natural floor of the ladder: it asks whether the frozen features carry
the open-vs-constricted vocal-tract distinction at all, before asking for identity.
Being binary it also reports ROC-AUC / average precision (consonant = positive),
which the multiclass tasks leave out.

Head = `attentive_lstm` in CLIP mode: V-JEPA's AttentivePooler pools the S'
spatial tokens per temporal step (learned, *where* in the vocal tract is
phonetic), a bi-LSTM runs over the phoneme's VARIABLE-LENGTH temporal-token
sequence, and its **last hidden state is the single vector representing the
phoneme** (both directions), fed to a linear head -> class logits. Metrics:
macro & weighted F1 / precision / recall + per-class + confusion matrix.

Data: the SAME combined Annot-16 + usc_lss manifest as Phase 1-2
(`configs/eval_phoneme_annot16_combined.yaml` data block). Train on the 14
Annot-16 train speakers, model-select on val (sub030), report `test` (in-domain
sub043) AND `test_lss` (cross-domain usc_s1). Frozen encoder -> the feature cache
is task-agnostic (all non-sil phonemes, one clip each): extracted ONCE per
encoder x split, then the four tasks are derived by filtering + relabeling.

Run:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.eval_phoneme_groups \
        --config dev_artiJEPA/configs/eval_phoneme_groups.yaml \
        --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt \
        --tag tssl256comb215 --seed 0
"""

import argparse
import hashlib
import json
import math
import mmap
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from decord import VideoReader, cpu

from artijepa import phonemes as P
from artijepa.eval_phoneme import load_frozen_encoder, _preproc
from artijepa.rtmri_dataset import _intensity_norm, _spatial, _to_gray


# --------------------------------------------------------------------------- #
# phoneme groupings (verbatim from artijepa/tsne_phonemes.py PHON_CLASS /
# PLACE_CLASS; inlined so the training job needs no matplotlib import).
# --------------------------------------------------------------------------- #
PHON_CLASS = {
    "aa": "Vowel", "ae": "Vowel", "ah": "Vowel", "ao": "Vowel", "eh": "Vowel",
    "er": "Vowel", "ih": "Vowel", "iy": "Vowel", "uh": "Vowel", "uw": "Vowel",
    "aw": "Diphthong", "ay": "Diphthong", "ey": "Diphthong", "ow": "Diphthong",
    "oy": "Diphthong",
    "b": "Plosive", "d": "Plosive", "g": "Plosive", "k": "Plosive", "p": "Plosive",
    "t": "Plosive",
    "dh": "Fricative", "f": "Fricative", "h": "Fricative", "hh": "Fricative",
    "s": "Fricative", "sh": "Fricative", "th": "Fricative", "v": "Fricative",
    "z": "Fricative", "zh": "Fricative",
    "ch": "Affricate", "jh": "Affricate",
    "m": "Nasal", "n": "Nasal", "ng": "Nasal",
    "l": "Approximant", "r": "Approximant", "w": "Approximant", "y": "Approximant",
    "sil": "Silence",
}
PLACE_CLASS = {
    "b": "Bilabial", "p": "Bilabial", "m": "Bilabial",
    "f": "Labiodental", "v": "Labiodental",
    "th": "Dental", "dh": "Dental",
    "t": "Alveolar", "d": "Alveolar", "s": "Alveolar", "z": "Alveolar",
    "n": "Alveolar", "l": "Alveolar", "r": "Alveolar",
    "ch": "Postalveolar", "jh": "Postalveolar", "sh": "Postalveolar", "zh": "Postalveolar",
    "y": "Palatal",
    "k": "Velar", "g": "Velar", "ng": "Velar", "w": "Velar",
    "h": "Glottal", "hh": "Glottal",
}
CONS_MANNER_ORDER = ["Plosive", "Fricative", "Affricate", "Nasal", "Approximant"]
PLACE_ORDER = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Postalveolar",
               "Palatal", "Velar", "Glottal"]
# binary vowel-vs-consonant: consonant is class 1 (the "positive" class for AUC/AP)
VOWCONS_ORDER = ["Vowel", "Consonant"]
VOWEL_CLASSES = {"Vowel", "Diphthong"}
CONS_CLASSES = set(CONS_MANNER_ORDER)
VOWEL_PHON = [p for p in P.ARPABET if PHON_CLASS.get(p) in VOWEL_CLASSES]   # 15
CONS_PHON = [p for p in P.ARPABET if PHON_CLASS.get(p) in CONS_CLASSES]     # 25


def build_tasks():
    """-> {task: (members set of phoneme-idx, {phon_idx: class_idx}, [class names])}.

    All four tasks read the same task-agnostic clip cache (phoneme index per clip)
    and select/relabel: `members` filters clips, the map turns a phoneme index into
    the task's class index, `names` gives the confusion-matrix axis order."""
    tasks = {}
    # binary: the only task spanning BOTH families -- every non-sil phoneme is a
    # member, relabelled to its family (vowel/diphthong -> 0, consonant -> 1).
    tasks["vowcons"] = (
        {P.PHON2IDX[p] for p in VOWEL_PHON + CONS_PHON},
        {**{P.PHON2IDX[p]: 0 for p in VOWEL_PHON},
         **{P.PHON2IDX[p]: 1 for p in CONS_PHON}},
        list(VOWCONS_ORDER))
    # identity tasks: class == the phoneme itself
    tasks["vowels"] = (
        {P.PHON2IDX[p] for p in VOWEL_PHON},
        {P.PHON2IDX[p]: i for i, p in enumerate(VOWEL_PHON)}, list(VOWEL_PHON))
    tasks["consonants"] = (
        {P.PHON2IDX[p] for p in CONS_PHON},
        {P.PHON2IDX[p]: i for i, p in enumerate(CONS_PHON)}, list(CONS_PHON))
    # grouped consonant tasks
    tasks["manner"] = (
        {P.PHON2IDX[p] for p in CONS_PHON},
        {P.PHON2IDX[p]: CONS_MANNER_ORDER.index(PHON_CLASS[p]) for p in CONS_PHON},
        list(CONS_MANNER_ORDER))
    tasks["place"] = (
        {P.PHON2IDX[p] for p in CONS_PHON},
        {P.PHON2IDX[p]: PLACE_ORDER.index(PLACE_CLASS[p]) for p in CONS_PHON},
        list(PLACE_ORDER))
    return tasks


# --------------------------------------------------------------------------- #
# per-phoneme clip dataset (one non-sil segment = one sample)
# --------------------------------------------------------------------------- #
def _flatten_segments(manifest, split, target_fps, tubelet, frames_per_clip, limit=None):
    """-> list of clip specs [(row_idx, path, fps, start_s, end_s, phon_idx,
    valid_frames)], one per NON-SIL segment, sorted by (row, start) so a worker
    streaming the list reuses one open video per utterance. valid_frames is the
    number of ~target_fps frames the phoneme spans, clamped to [tubelet, F].
    `limit` (debug) caps clips per split (evenly strided so all classes appear)."""
    import csv
    rows = [r for r in csv.DictReader(open(manifest))
            if split is None or r.get("split") == split]
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {manifest}")
    specs = []
    for ri, r in enumerate(rows):
        fps = float(r["fps"])
        for ph, s, e in P.load_gold_segments(r["phoneme_json"]):
            if ph == P.SIL:
                continue
            pi = P.PHON2IDX.get(ph)
            if pi is None:
                continue
            n = int(round((e - s) * target_fps))
            n = max(tubelet, min(frames_per_clip, n))   # >=1 token, <=one window
            specs.append((ri, r["path"], fps, float(s), float(e), pi, n))
    if limit and len(specs) > limit:
        step = len(specs) / float(limit)
        specs = [specs[int(k * step)] for k in range(limit)]
    return specs, rows


class _ClipExtractDS(torch.utils.data.Dataset):
    """Loads one phoneme clip on demand: the [start,end] interval sampled to
    `valid_frames` frames at ~target_fps (linear temporal interp), then edge-padded
    to the encoder's fixed window F. Returns (clip [3,F,S,S], phon_idx, valid_frames).

    Per-worker 1-entry WHOLE-VIDEO cache: the first clip of a video decodes ALL its
    gray frames once ([n_src,H,W]); every phoneme of that utterance then slices the
    cached frames -- no per-clip re-decode. Specs are sorted by (row,start) so a
    worker streams an utterance's phonemes back-to-back against one cached decode."""

    def __init__(self, specs, cfg):
        self.specs, self.cfg = specs, cfg
        self.F = cfg.frames_per_clip
        self._path = None
        self._gray = None            # cached [n_src,H,W] float gray frames

    def __len__(self):
        return len(self.specs)

    def _frames(self, path):
        if path != self._path:
            vr = VideoReader(path, num_threads=2, ctx=cpu(0))
            self._gray = _to_gray(vr.get_batch(np.arange(len(vr))).asnumpy())  # [n_src,H,W]
            self._path = path
        return self._gray

    def __getitem__(self, i):
        ri, path, fps, start, end, pi, n = self.specs[i]
        cfg = self.cfg
        F = self.F
        # ~target_fps samples spanning the phoneme, then edge-pad the tail to F.
        t = np.linspace(start, end, n, dtype=np.float64)         # seconds
        s = t * fps                                              # source frame pos
        if n < F:
            s = np.concatenate([s, np.full(F - n, s[-1])])      # edge-repeat pad
        gray = self._frames(path)                               # cached [n_src,H,W]
        n_src = gray.shape[0]
        s = np.clip(s, 0.0, n_src - 1.0)
        f0 = np.floor(s).astype(np.int64)
        f1 = np.minimum(f0 + 1, n_src - 1)
        frac = torch.from_numpy((s - f0).astype("float32")).view(-1, 1, 1)
        clip = (1.0 - frac) * gray[torch.from_numpy(f0)] + frac * gray[torch.from_numpy(f1)]
        clip = _intensity_norm(clip, cfg)                        # [F,H,W]
        clip = _spatial(clip, cfg.spatial_mode, cfg.spatial_size)
        clip = (clip - cfg.grayscale_mean) / (cfg.grayscale_std + 1e-6)
        clip = clip.unsqueeze(0).repeat(3, 1, 1, 1)              # [3,F,S,S]
        return clip, pi, n


def _extract_collate(batch):
    clips = torch.stack([b[0] for b in batch], 0)
    phon = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    vframes = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return clips, phon, vframes


# --------------------------------------------------------------------------- #
# feature cache (ragged: only the valid temporal tokens per clip)
# --------------------------------------------------------------------------- #
def _tag(cfg, split):
    d, ec = cfg["data"], cfg["encoder"]
    hd = {"spec": ec.get("spec", "pretrained"), "key": ec.get("key", "auto"),
          "sz": d["spatial_size"], "fpc": d["frames_per_clip"],
          "fps": d.get("target_fps", 50.0), "manifest": d["manifest"],
          "task": "phgroups"}
    if d.get("max_clips"):                       # debug subset -> its own cache
        hd["max_clips"] = int(d["max_clips"])
    h = hashlib.sha1(json.dumps(hd, sort_keys=True).encode()).hexdigest()[:10]
    tag = cfg["meta"].get("tag") or ec.get("spec", "pretrained")
    return f"{os.path.basename(str(tag))}phg_{h}", split


def clip_rows(cfg, split):
    """-> i64 [N]: the manifest ROW index each cached clip came from.

    Written by `extract` as `<split>.row.npy`; recomputed from the manifest when the
    cache predates it (the spec list is deterministic — same manifest/fps/geometry ⇒
    same clip order — so this is exact, not an approximation). Used to slice a
    condition's predictions per speaker/session without re-extracting
    (`eval_gloss_groups.py`, docs/GLOSS.md §12.7)."""
    d = cfg["data"]
    name, split = _tag(cfg, split)
    rp = os.path.join(cfg["meta"]["cache_dir"], name, f"{split}.row.npy")
    if os.path.exists(rp):
        return np.load(rp)
    specs, _ = _flatten_segments(d["manifest"], split, d.get("target_fps", 50.0),
                                 d["tubelet_size"], d["frames_per_clip"],
                                 limit=d.get("max_clips"))
    row = np.array([sp[0] for sp in specs], dtype=np.int64)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    np.save(rp, row)
    return row


@torch.no_grad()
def extract(encoder, cfg, split, device, dtype):
    """Ragged clip-feature cache. Files under <cache>/<tag>/<split>.*:
      feats  f16 [sum_tok, S', D]   valid temporal tokens of every clip, concatenated
      off    i64 [N+1]              clip i tokens = feats[off[i]:off[i+1]]
      phon   i64 [N]                phoneme index (0..40) of each clip
      row    i64 [N]                manifest row index of each clip (see `clip_rows`)
    Returns (feats_memmap, off, phon)."""
    d = cfg["data"]
    name, split = _tag(cfg, split)
    cdir = os.path.join(cfg["meta"]["cache_dir"], name)
    os.makedirs(cdir, exist_ok=True)
    fp, op, pp = (os.path.join(cdir, f"{split}.{x}.npy") for x in ("feats", "off", "phon"))
    if all(os.path.exists(x) for x in (fp, op, pp)):
        print(f"[phg] cache hit {split} <- {cdir}", flush=True)
        return np.load(fp, mmap_mode="r"), np.load(op), np.load(pp)

    tub = d["tubelet_size"]
    F = d["frames_per_clip"]
    pc = _preproc(cfg)
    specs, _ = _flatten_segments(d["manifest"], split, d.get("target_fps", 50.0),
                                 tub, F, limit=d.get("max_clips"))
    N = len(specs)
    # valid temporal tokens per clip, known upfront from valid_frames -> offsets
    vtok = np.array([int(math.ceil(n / tub)) for *_, n in specs], dtype=np.int64)
    off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(vtok, out=off[1:])
    sum_tok = int(off[-1])
    phon = np.array([sp[5] for sp in specs], dtype=np.int64)
    np.save(os.path.join(cdir, f"{split}.row.npy"),
            np.array([sp[0] for sp in specs], dtype=np.int64))
    print(f"[phg] extract {split}: {N} clips, {sum_tok} valid tokens "
          f"(mean {sum_tok / max(1, N):.2f}/clip), F={F} tub={tub}", flush=True)

    ds = _ClipExtractDS(specs, pc)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=d.get("batch_size", 32), shuffle=False,
        num_workers=d.get("num_workers", 6), collate_fn=_extract_collate)
    Tp = F // tub
    feats = None
    pos_tok = 0            # write cursor into the ragged feats memmap
    done = 0
    t0 = time.time()
    for bi, (clips, _phon, vframes) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            tok = encoder.backbone(clips)                        # [B,Ntot,D]
        B, Ntot, D = tok.shape
        tok = tok.float().reshape(B, Tp, Ntot // Tp, D)          # [B,T',S',D]
        S = tok.shape[2]
        if feats is None:
            feats = np.lib.format.open_memmap(
                fp, mode="w+", dtype=np.float16, shape=(sum_tok, S, D))
        tok = tok.cpu().numpy().astype(np.float16)
        for b in range(B):
            nt = int(math.ceil(int(vframes[b]) / tub))           # valid tokens
            feats[pos_tok:pos_tok + nt] = tok[b, :nt]            # leading (real) tokens
            pos_tok += nt
        done += B
        if bi % 20 == 0:
            feats.flush()
            try:
                feats._mmap.madvise(mmap.MADV_DONTNEED)          # bound RSS (see eval_phoneme)
            except (AttributeError, OSError):
                pass
            print(f"[phg]  extract {split} {done}/{N} ({time.time()-t0:.0f}s)", flush=True)
    if feats is None:
        raise RuntimeError(f"no clips extracted for split={split}")
    feats.flush()
    assert pos_tok == sum_tok, (pos_tok, sum_tok)
    np.save(op, off)
    np.save(pp, phon)
    print(f"[phg] extracted {split}: {feats.shape} in {time.time()-t0:.0f}s -> {cdir}",
          flush=True)
    return np.load(fp, mmap_mode="r"), off, phon


# --------------------------------------------------------------------------- #
# probe: attentive_lstm clip classifier (last hidden = phoneme vector)
# --------------------------------------------------------------------------- #
class ClipProbe(nn.Module):
    """AttentivePooler over S' per temporal token -> bi-LSTM over the phoneme's
    variable-length token sequence -> LAST hidden state (fwd+bwd) = the single
    phoneme vector -> linear head -> class logits [B,C]."""

    def __init__(self, dim, num_classes, hidden=512, layers=2, heads=8, dropout=0.1):
        super().__init__()
        from artijepa._vendor.src.models.attentive_pooler import AttentivePooler
        self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                      num_heads=heads, mlp_ratio=4.0, depth=1)
        self.norm = nn.LayerNorm(dim)
        self.layers = layers
        self.rnn = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden, num_classes))

    def forward(self, x, lens, return_vec=False):                # x: [B,T,S',D]
        B, T, S, D = x.shape
        q = self.pooler(x.reshape(B * T, S, D)).squeeze(1)       # [B*T,D]
        h = self.norm(q.reshape(B, T, D))                        # [B,T,D]
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lens.cpu(), batch_first=True, enforce_sorted=False)
        _out, (hn, _cn) = self.rnn(packed)                       # hn: [2*L,B,H]
        # last-layer forward + backward hidden states -> the phoneme vector [B,2H]
        vec = torch.cat([hn[-2], hn[-1]], dim=1)
        logits = self.head(vec)                                  # [B,C]
        # rep A for the clustering readout: the 2H phoneme vector the linear head reads
        return (logits, vec) if return_vec else logits


class _ClipDS(torch.utils.data.Dataset):
    """One task's clips from the ragged cache: (feat [t,S',D] f32, class_idx).
    `members`/`remap` select and relabel the task-agnostic cache.

    `preload`: if the task's clips fit in `budget_gb`, copy their f16 slices into a
    RAM list once (avoids re-reading the memmap every epoch -- the val split is read
    40x, and random-access reads over the ~100 GB spatial cache are the probe I/O
    wall). Falls back to on-demand memmap reads when too big to preload."""

    def __init__(self, feats, off, phon, members, remap, preload=False, budget_gb=40.0):
        self.feats, self.off = feats, off
        self.idx = [i for i in range(len(phon)) if int(phon[i]) in members]
        self.labels = [remap[int(phon[i])] for i in self.idx]
        self.lengths = [int(off[i + 1] - off[i]) for i in self.idx]   # temporal tokens
        self.mem = None
        if preload and self.idx:
            S, D = feats.shape[1], feats.shape[2]
            ntok = int(sum(off[i + 1] - off[i] for i in self.idx))
            gb = ntok * S * D * 2 / 1e9
            if gb <= budget_gb:
                self.mem = [np.array(feats[off[i]:off[i + 1]], dtype=np.float16)
                            for i in self.idx]
                print(f"[phg]   preloaded {len(self.idx)} clips ({gb:.1f} GB) to RAM",
                      flush=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self.mem is not None:
            f = self.mem[j].astype(np.float32)
        else:
            i = self.idx[j]
            f = np.array(self.feats[self.off[i]:self.off[i + 1]], dtype=np.float32)
        return torch.from_numpy(f), self.labels[j]


def _clip_collate(batch):
    feats, labs = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.int64)
    T = int(lens.max())
    x = torch.zeros((len(feats), T) + tuple(feats[0].shape[1:]), dtype=torch.float32)
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f
    return x, torch.tensor(labs, dtype=torch.int64), lens


class _LenBucketSampler(torch.utils.data.Sampler):
    """Batch clips of SIMILAR token length together.

    Phoneme clips run 1-16 temporal tokens (mean ~3.9). `_clip_collate` pads to the
    batch max, so random batching pads nearly every batch to ~16 -> ~4x wasted compute
    and ~2 GB activation tensors. Sorting by length before batching makes the padded
    size ~= the true size; batch ORDER is reshuffled each epoch so training still sees
    a fresh batch sequence."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = list(lengths); self.bs = int(batch_size)
        self.shuffle, self.seed, self.epoch = shuffle, int(seed), 0

    def __iter__(self):
        order = np.argsort(np.asarray(self.lengths), kind="stable")
        batches = [order[i:i + self.bs].tolist()
                   for i in range(0, len(order), self.bs)]
        if self.shuffle:
            np.random.RandomState(self.seed + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


@torch.no_grad()
def _predict(clf, ds, device, bs, workers, amp):
    clf.eval()
    # data already in RAM -> forking workers only risks OOM (they inherit the parent's
    # multi-GB preload) and buys nothing; length-bucket to avoid padding waste.
    nw = 0 if ds.mem is not None else workers
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=_LenBucketSampler(ds.lengths, bs, shuffle=False),
        num_workers=nw, collate_fn=_clip_collate)
    ps, ts, sc = [], [], []
    for x, y, lens in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = clf(x, lens)
        logits = logits.float()
        ps.append(logits.argmax(-1).cpu().numpy())
        # softmax scores kept for threshold-free binary metrics (AUC/AP); a few MB
        # even on test_lss, and the multiclass tasks simply ignore them.
        sc.append(torch.softmax(logits, -1).cpu().numpy())
        ts.append(y.numpy())
    return np.concatenate(ts), np.concatenate(ps), np.concatenate(sc)


def kappa_from_cm(cm):
    """Cohen's kappa from a confusion matrix (rows=true, cols=pred).

    kappa is a pure function of the matrix, so the Phase-3 anchor rows can be
    back-filled from the `confusion_matrix` already stored in phgroups_*.json
    without re-running anything (docs/GLOSS.md §12.5)."""
    M = np.asarray(cm, dtype=np.float64)
    n = M.sum()
    if n <= 0:
        return float("nan")
    po = np.trace(M) / n
    pe = float((M.sum(0) * M.sum(1)).sum()) / (n * n)
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else float("nan")


def _metrics(y_true, y_pred, n_classes, names, scores=None):
    """Macro/weighted P/R/F1 + per-class + confusion matrix (+ kappa).

    `scores` (optional, [N, n_classes] softmax) adds the threshold-free binary
    metrics for a 2-class task -- `roc_auc` and `average_precision`, both with
    class 1 (`vowcons`: Consonant) as the positive class. Multiclass callers may
    pass it or not; it is ignored there."""
    from sklearn.metrics import (precision_recall_fscore_support,
                                  confusion_matrix, accuracy_score)
    labels = list(range(n_classes))
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    pp, rr, ff, ss = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    extra = {}
    if n_classes == 2 and scores is not None and len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        pos = np.asarray(scores)[:, 1]
        extra["roc_auc"] = round(float(roc_auc_score(y_true, pos)), 4)
        extra["average_precision"] = round(
            float(average_precision_score(y_true, pos)), 4)
        # the binary headline alongside macro-F1: F1 of the positive class alone
        extra["f1_binary"] = round(float(ff[1]), 4)
    return {
        "n": int(len(y_true)),
        **extra,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "kappa": round(kappa_from_cm(cm), 4),
        "f1_macro": round(float(f_ma), 4), "f1_weighted": round(float(f_w), 4),
        "precision_macro": round(float(p_ma), 4), "recall_macro": round(float(r_ma), 4),
        "precision_weighted": round(float(p_w), 4), "recall_weighted": round(float(r_w), 4),
        "per_class": {names[i]: {"precision": round(float(pp[i]), 4),
                                 "recall": round(float(rr[i]), 4),
                                 "f1": round(float(ff[i]), 4), "support": int(ss[i])}
                      for i in range(n_classes)},
        "confusion_matrix": cm.tolist(), "class_names": names,
    }


def train_task(cfg, task, splits, device):
    """Train + select + evaluate one grouping task. `splits` = {name:(feats,off,phon)}.
    Returns a result dict with val/test/test_lss metrics and the best-val history."""
    pc = cfg["probe"]
    members, remap, names = task
    n_classes = len(names)
    # TRAIN stays on the memmap + parallel workers -- the known-good path (~2 min/epoch).
    # (Preloading train to RAM was tried and REVERTED: it forces num_workers=0 to avoid
    # the fork/OOM of a 40-60 GB parent, and serial loading cost more than the I/O it
    # saved. Keeping it identical to the videomae/videomae_rtmri runs also preserves
    # cross-encoder comparability.) val/test/test_lss ARE preloaded: they are small and
    # re-read (val every epoch), so it is a free win with no OOM risk.
    budget = pc.get("preload_gb", 20.0)
    ds_tr = _ClipDS(*splits["train"], members, remap)
    ds_va = _ClipDS(*splits["val"], members, remap, preload=True, budget_gb=budget)
    dim = int(splits["train"][0].shape[-1])
    clf = ClipProbe(dim, n_classes, hidden=pc.get("hidden", 512),
                    layers=pc.get("layers", 2), heads=pc.get("heads", 8),
                    dropout=pc.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=pc.get("lr", 1e-3),
                            weight_decay=pc.get("wd", 0.01))
    ce = nn.CrossEntropyLoss()
    bs = pc.get("batch_size", 128)
    workers = pc.get("workers", 6)
    amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs, warmup, base = pc.get("epochs", 40), pc.get("warmup", 4), pc.get("lr", 1e-3)
    # plain shuffled batching + parallel workers (identical to the videomae/rtmri runs)
    loader = torch.utils.data.DataLoader(
        ds_tr, batch_size=bs, shuffle=True, num_workers=workers,
        collate_fn=_clip_collate, persistent_workers=(workers > 0), drop_last=False)
    print(f"[phg:{n_classes}c] clips train/val = {len(ds_tr)}/{len(ds_va)}; "
          f"classes={n_classes}", flush=True)
    best_f1 = -1.0; best_state = None; best_epoch = 0; history = []
    for ep in range(epochs):
        lr = base * (ep + 1) / max(1, warmup) if ep < warmup else \
            0.5 * base * (1 + np.cos(np.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr
        clf.train(); run = nb = 0; t0 = time.time()
        for x, y, lens in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = ce(clf(x, lens), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss); nb += 1
        yt, yp, ysc = _predict(clf, ds_va, device, bs, workers, amp)
        vm = _metrics(yt, yp, n_classes, names, ysc)
        history.append({"epoch": ep + 1, "train_loss": round(run / max(1, nb), 4),
                        "val_f1_macro": vm["f1_macro"], "val_acc": vm["accuracy"],
                        "mins": round((time.time() - t0) / 60, 2)})
        if vm["f1_macro"] > best_f1:            # snapshot best-val weights; eval later
            best_f1 = vm["f1_macro"]; best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            best_val = vm
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[phg e{ep+1}/{epochs}] loss={run/max(1,nb):.3f} lr={lr:.2e} "
                  f"val_f1={vm['f1_macro']:.3f} best_val_f1={best_f1:.3f} "
                  f"({history[-1]['mins']:.1f} min)", flush=True)
    # free the RAM-preloaded train set + its loader/workers before the eval preloads
    del loader, ds_tr, ds_va
    import gc
    gc.collect()
    # ONE eval of test + test_lss on the best-val weights (not every improving epoch)
    if best_state is not None:
        clf.load_state_dict(best_state)
    best = {"epoch": best_epoch, "val_f1_macro": best_f1, "val": best_val,
            "history": history, "n_classes": n_classes}
    for nm in ("test", "test_lss"):
        if nm in splits:
            ds = _ClipDS(*splits[nm], members, remap, preload=True, budget_gb=budget)
            yt, yp, ysc = _predict(clf, ds, device, bs, workers, amp)
            best[nm] = _metrics(yt, yp, n_classes, names, ysc)
            del ds; gc.collect()
    best["_state"] = best_state          # popped + written to .pt by run()
    return best


# --------------------------------------------------------------------------- #
def run(cfg):
    meta = cfg["meta"]; os.makedirs(meta["out"], exist_ok=True)
    seed = meta.get("seed", 0); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        meta.get("dtype", "float16").lower(), torch.float32)
    cfg["data"]["pool_spatial"] = False              # attentive head needs the S' grid
    print(f"[phg] device={device} dtype={dtype} tag={meta.get('tag')}", flush=True)

    encoder = load_frozen_encoder(cfg, device)
    split_names = ["train", "val", "test"] + list(cfg["data"].get("extra_test_splits", []))
    splits = {nm: extract(encoder, cfg, nm, device, dtype) for nm in split_names}
    del encoder; torch.cuda.empty_cache()

    tasks = build_tasks()
    only = cfg["probe"].get("tasks") or list(tasks)
    stem = os.path.join(meta["out"],
                        f"phgroups_{os.path.basename(str(meta.get('tag','pretrained')))}_s{seed}")
    out = {"encoder": cfg["encoder"].get("spec", "pretrained"),
           "tag": meta.get("tag"), "seed": seed,
           "spatial_size": cfg["data"]["spatial_size"],
           "frames_per_clip": cfg["data"]["frames_per_clip"],
           "target_fps": cfg["data"].get("target_fps", 50.0),
           "head": "attentive_lstm_clip", "tasks": {}}
    # A task-subset run (e.g. --tasks vowcons on top of an already-complete file)
    # MERGES into the existing JSON for this tag+seed instead of truncating it:
    # tasks in `only` are recomputed, every other previously-written task is kept.
    if os.path.exists(stem + ".json"):
        try:
            prev = json.load(open(stem + ".json")).get("tasks", {})
        except (ValueError, OSError) as e:
            print(f"[phg] WARNING: ignoring unreadable {stem}.json ({e})", flush=True)
            prev = {}
        keep = {k: v for k, v in prev.items() if k not in only}
        if keep:
            out["tasks"].update(keep)
            print(f"[phg] merging into existing JSON; keeping {sorted(keep)}", flush=True)
    for tname in only:
        print(f"\n===== TASK: {tname} =====", flush=True)
        t0 = time.time()
        res = train_task(cfg, tasks[tname], splits, device)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        state = res.pop("_state", None)
        out["tasks"][tname] = res
        line = {k: res.get(k, {}).get("f1_macro") for k in ("val", "test", "test_lss")
                if isinstance(res.get(k), dict)}
        print(f"[phg] {tname} f1_macro  {line}", flush=True)
        # best-val probe weights per TASK (mirrors eval_phoneme.py's save_probe), so the
        # trained head is reusable for confusion-matrix / t-SNE / re-scoring without
        # retraining -- the frozen features are reproducible from meta['feature_tag'].
        if state is not None and meta.get("save_probe", True):
            pc = cfg["probe"]
            wp = f"{stem}_{tname}.pt"
            torch.save({
                "probe_state": state, "probe_kind": "attentive_lstm_clip",
                "task": tname, "class_names": tasks[tname][2],
                "num_classes": res["n_classes"],
                "dim": int(splits["train"][0].shape[-1]),
                "hidden": pc.get("hidden", 512), "layers": pc.get("layers", 2),
                "heads": pc.get("heads", 8), "dropout": pc.get("dropout", 0.1),
                "encoder_spec": cfg["encoder"].get("spec", "pretrained"),
                "encoder_checkpoint": cfg["encoder"].get("checkpoint"),
                "feature_tag": _tag(cfg, "train")[0], "seed": seed,
                "best_epoch": res.get("epoch"), "manifest": cfg["data"].get("manifest"),
                "spatial_size": cfg["data"]["spatial_size"],
                "frames_per_clip": cfg["data"]["frames_per_clip"],
                "target_fps": cfg["data"].get("target_fps", 50.0),
                "metrics": {k: res.get(k) for k in ("val", "test", "test_lss")},
            }, wp)
            print(f"[phg] wrote probe weights {wp}", flush=True)
        json.dump(out, open(stem + ".json", "w"), indent=2)   # incremental: never lose a task
        ndone = sum(t in out["tasks"] for t in only)
        print(f"[phg] wrote {stem}.json ({ndone}/{len(only)} tasks this run, "
              f"{len(out['tasks'])} total)", flush=True)

    print("\n===== PHONEME-GROUPS RESULT (f1_macro) =====", flush=True)
    for tname, res in out["tasks"].items():
        row = " ".join(
            f"{k}={res.get(k, {}).get('f1_macro', '-')}" +
            (f"(auc={res[k]['roc_auc']})" if "roc_auc" in res.get(k, {}) else "")
            for k in ("val", "test", "test_lss") if k in res)
        print(f"  {tname:12s} {row}", flush=True)
    return out


def load_probe(path, device="cpu"):
    """Reconstruct a trained per-task ClipProbe from a `phgroups_*_<task>.pt`.

    Returns (probe.eval(), meta_dict). meta carries task/class_names/num_classes,
    encoder_spec + feature_tag (so the frozen features are reproducible), best_epoch,
    seed and the val/test/test_lss metrics. Mirrors eval_phoneme.load_probe."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    clf = ClipProbe(ck["dim"], ck["num_classes"], hidden=ck["hidden"],
                    layers=ck["layers"], heads=ck["heads"], dropout=ck["dropout"])
    clf.load_state_dict(ck["probe_state"])
    return clf.to(device).eval(), ck


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default=None, help="checkpoint path -> encoder.spec")
    ap.add_argument("--model", default=None,
                    help="'videomae' (3-D) or an image-baseline timm name -> encoder.type")
    ap.add_argument("--checkpoint", default=None,
                    help="VideoMAE repo-format weights -> encoder.checkpoint (rt-MRI ckpt-214)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tasks", default=None,
                    help="comma list subset of vowcons,vowels,consonants,manner,place "
                         "(a subset run merges into the existing JSON for tag+seed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="DEBUG: cap clips per split (own cache); smoke-test the pipeline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["meta"]["seed"] = args.seed
    if args.dtype is not None:
        cfg["meta"]["dtype"] = args.dtype
    if args.model is not None:
        if args.model.lower().startswith("videomae"):
            cfg["encoder"]["type"] = "videomae"
            cfg["encoder"]["model"] = ("MCG-NJU/videomae-large"
                                       if args.model.lower() == "videomae" else args.model)
        else:
            cfg["encoder"]["type"] = "image_baseline"
            cfg["encoder"]["model"] = args.model
    if args.encoder is not None:
        cfg["encoder"]["spec"] = args.encoder
    if args.checkpoint is not None:
        cfg["encoder"]["checkpoint"] = args.checkpoint
    if args.tag is not None:
        cfg["meta"]["tag"] = args.tag
    if args.tasks is not None:
        cfg["probe"]["tasks"] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.limit is not None:
        cfg["data"]["max_clips"] = args.limit
    run(cfg)


if __name__ == "__main__":
    main()
