"""Offline articulator (+ MRI frame) cache for AC-JEPA (aucjepa_plans_new.md §0.5, M0).

Each ``usc_lss/articulators/usc_s1_<NN>_mview.mat`` is one ~35 s **session** holding,
all at 100 Hz and FRAME-EXACT with each other:

    AUDIO  16 kHz                                    (unused here)
    IMAGE  [104,104,T] uint8   the rtMRI frames
    Bilabial / Alveolar / Palatal / Velum / Pharyngeal / Larynx
           [T,1] float64       the six constriction signals  -> arti-6

Because the frames and the articulators live in the SAME file at the SAME rate, there
is NO fps mismatch and NO trimming offset (the make-or-break risk the plan flagged in
§1.1 is eliminated by construction). We cache, per session:

    <out>/<stem>.arti.npy    [T,6] float16   arti-6 (ARTICULATORS order)
    <out>/<stem>.image.npy   [T,104,104] uint8   MRI frames (T-major)
    <out>/meta.json          dim=6, arti_rate_hz=100, per-dim z-score corpus stats,
                             articulator names, per-session n_frames

and write a manifest CSV (stem, paths, n_frames, fps, duration_s, split) that the
``RTMRIArtiDataset`` / trainers consume. This runs in the plain ``artijepa`` env
(scipy only -- no transformers, unlike the trashed audio cache).

    PYTHONPATH=.:dev_artiJEPA python -m artijepa.arti_cache \
        --mat-dir /scratch1/hongn/usc_lss/articulators \
        --out /scratch1/hongn/artijepa/arti_feats/usc_lss \
        --manifest /scratch1/hongn/artijepa/arti_manifest.csv
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from artijepa.arti_cond import ARTICULATORS, ARTI_DIM

ARTI_RATE_HZ = 100.0          # SRATE of every signal in the .mat sessions
DEFAULT_MAT = "/scratch1/hongn/usc_lss/articulators"
DEFAULT_OUT = "/scratch1/hongn/artijepa/arti_feats/usc_lss"


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_session(mat_path):
    """Load one ``*_mview.mat`` -> (arti [T,6] float32, image [T,H,W] uint8, rate).

    The articulator order is forced to ``ARTICULATORS`` (frozen 6-D index order).
    Raises if a required field is missing or the lengths disagree.
    """
    import scipy.io as sio

    m = sio.loadmat(mat_path)
    data = m["data"]                                       # struct array [1, n]
    names = [str(np.asarray(data[0, i]["NAME"]).squeeze()) for i in range(data.shape[1])]
    name2i = {n: i for i, n in enumerate(names)}

    if "IMAGE" not in name2i:
        raise KeyError(f"{_stem(mat_path)}: no IMAGE field (have {names})")
    img = np.asarray(data[0, name2i["IMAGE"]]["SIGNAL"])   # [H, W, T] uint8
    if img.ndim != 3:
        raise ValueError(f"{_stem(mat_path)}: IMAGE ndim {img.ndim} != 3")
    image = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))     # -> [T, H, W]
    T = image.shape[0]

    cols = []
    for a in ARTICULATORS:
        if a not in name2i:
            raise KeyError(f"{_stem(mat_path)}: missing articulator {a!r} (have {names})")
        sig = np.asarray(data[0, name2i[a]]["SIGNAL"], dtype=np.float32).reshape(-1)
        if sig.shape[0] != T:
            raise ValueError(f"{_stem(mat_path)}: {a} len {sig.shape[0]} != IMAGE T {T}")
        cols.append(sig)
    arti = np.stack(cols, axis=1).astype(np.float32)       # [T, 6]
    return arti, image.astype(np.uint8), ARTI_RATE_HZ


def build_arti_cache(mat_dir=DEFAULT_MAT, out_dir=DEFAULT_OUT, manifest=None,
                     limit=None, stats_split="train", val_frac=0.15, test_frac=0.15,
                     seed=0, save_image=True):
    """Extract every session, cache arti(+image) .npy, write meta.json + manifest.

    Splits are SESSION-disjoint (usc_lss is a single speaker ``usc_s1``, so a
    speaker-disjoint split is impossible -- see plan §1.3; this is the documented
    fallback). Per-dim z-score stats are accumulated over the ``stats_split``
    sessions only. Re-running is safe (skips already-cached .npy, recomputes stats).
    """
    os.makedirs(out_dir, exist_ok=True)
    mats = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    if limit:
        mats = mats[:limit]
    if not mats:
        raise FileNotFoundError(f"no .mat sessions in {mat_dir}")

    # deterministic session-disjoint split
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mats))
    n_test = int(round(test_frac * len(mats)))
    n_val = int(round(val_frac * len(mats)))
    split_of = {}
    for rank, idx in enumerate(order):
        if rank < n_test:
            split_of[idx] = "test"
        elif rank < n_test + n_val:
            split_of[idx] = "val"
        else:
            split_of[idx] = "train"

    rows, sums, sq, cnt, dim = [], np.zeros(ARTI_DIM, np.float64), \
        np.zeros(ARTI_DIM, np.float64), 0, ARTI_DIM
    n_done = n_fail = 0
    for i, mp in enumerate(mats):
        stem = _stem(mp)
        arti_p = os.path.join(out_dir, stem + ".arti.npy")
        img_p = os.path.join(out_dir, stem + ".image.npy")
        split = split_of[i]
        try:
            if os.path.exists(arti_p) and (not save_image or os.path.exists(img_p)):
                arti = np.load(arti_p).astype(np.float32)
                T = arti.shape[0]
            else:
                arti, image, _ = read_session(mp)
                T = arti.shape[0]
                np.save(arti_p, arti.astype(np.float16))
                if save_image:
                    np.save(img_p, image)
                n_done += 1
            if split == stats_split:
                sums += arti.sum(0); sq += (arti * arti).sum(0); cnt += T
            rows.append({
                "stem": stem, "mat_path": mp,
                "image_npy": img_p if save_image else "",
                "arti_npy": arti_p, "n_frames": int(T),
                "fps": ARTI_RATE_HZ, "duration_s": T / ARTI_RATE_HZ, "split": split,
            })
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[arti] FAIL {stem}: {type(e).__name__}: {e}")
        if i % 10 == 0:
            print(f"[arti] {i}/{len(mats)} (new {n_done}, fail {n_fail})")

    mean = sums / max(1, cnt)
    var = np.maximum(sq / max(1, cnt) - mean ** 2, 1e-8)
    std = np.sqrt(var)
    meta = {
        "source": "usc_lss/articulators *_mview.mat", "dim": int(dim),
        "articulators": ARTICULATORS, "arti_rate_hz": ARTI_RATE_HZ,
        "stats_split": stats_split, "stats_n_frames": int(cnt),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if manifest:
        cols = ["stem", "mat_path", "image_npy", "arti_npy", "n_frames", "fps",
                "duration_s", "split"]
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        from collections import Counter
        print(f"[arti] manifest -> {manifest}  "
              f"{dict(Counter(r['split'] for r in rows))}")
    print(f"[arti] done -> {out_dir}  ({len(rows)} sessions, dim={dim}, "
          f"rate {ARTI_RATE_HZ} Hz, z-score over {cnt} frames / {stats_split})")
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-dir", default=DEFAULT_MAT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default="/scratch1/hongn/artijepa/arti_manifest.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats-split", default="train")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-image", action="store_true",
                    help="skip caching IMAGE .npy (arti-only / Energy-3 baseline)")
    args = ap.parse_args()
    build_arti_cache(args.mat_dir, args.out, args.manifest, args.limit,
                     args.stats_split, args.val_frac, args.test_frac, args.seed,
                     save_image=not args.no_image)


if __name__ == "__main__":
    main()
