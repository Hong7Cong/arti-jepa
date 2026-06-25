"""Build a manifest CSV for the longitudinal rtMRI corpus (Arti-JEPA Phase 0).

This is the **pre-training-only** companion to ``build_manifest.py``: the
longitudinal corpus (USC `rt_ssfp` real-time MRI, same scanner family as the
75-speaker set) is repeated sessions of disjoint speakers, with *no* dense labels
and a different on-disk layout, so it feeds T-SSL but never the phoneme eval.

Layout::

    <data-root>/ID<NN>/D<day>.<sess>/video/usc_disc_<date>_<time>_..._nframes<F>_..._tRes<MS>_...avi

Most clip filenames encode everything the dataloader needs:
``nframes<F>`` -> n_frames and ``tRes<MS>`` (ms / frame) -> fps = 1000 / MS. We
parse those by default (instant for the ~7k clips) and **decord-probe only the
handful of short-named clips that lack the tokens**, so the default run gets full
coverage without a slow whole-corpus probe. ``--probe`` forces decord on every
file (ground-truth verification); ``--no-probe-fallback`` skips even the residual
probe (pure filename parse, dropping unparseable clips). ``--workers`` sets the
parallelism of whichever probe runs.

Emits the **same column schema** as ``build_manifest.py`` + a ``split`` column so
it can be concatenated with the 75-speaker manifest by ``merge_manifests.py``:

    path, subject, stimulus_name, stimulus_group, group_idx, repeat, note,
    n_frames, fps, duration_s, split

Speakers are disjoint from the 75-set, so every row is ``split=train`` (the
held-out collapse-monitor + phoneme eval stay on the 75-speaker / usc_lss data).
Subjects are namespaced (``--subject-prefix``, default ``longi_``) so they can
never collide with the ``subNNN`` 75-speaker ids.

Usage:
    python -m artijepa.build_manifest_longitudinal \
        --data-root /project2/shrikann_35/kevinyhu/data/longitudinal \
        --out /scratch1/hongn/artijepa/manifest_longitudinal.csv
    # optional ground-truth verification of n_frames/fps (slow):
    #   ... --probe --workers 16
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

# canonical schema -- MUST match build_manifest.py (probe columns) + split, so the
# two manifests concatenate cleanly in merge_manifests.py.
FIELDS = [
    "path", "subject", "stimulus_name", "stimulus_group", "group_idx",
    "repeat", "note", "n_frames", "fps", "duration_s", "split",
]
STIMULUS_GROUP = "longitudinal"   # no stimulus taxonomy here; -1 group_idx
GROUP_IDX = -1

_NFRAMES_RE = re.compile(r"nframes(\d+)")
_TRES_RE = re.compile(r"tRes(\d+(?:\.\d+)?)")   # ms per frame -> fps = 1000 / tRes


def parse_avi_name(path: str):
    """(subject_dir, day_dir, n_frames, fps) from a longitudinal clip path.

    ``n_frames`` / ``fps`` come from the filename tokens ``nframes<F>`` /
    ``tRes<MS>``; either is ``None`` if its token is absent (then probe is needed).
    ``subject_dir`` / ``day_dir`` are the ``ID<NN>`` and ``D<day>.<sess>`` path
    components. Pure string parsing -- no file I/O, so it is unit-testable.
    """
    p = Path(path)
    parts = p.parts
    # .../ID<NN>/D<day>.<sess>/video/<name>.avi
    day_dir = parts[-3] if len(parts) >= 3 else ""
    subject_dir = parts[-4] if len(parts) >= 4 else ""
    name = p.name
    mf = _NFRAMES_RE.search(name)
    mt = _TRES_RE.search(name)
    n_frames = int(mf.group(1)) if mf else None
    tres = float(mt.group(1)) if mt else None
    fps = (1000.0 / tres) if (tres and tres > 0) else None
    return subject_dir, day_dir, n_frames, fps


def _probe_one(path: str):
    """(n_frames, fps) via decord, or (None, None) on failure."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        n = len(vr)
        fps = float(vr.get_avg_fps())
        return n, fps
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] probe failed for {os.path.basename(path)}: {e}")
        return None, None


def _probe_all(paths, workers):
    """Map ``_probe_one`` over ``paths`` (parallel if workers > 1)."""
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_probe_one, paths, chunksize=16))
    else:
        results = [_probe_one(p) for p in paths]
    return results


def build(data_root, out, subject_prefix="longi_", probe=False, workers=8,
          probe_fallback=True, split="train"):
    data_root = Path(data_root)
    videos = sorted(str(p) for p in data_root.glob("ID*/D*/video/*.avi"))
    print(f"Found {len(videos)} .avi clips under {data_root}")
    if not videos:
        raise SystemExit(f"no ID*/D*/video/*.avi under {data_root}")

    # 1) parse n_frames/fps from the filename tokens (instant)
    parsed = {vp: parse_avi_name(vp) for vp in videos}

    # 2) decide which clips need a decord probe: all of them with --probe, else
    #    only the short-named residual that filename-parsing could not resolve.
    if probe:
        to_probe = videos
    elif probe_fallback:
        to_probe = [vp for vp, (_, _, n, f) in parsed.items() if not n or not f]
    else:
        to_probe = []
    if to_probe:
        print(f"decord-probing {len(to_probe)} clip(s) "
              f"({'all (--probe)' if probe else 'filename-parse residual'}), "
              f"workers={workers}...")
    probed = dict(zip(to_probe, _probe_all(to_probe, workers))) if to_probe else {}

    rows = []
    n_missing = 0
    repeat_ctr = defaultdict(int)        # (subject, day) -> running clip index
    for vp in videos:
        subj_dir, day_dir, n_frames, fps = parsed[vp]
        if vp in probed:                 # decord ground truth fills/overrides
            pn, pf = probed[vp]
            n_frames = pn if pn is not None else n_frames
            fps = pf if pf is not None else fps
        if not n_frames or not fps:
            n_missing += 1
            print(f"  [skip] no n_frames/fps for {os.path.basename(vp)} "
                  f"(decord probe also failed)")
            continue
        subject = f"{subject_prefix}{subj_dir}"
        repeat_ctr[(subject, day_dir)] += 1
        rows.append({
            "path": vp,
            "subject": subject,
            "stimulus_name": day_dir,          # longitudinal timepoint (D<day>.<sess>)
            "stimulus_group": STIMULUS_GROUP,
            "group_idx": GROUP_IDX,
            "repeat": repeat_ctr[(subject, day_dir)],
            "note": "",
            "n_frames": n_frames,
            "fps": fps,
            "duration_s": (n_frames / fps) if fps else "",
            "split": split,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    subjects = {r["subject"] for r in rows}
    print(f"Wrote {len(rows)} rows ({len(subjects)} subjects) -> {out}")
    if n_missing:
        print(f"  [warn] dropped {n_missing} clips with unparseable n_frames/fps")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default="/project2/shrikann_35/kevinyhu/data/longitudinal")
    ap.add_argument("--out",
                    default="/scratch1/hongn/artijepa/manifest_longitudinal.csv")
    ap.add_argument("--subject-prefix", default="longi_",
                    help="namespace longitudinal subjects so they never collide "
                         "with the 75-speaker subNNN ids")
    ap.add_argument("--split", default="train",
                    help="split label for every row (disjoint speakers -> train)")
    ap.add_argument("--probe", action="store_true",
                    help="re-read n_frames/fps from EVERY file with decord instead "
                         "of trusting the filename tokens (slow, ground truth)")
    ap.add_argument("--no-probe-fallback", dest="probe_fallback",
                    action="store_false",
                    help="do NOT decord-probe the short-named residual; drop clips "
                         "whose filename lacks nframes/tRes tokens")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel decord workers for whichever probe runs")
    args = ap.parse_args()
    build(args.data_root, args.out, subject_prefix=args.subject_prefix,
          probe=args.probe, workers=args.workers,
          probe_fallback=args.probe_fallback, split=args.split)


if __name__ == "__main__":
    main()
