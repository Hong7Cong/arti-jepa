"""Scan the rtMRI corpus and build a manifest CSV (Arti-JEPA Phase 0 / A.8).

Output columns:
    path, subject, stimulus_name, stimulus_group, group_idx,
    repeat, note, [n_frames, fps, duration_s]   (last three only with --probe)

Usage:
    python -m artijepa.build_manifest \
        --data-root /scratch1/hongn/speaker75 \
        --out /scratch1/hongn/artijepa/manifest.csv --probe
"""

import argparse
import csv
import json
import os
from pathlib import Path

from artijepa.labels import (
    GROUP_TO_IDX,
    filename_to_prefix,
    parse_prefix,
    stimulus_to_group,
)


def load_notes(metafile: str):
    """Build {prefix: note} from metafile_public_*.json (notes flag artifacts)."""
    notes = {}
    if not metafile or not os.path.exists(metafile):
        return notes
    with open(metafile) as f:
        meta = json.load(f)
    for subj, rec in meta.items():
        for task in rec.get("task", []):
            pref = task.get("prefix")
            if pref:
                notes[pref] = task.get("note", "")
    return notes


def probe_video(path: str):
    """Return (n_frames, fps, duration_s) via decord, or (None, None, None)."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        n = len(vr)
        fps = float(vr.get_avg_fps())
        return n, fps, (n / fps if fps else None)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] probe failed for {os.path.basename(path)}: {e}")
        return None, None, None


def build(data_root: str, out: str, metafile: str = None, probe: bool = False):
    data_root = Path(data_root)
    if metafile is None:
        cand = sorted(data_root.glob("metafile_public_*.json"))
        metafile = str(cand[0]) if cand else None
    notes = load_notes(metafile)

    videos = sorted(data_root.glob("sub*/2drt/video/*_video.mp4"))
    print(f"Found {len(videos)} videos under {data_root}")

    rows = []
    for i, vp in enumerate(videos):
        prefix = filename_to_prefix(vp.name)
        parsed = parse_prefix(prefix)
        if parsed is None:
            print(f"  [skip] cannot parse {vp.name}")
            continue
        group = stimulus_to_group(parsed["stimulus_name"])
        row = {
            "path": str(vp),
            "subject": parsed["subject"],
            "stimulus_name": parsed["stimulus_name"],
            "stimulus_group": group,
            "group_idx": GROUP_TO_IDX.get(group, -1),
            "repeat": parsed["repeat"],
            "note": notes.get(prefix, ""),
        }
        if probe:
            n, fps, dur = probe_video(str(vp))
            row.update(n_frames=n, fps=fps, duration_s=dur)
            if (i + 1) % 200 == 0:
                print(f"  probed {i + 1}/{len(videos)}")
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fields = [
        "path", "subject", "stimulus_name", "stimulus_group", "group_idx",
        "repeat", "note",
    ]
    if probe:
        fields += ["n_frames", "fps", "duration_s"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_subj = len({r["subject"] for r in rows})
    print(f"Wrote {len(rows)} rows ({n_subj} subjects) -> {out}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/scratch1/hongn/speaker75")
    ap.add_argument("--out", default="/scratch1/hongn/artijepa/manifest.csv")
    ap.add_argument("--metafile", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="open each video with decord to record fps/n_frames")
    args = ap.parse_args()
    build(args.data_root, args.out, args.metafile, args.probe)


if __name__ == "__main__":
    main()
