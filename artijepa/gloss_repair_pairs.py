"""Recover mis-paired glossectomy video<->TextGrid alignments by DURATION matching.

Some gloss sessions (notably **spk3/post**) have their TextGrid force-alignments scrambled
relative to the resampled videos: the file *names* line up (`<utt>.avi` <-> `<utt>.TextGrid`)
but the video length disagrees with the TextGrid `xmax`, so `gloss.build_manifest`'s strict
duration sanity-check drops them (spk3 post: 17 clips -> only 3 kept). The scramble is a
permutation, though — nearly every video's true alignment is *some other* TextGrid in the
same folder with a matching duration.

This tool solves the assignment: cost[i,j] = |video_dur_i - tg_xmax_j|, optimal 1:1 match via
Hungarian (scipy) or a greedy fallback. Confidence per recovered pair:
  * high   -- residual < 0.30 s AND it is the ONLY TextGrid within 0.30 s of that video
              (unique tight match — the duration coincidence is not shared).
  * medium -- residual < tol=max(1.0, 0.1*xmax) but not uniquely tight (a second TextGrid is
              also within tolerance; the global permutation still prefers this one).
  * drop   -- residual > tol (no credible alignment; e.g. a video whose TextGrid is missing).

Outputs (NON-destructive — never touches the canonical manifest / phonemes_json):
  <root>/phonemes_json_repaired/<video_utt>.json         corrected per-video alignment
  <root>/gloss_<cond>_<speaker>_repaired_manifest.csv    usc_lss cols + tg_source/residual/confidence
  eval/gloss/gloss_repair_<cond>_<speaker>.json          full assignment report

Usage:
    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.gloss_repair_pairs --speaker spk3 --cond post
    #   --min-confidence high|medium   (which pairs to write into the manifest; default medium)
    #   --write                        actually write JSONs+manifest (default: dry-run report only)
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from decord import VideoReader, cpu

from artijepa.gloss import (parse_phones_tier, _resampled_video, _source_video,
                            native_fps, GLOSS_ROOT, COND_DIRS, JSON_DIR, MIN_FPS)

TIGHT = 0.30            # s: "uniquely tight" threshold for high confidence


def _tol(xmax):
    return max(1.0, 0.1 * xmax)


def _collect(cond_dir, min_fps=MIN_FPS):
    """-> [(utt, video_dur_s, xmax, segs)] for every TextGrid whose video is readable
    and whose SOURCE acquisition is at least `min_fps` (see artijepa.gloss)."""
    tgdir = os.path.join(GLOSS_ROOT, cond_dir, "textgrids")
    items = []
    for tg in sorted(glob.glob(os.path.join(tgdir, "*.TextGrid"))):
        utt = os.path.basename(tg)[:-len(".TextGrid")]
        vid = _resampled_video(cond_dir, utt)
        if not os.path.exists(vid):
            continue
        src = _source_video(cond_dir, utt)
        sfps = native_fps(src) if src else None
        if min_fps and not (sfps and sfps >= min_fps):
            continue
        segs, xmax = parse_phones_tier(tg)
        try:
            vr = VideoReader(vid, num_threads=1, ctx=cpu(0))
            dur = len(vr) / float(vr.get_avg_fps())
            n, fps = len(vr), float(vr.get_avg_fps())
        except Exception:
            continue
        items.append({"utt": utt, "vid": vid, "dur": dur, "n_frames": n, "fps": fps,
                      "src_fps": sfps, "xmax": xmax or 0.0, "segs": segs})
    return items


def _assign(dur, xmax):
    """Optimal 1:1 video->textgrid assignment minimising sum |dur-xmax|."""
    C = np.abs(np.asarray(dur)[:, None] - np.asarray(xmax)[None, :])
    try:
        from scipy.optimize import linear_sum_assignment
        ri, cj = linear_sum_assignment(C)
        return list(zip(ri.tolist(), cj.tolist())), C, "hungarian"
    except Exception:
        pairs, usedj = [], set()
        for i, j in np.dstack(np.unravel_index(np.argsort(C, axis=None), C.shape))[0]:
            i, j = int(i), int(j)
            if any(p[0] == i for p in pairs) or j in usedj:
                continue
            pairs.append((i, j)); usedj.add(j)
            if len(pairs) == min(C.shape):
                break
        return pairs, C, "greedy"


def repair(cond_dir, speaker):
    items = _collect(cond_dir)
    if not items:
        return None
    dur = [it["dur"] for it in items]
    xmax = [it["xmax"] for it in items]
    pairs, C, method = _assign(dur, xmax)
    out = []
    for i, j in pairs:
        resid = float(C[i, j])
        # uniqueness: how many TextGrids sit within TIGHT of this video?
        n_tight = int((np.abs(np.asarray(xmax) - dur[i]) < TIGHT).sum())
        if resid > _tol(xmax[j]):
            conf = "drop"
        elif resid < TIGHT and n_tight == 1:
            conf = "high"
        else:
            conf = "medium"
        out.append({
            "video_utt": items[i]["utt"], "video": items[i]["vid"],
            "n_frames": items[i]["n_frames"], "fps": round(items[i]["fps"], 4),
            "src_fps": (round(items[i]["src_fps"], 4) if items[i]["src_fps"] else ""),
            "video_dur": round(dur[i], 3),
            "tg_source": items[j]["utt"], "tg_xmax": round(xmax[j], 3),
            "residual": round(resid, 3), "name_match": items[i]["utt"] == items[j]["utt"],
            "confidence": conf, "segs": items[j]["segs"],
        })
    out.sort(key=lambda r: r["residual"])
    return {"cond_dir": cond_dir, "speaker": speaker, "method": method, "pairs": out}


CONF_RANK = {"high": 2, "medium": 1, "drop": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default="spk3")
    ap.add_argument("--cond", default="post", choices=["pre", "post"])
    ap.add_argument("--min-confidence", default="medium", choices=["high", "medium"],
                    help="minimum confidence to write into the repaired manifest")
    ap.add_argument("--write", action="store_true",
                    help="write corrected JSONs + repaired manifest (default: dry-run)")
    ap.add_argument("--arti-out", default="/scratch1/hongn/artijepa")
    args = ap.parse_args()

    cond_dirs = [d for d in COND_DIRS[args.cond] if d.split("/")[0] == args.speaker]
    reports = [repair(cd, args.speaker) for cd in cond_dirs]
    reports = [r for r in reports if r]

    thr = CONF_RANK[args.min_confidence]
    print(f"\n=== duration re-pairing: {args.speaker} {args.cond} ===")
    hdr = f"{'video(utt)':<24}{'dur':>8}{'-> tg(source)':>26}{'xmax':>8}{'|Δ|':>7}  conf   name?"
    all_pairs = []
    for r in reports:
        print(f"\n[{r['cond_dir']}]  method={r['method']}")
        print(hdr); print("-" * len(hdr))
        for p in r["pairs"]:
            print(f"{p['video_utt']:<24}{p['video_dur']:>8.2f}{'  '+p['tg_source']:>26}"
                  f"{p['tg_xmax']:>8.2f}{p['residual']:>7.2f}  {p['confidence']:<6} "
                  f"{'name-ok' if p['name_match'] else 'REPAIR'}")
            all_pairs.append((r["cond_dir"], p))
    n_high = sum(1 for _, p in all_pairs if p["confidence"] == "high")
    n_med = sum(1 for _, p in all_pairs if p["confidence"] == "medium")
    n_drop = sum(1 for _, p in all_pairs if p["confidence"] == "drop")
    n_write = sum(1 for _, p in all_pairs if CONF_RANK[p["confidence"]] >= thr)
    print(f"\nsummary: {len(all_pairs)} clips -> high={n_high} medium={n_med} drop={n_drop}"
          f"  | writable(>= {args.min_confidence})={n_write}  (name-based baseline kept 3)")

    # always save the full report
    rep_dir = os.path.join(args.arti_out, "eval", "gloss")
    os.makedirs(rep_dir, exist_ok=True)
    rpath = os.path.join(rep_dir, f"gloss_repair_{args.cond}_{args.speaker}.json")
    json.dump({"cond": args.cond, "speaker": args.speaker,
               "pairs": [{k: v for k, v in p.items() if k != "segs"}
                         for _, p in all_pairs]}, open(rpath, "w"), indent=2)
    print(f"[repair] wrote report {rpath}")

    if not args.write:
        print("[repair] dry-run (pass --write to emit corrected JSONs + repaired manifest)")
        return

    jdir = os.path.join(GLOSS_ROOT, "phonemes_json_repaired"); os.makedirs(jdir, exist_ok=True)
    cols = ["utt_id", "path", "phoneme_json", "audio", "n_frames", "fps", "duration_s",
            "split", "speaker", "condition", "src_fps", "tg_source", "residual_s",
            "confidence"]
    rows = []
    for cond_dir, p in all_pairs:
        if CONF_RANK[p["confidence"]] < thr:
            continue
        pj = os.path.join(jdir, f"{p['video_utt']}.json")
        json.dump([{"phoneme": ph, "start": s0, "end": s1} for ph, s0, s1 in p["segs"]],
                  open(pj, "w"))
        rows.append({"utt_id": p["video_utt"], "path": p["video"], "phoneme_json": pj,
                     "audio": "", "n_frames": p["n_frames"], "fps": p["fps"],
                     "duration_s": p["video_dur"], "split": "test", "speaker": args.speaker,
                     "condition": args.cond, "src_fps": p["src_fps"],
                     "tg_source": p["tg_source"],
                     "residual_s": p["residual"], "confidence": p["confidence"]})
    rows.sort(key=lambda r: r["utt_id"])
    mpath = os.path.join(GLOSS_ROOT,
                         f"gloss_{args.cond}_{args.speaker}_repaired_manifest.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[repair] wrote {len(rows)} corrected JSONs -> {jdir}")
    print(f"[repair] wrote repaired manifest ({len(rows)} utts) -> {mpath}")


if __name__ == "__main__":
    main()
