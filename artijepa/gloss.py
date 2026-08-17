"""Pre/post-glossectomy dataset: TextGrid phones -> usc_lss-format manifest+JSON.

The gloss corpus (`/scratch1/hongn/gloss`) has 3 speakers each recorded **pre** and
**post** glossectomy, with MFA forced-aligned `phones` tiers (uppercase ARPABET +
stress). We decode phonemes on pre vs post to quantify articulatory change under
altered anatomy, reusing the *entire* usc_lss Task-2 path: this module converts each
TextGrid `phones` tier to a usc_lss-style phoneme JSON (`[{phoneme,start,end}]`,
lowercase, stress-stripped -> the 41-symbol inventory in `phonemes.py`) and writes a
manifest CSV with the usc_lss columns, so `USCLSSPhonemeDataset` / `eval_phoneme.py`
consume it verbatim (`data.kind: usc_lss`).

**Paired video = the NATIVE source `spk*/<cond>/{video,avi}/<utt>.avi`** (~99.4 fps,
104x104, 1:1 basename with the TextGrid). Pairing is **strict**: an utterance is kept
only if its video exists AND `|video_duration - TextGrid.xmax| <= tol`.

*Why not `resampled_video/`?* That tree (a uniform-100-fps re-encode of the sources) was
what earlier manifests pointed at; it is **not needed and slightly lossy**. The dataloader
is frame-rate agnostic: `rtmri_dataset` resamples native -> `target_fps` (50.0 for the
gloss/annot16 configs) by linear interpolation on a time grid, "never by re-encoding",
using the manifest's declared `fps`; alignment is in seconds. Training data is itself
heterogeneous (annot16 83.28 fps, usc_lss 99.01 fps). Going through a 100 fps intermediate
therefore just inserted an extra lossy step (99.4 -> 100 by frame duplication -> 50) where
one interpolation (99.4 -> 50) does the job, exactly as usc_lss is handled. Numbers
computed off these manifests differ slightly from ones published before 2026-08-06 for
this reason.

**Native-fps gate (`--min-fps`, default 20).** The corpus is *not* acquired at one
frame rate: sources split cleanly into **99.4 fps** and **15.29 fps**, and the split cuts
*across* speakers and sessions (e.g. spk1/post is 9x 15.29 + 4x 99.4). A 15.29 fps clip
resampled up to the 50 fps grid is mostly duplicated frames — temporally fake to the
encoder — so we drop anything below `min_fps`. The 15.29 fps material is also exactly the
word-list/VCV subset in spk1/post, spk2/pre and spk3/pre, so the gate changes stimulus
composition in those cells; see RESULTS_gloss.md. Pass `--min-fps 0` to disable it.

!! CACHE INVALIDATION !! `eval_phoneme._tag` / `eval_phoneme_groups._tag` hash the manifest
**path**, not its contents, so rebuilding these manifests IN PLACE does *not* change the
feature-cache name: a later run will happily cache-hit features extracted from the previous
corpus and report silently wrong numbers. Whenever a rebuild changes which utterances are
included (an `--min-fps` change, a filename correction, repointing at a different video
tree), move the affected `feat_cache/{phoneme,glossphg}/gloss*` dirs aside first. The
2026-08-06 rebuild's stale caches are parked in `feat_cache/_stale_pre20260806/`.

CLI:
    source dev_artiJEPA/scripts/_env.sh
    python -m artijepa.gloss                 # build gloss_{pre,post}_manifest.csv
    python -m artijepa.gloss --speaker spk1  # restrict to one speaker
    python -m artijepa.gloss --min-fps 0     # keep the 15.29 fps acquisitions too
"""

import argparse
import csv
import json
import os
import re
import glob

from decord import VideoReader, cpu

from artijepa import phonemes as P

GLOSS_ROOT = "/scratch1/hongn/gloss"
JSON_DIR = os.path.join(GLOSS_ROOT, "phonemes_json")

# native acquisition rate below which a clip is rejected. The corpus has exactly two
# rates (99.4 and 15.29 fps), so anything in 20..99 separates them identically.
MIN_FPS = 20.0

# where the *source* (pre-resample) videos live under a cond dir; spk2/pre & spk3/pre
# use `avi/`, everyone else `video/`.
SRC_SUBDIRS = ("video", "avi")
SRC_EXTS = (".avi", ".mp4")

# speaker/condition dirs holding the co-located `textgrids/` (source of alignment),
# grouped into the two conditions we compare. spk2 post1/post2 both -> "post".
COND_DIRS = {
    "pre": ["spk1/pre", "spk2/pre", "spk3/pre"],
    "post": ["spk1/post", "spk2/post1", "spk2/post2", "spk3/post"],
}


# --------------------------------------------------------------------------- #
# TextGrid phones-tier parser
# --------------------------------------------------------------------------- #
_STRESS = re.compile(r"[0-2]$")


def _map_phone(text):
    """MFA phones symbol -> usc_lss 41-inventory symbol (or None to skip).

    '' / sil / sp / spn (silence & non-speech noise) -> 'sil'; else lowercase +
    strip the trailing stress digit (AH0->ah, AY1->ay). Symbols outside the
    inventory fall through to PHON2IDX.get(...) = IGNORE downstream."""
    t = text.strip().lower()
    if t in ("", "sil", "sp", "spn", "sils"):
        return "sil"
    return _STRESS.sub("", t)


def parse_phones_tier(tg_path):
    """-> ([(phoneme, start_s, end_s), ...] sorted, xmax_float). Reads the interval
    tier named 'phones' from a Praat ooTextFile TextGrid."""
    txt = open(tg_path, encoding="utf-8", errors="replace").read()
    xmax = None
    m = re.search(r"xmax = ([\d.]+)", txt)
    if m:
        xmax = float(m.group(1))
    # slice out the 'phones' item block (to the next item[...] or EOF)
    mi = re.search(r'name = "phones"', txt)
    if not mi:
        return [], xmax
    tail = txt[mi.end():]
    nxt = re.search(r'\n\s*item \[', tail)
    block = tail[: nxt.start()] if nxt else tail
    segs = []
    for im in re.finditer(
            r"xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = \"([^\"]*)\"",
            block):
        s0, s1, text = float(im.group(1)), float(im.group(2)), im.group(3)
        ph = _map_phone(text)
        if ph is not None:
            segs.append((ph, s0, s1))
    segs.sort(key=lambda x: x[1])
    return segs, xmax


# --------------------------------------------------------------------------- #
# manifest builder (strict video<->textgrid pairing)
# --------------------------------------------------------------------------- #
def _resampled_video(cond_dir, utt_id):
    """resampled_video/<spk>/<cond>/<utt>.avi for a co-located textgrid dir."""
    return os.path.join(GLOSS_ROOT, "resampled_video", cond_dir, f"{utt_id}.avi")


def _source_video(cond_dir, utt_id, root=GLOSS_ROOT):
    """<cond_dir>/{video,avi}/<utt>.{avi,mp4} -- the pre-resample acquisition, whose
    frame rate is the *real* one. Returns None if no source file is found."""
    for sub in SRC_SUBDIRS:
        for ext in SRC_EXTS:
            p = os.path.join(root, cond_dir, sub, f"{utt_id}{ext}")
            if os.path.exists(p):
                return p
    return None


_FPS_CACHE = {}


def native_fps(path):
    """Native frame rate of a source video (decord avg fps), or None if unreadable."""
    if path not in _FPS_CACHE:
        try:
            vr = VideoReader(path, num_threads=1, ctx=cpu(0))
            _FPS_CACHE[path] = float(vr.get_avg_fps())
        except Exception:
            _FPS_CACHE[path] = None
    return _FPS_CACHE[path]


def build_manifest(root=GLOSS_ROOT, speaker=None, tol_frac=0.1, tol_min=1.0,
                   min_fps=MIN_FPS, verbose=True):
    """Write gloss_{pre,post}_manifest.csv (usc_lss columns + speaker/condition/src_fps).

    Returns {cond: csv_path}. Drops utts with no matching resampled video, a
    video/TextGrid duration disagreement > max(tol_min, tol_frac*xmax), or a *native*
    (pre-resample) frame rate below `min_fps` -- see the module docstring on why the
    uniform 100 fps of `resampled_video/` is not the real rate. `min_fps=0` disables
    the rate gate."""
    os.makedirs(JSON_DIR, exist_ok=True)
    # `session` is explicit because it can NOT be recovered from `path`: the source tree
    # is <spk>/<session>/{video,avi}/<utt>.avi, so the video's parent dir is the media
    # subdir, not the session. Consumers must read this column (see _speaker_groups).
    cols = ["utt_id", "path", "phoneme_json", "audio", "n_frames", "fps",
            "duration_s", "split", "speaker", "session", "condition", "src_fps"]
    out = {}
    for cond, dirs in COND_DIRS.items():
        rows, dropped = [], []
        for cdir in dirs:
            spk = cdir.split("/")[0]
            if speaker and spk != speaker:
                continue
            tgdir = os.path.join(root, cdir, "textgrids")
            for tg in sorted(glob.glob(os.path.join(tgdir, "*.TextGrid"))):
                utt = os.path.basename(tg)[:-len(".TextGrid")]
                vid = _source_video(cdir, utt, root)
                if vid is None:
                    dropped.append((utt, "no_video")); continue
                segs, xmax = parse_phones_tier(tg)
                if not segs:
                    dropped.append((utt, "no_phones")); continue
                try:
                    vr = VideoReader(vid, num_threads=1, ctx=cpu(0))
                    n = len(vr); fps = float(vr.get_avg_fps())
                except Exception as e:
                    dropped.append((utt, f"unreadable:{type(e).__name__}")); continue
                # native-rate gate -- `fps` here IS the acquisition rate (no re-encode)
                if min_fps and fps < min_fps:
                    dropped.append((utt, f"low_fps {fps:.2f}<{min_fps:g}")); continue
                sfps = fps
                dur = n / fps
                if xmax and abs(dur - xmax) > max(tol_min, tol_frac * xmax):
                    dropped.append((utt, f"dur {dur:.1f}!=xmax {xmax:.1f}")); continue
                pj = os.path.join(JSON_DIR, f"{utt}.json")
                json.dump([{"phoneme": p, "start": s0, "end": s1}
                           for p, s0, s1 in segs], open(pj, "w"))
                rows.append({
                    "utt_id": utt, "path": vid, "phoneme_json": pj, "audio": "",
                    "n_frames": n, "fps": round(fps, 4), "duration_s": round(dur, 3),
                    "split": "test", "speaker": spk,
                    "session": cdir.split("/")[1], "condition": cond,
                    "src_fps": round(sfps, 4) if sfps else ""})
        rows.sort(key=lambda r: r["utt_id"])
        csv_path = os.path.join(root, f"gloss_{cond}_manifest.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        out[cond] = csv_path
        if verbose:
            from collections import Counter
            bs = Counter(r["speaker"] for r in rows)
            n_low = sum(1 for _, why in dropped if why.startswith("low_fps"))
            print(f"[gloss] {cond}: {len(rows)} utts kept {dict(bs)}  "
                  f"({len(dropped)} dropped, {n_low} for native fps < {min_fps:g}) "
                  f"-> {csv_path}")
            print(f"[gloss]   kept src_fps: "
                  f"{dict(Counter(r['src_fps'] for r in rows))}")
            for utt, why in dropped[:12]:
                print(f"[gloss]   drop {utt}: {why}")
            if len(dropped) > 12:
                print(f"[gloss]   ... +{len(dropped)-12} more dropped")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=GLOSS_ROOT)
    ap.add_argument("--speaker", default=None, help="restrict to spk1|spk2|spk3")
    ap.add_argument("--tol-min", type=float, default=1.0,
                    help="min |vid_dur - tg_xmax| tolerance in seconds")
    ap.add_argument("--min-fps", type=float, default=MIN_FPS,
                    help="drop utts whose SOURCE video's native fps is below this "
                         "(corpus is 99.4 or 15.29 fps); 0 disables the gate")
    args = ap.parse_args()
    build_manifest(args.root, speaker=args.speaker, tol_min=args.tol_min,
                   min_fps=args.min_fps)


if __name__ == "__main__":
    main()
