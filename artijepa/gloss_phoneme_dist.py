"""Phoneme / vowel distribution of the glossectomy corpus, per speaker x condition.

Reads the gold ARPABET segment annotations straight from the gloss manifests
(`gloss_{pre,post}_manifest.csv` -> per-utt phoneme JSON) -- NO encoder / GPU. Each
gold segment counts as one phoneme *instance*; we also sum segment durations (s) so
you can read either occurrence- or time-weighted composition. Groups by manner via
`tsne_phonemes.PHON_CLASS` and flags vowels via `VOWEL_PHON`.

**Native-fps gate (`--min-fps`).** The corpus mixes two acquisition rates (99.4 and 15.29
fps) and the manifest `fps` column is a uniform 100.0 for every row, so it cannot be used
for this -- we probe each utterance's *source* video exactly like `gloss.build_manifest`
(see that module's docstring). A row is excluded when its source is missing (e.g. moved to
`_lowfps_quarantine/`) or its native rate is below `min_fps`. Pass `--min-fps 0` for the
unfiltered whole-corpus distribution. Because the 15.29 fps material is the word-list/VCV
subset in spk1/post, spk2/pre and spk3/pre, filtering changes the *composition*, not just
the totals -- compare the two runs rather than reading the filtered one alone.

Outputs (to eval/gloss/, with `--suffix` appended before the extension):
  - gloss_phoneme_dist.csv    (long form: condition,speaker,phoneme,class,is_vowel,count,dur_s)
  - gloss_phoneme_dist.json   (nested summary incl. per-group + vowel-only tables)
  - gloss_vowel_dist.png      (per-speaker x condition vowel-count bars)

Usage:
    source dev_artiJEPA/scripts/_env.sh
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.gloss_phoneme_dist              # >=20 fps
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.gloss_phoneme_dist --min-fps 0  # everything
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decord import VideoReader, cpu

import artijepa.phonemes as P
from artijepa.gloss import (MIN_FPS, COND_DIRS, _source_video, native_fps,
                            parse_phones_tier)
from artijepa.tsne_phonemes import PHON_CLASS, VOWEL_PHON, CLASS_ORDER

GLOSS_ROOT = "/scratch1/hongn/gloss"
OUT_DIR = "/scratch1/hongn/artijepa/eval/gloss"
CONDS = ["pre", "post"]
VOWSET = set(VOWEL_PHON)


def _rows(cond):
    with open(os.path.join(GLOSS_ROOT, f"gloss_{cond}_manifest.csv")) as f:
        return list(csv.DictReader(f))


def _group(r, cond):
    """speaker, OR speaker/<session> when a speaker has >1 acquisition session within a
    condition (spk2 post -> spk2/post1, spk2/post2). Matches eval_gloss._speaker_groups
    so the distribution keys line up with the transfer per-speaker rows."""
    # prefer the explicit column; source-tree paths end in the media subdir, not the session
    sess = r.get("session") or os.path.basename(os.path.dirname(r.get("path", "")))
    spk = r["speaker"]
    return f"{spk}/{sess}" if sess and sess != cond else spk


def _row_src_fps(r):
    """Native fps of a manifest row's SOURCE video (None if the source is gone).

    The row's `path` is the resampled clip `.../resampled_video/<spk>/<sess>/<utt>.avi`,
    whose own fps is a uniform 100.0 and therefore uninformative; `<spk>/<sess>` is the
    cond dir under which the real acquisition lives. Prefers an `src_fps` column when the
    manifest already carries one (written by newer `gloss.build_manifest` runs)."""
    if r.get("src_fps"):
        return float(r["src_fps"])
    parts = r["path"].split("/")
    cond_dir = "/".join(parts[-3:-1])
    utt = os.path.splitext(parts[-1])[0]
    src = _source_video(cond_dir, utt)
    return native_fps(src) if src else None


def collect_from_textgrids(min_fps=MIN_FPS, tol_frac=0.1, tol_min=1.0):
    """Same return shape as `collect()`, but enumerated from the TextGrids + **source**
    videos instead of the manifests.

    Use this when the manifests are unavailable or stale -- notably (a) `resampled_video/`
    is missing, so `build_manifest` cannot run at all, and (b) the canonical manifests were
    built before the spk3/post filename permutation was corrected, so their spk3/post rows
    carry the wrong video<->annotation pairing. Applies exactly the gates `build_manifest`
    applies: source video present, native fps >= `min_fps`, and
    |video_dur - TextGrid.xmax| <= max(tol_min, tol_frac*xmax)."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    n_utt = defaultdict(int)
    dropped = defaultdict(list)
    for cond, cdirs in COND_DIRS.items():
        for cdir in cdirs:
            spk, sess = cdir.split("/")
            g = spk if sess == cond else f"{spk}/{sess}"
            k = (cond, g)
            n_utt.setdefault(k, 0)
            for tg in sorted(glob.glob(
                    os.path.join(GLOSS_ROOT, cdir, "textgrids", "*.TextGrid"))):
                utt = os.path.basename(tg)[:-len(".TextGrid")]
                src = _source_video(cdir, utt)
                if src is None:
                    dropped[k].append((utt, "no_source")); continue
                f = native_fps(src)
                if min_fps and (f is None or f < min_fps):
                    dropped[k].append(
                        (utt, "unreadable" if f is None else f"{f:.2f}fps")); continue
                segs, xmax = parse_phones_tier(tg)
                if not segs:
                    dropped[k].append((utt, "no_phones")); continue
                vr = VideoReader(src, num_threads=1, ctx=cpu(0))
                dur = len(vr) / float(vr.get_avg_fps())
                if xmax and abs(dur - xmax) > max(tol_min, tol_frac * xmax):
                    dropped[k].append((utt, f"dur {dur:.2f}!={xmax:.2f}")); continue
                n_utt[k] += 1
                for ph, s0, s1 in segs:
                    c = counts[k][ph]
                    c[0] += 1
                    c[1] += max(0.0, s1 - s0)
    return counts, n_utt, dropped


def collect(min_fps=MIN_FPS):
    """-> counts[(cond,group)][phoneme] = [n_instances, total_dur_s], n_utt, dropped."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    n_utt = defaultdict(int)
    dropped = defaultdict(list)
    for cond in CONDS:
        for r in _rows(cond):
            g = _group(r, cond)
            if min_fps:
                f = _row_src_fps(r)
                if f is None or f < min_fps:
                    dropped[(cond, g)].append(
                        (r["utt_id"], "no_source" if f is None else f"{f:.2f}fps"))
                    continue
            n_utt[(cond, g)] += 1
            for ph, s0, s1 in P.load_gold_segments(r["phoneme_json"]):
                c = counts[(cond, g)][ph]
                c[0] += 1
                c[1] += max(0.0, s1 - s0)
    return counts, n_utt, dropped


def _keys(counts):
    """Ordered (cond, group) keys: condition order, then speaker, then session."""
    ks = [k for k in counts]
    return sorted(ks, key=lambda k: (CONDS.index(k[0]), k[1]))


def _col(k):
    """Compact column label for a (cond, group) key, e.g. 'pre:spk1', 'post:spk2/post1'."""
    return f"{k[0]}:{k[1]}"


W = 15   # column width (fits the longest label, 'post:spk2/post1')

SPEECH = [p for p in P.ARPABET if p != "sil"]


def _totals(counts, keys):
    """-> tot[k] = {n_speech, dur_speech, n_all, dur_all, n_vowel} per (cond,group)."""
    tot = {}
    for k in keys:
        ns = sum(counts[k][p][0] for p in SPEECH)
        ds = sum(counts[k][p][1] for p in SPEECH)
        tot[k] = {"n_speech": ns, "dur_speech": round(ds, 3),
                  "n_all": ns + counts[k]["sil"][0],
                  "dur_all": round(ds + counts[k]["sil"][1], 3),
                  "n_vowel": sum(counts[k][p][0] for p in VOWEL_PHON)}
    return tot


def _cell(n, denom):
    """'<count> (<pct>%)' -- pct of `denom`, blank-safe when denom is 0."""
    return f"{n} ({100.0 * n / denom:.1f}%)" if denom else f"{n} (--)"


def _phoneme_table(counts, n_utt, tot):
    """Pretty per-condition x group phoneme table, each cell `n (% of speech segments)`.
    `sil` is excluded from the speech denominator and reported separately against the
    all-segment total."""
    keys = _keys(counts)
    hdr = f"{'phoneme':<8}{'class':<12}" + "".join(_col(k).rjust(W) for k in keys)
    lines = [hdr, "-" * len(hdr)]
    for ph in SPEECH:
        cls = PHON_CLASS.get(ph, "?")
        vtag = "*" if ph in VOWSET else " "
        cells = "".join(_cell(counts[k][ph][0], tot[k]["n_speech"]).rjust(W) for k in keys)
        lines.append(f"{ph+vtag:<8}{cls:<12}{cells}")
    lines.append("-" * len(hdr))
    lines.append(f"{'sil':<8}{'Silence':<12}" + "".join(
        _cell(counts[k]["sil"][0], tot[k]["n_all"]).rjust(W) for k in keys) + "   [% of ALL]")
    lines.append(f"{'TOTAL(sp)':<8}{'':<12}" + "".join(
        f"{tot[k]['n_speech']:>{W}}" for k in keys))
    lines.append(f"{'TOTAL(all)':<8}{'':<12}" + "".join(
        f"{tot[k]['n_all']:>{W}}" for k in keys))
    lines.append(f"{'dur(sp)s':<8}{'':<12}" + "".join(
        f"{tot[k]['dur_speech']:>{W}.1f}" for k in keys))
    lines.append(f"{'n_utt':<8}{'':<12}" + "".join(f"{n_utt[k]:>{W}}" for k in keys))
    return "\n".join(lines), keys


def _group_summary(counts, keys):
    """Per manner-group instance counts + shares (sil-excluded)."""
    out = {}
    for k in keys:
        g = Counter()
        for ph, (n, _d) in counts[k].items():
            if ph == "sil":
                continue
            g[PHON_CLASS.get(ph, "?")] += n
        tot = sum(g.values()) or 1
        out[_col(k)] = {cls: {"n": g.get(cls, 0), "pct": round(100 * g.get(cls, 0) / tot, 1)}
                        for cls in CLASS_ORDER}
    return out


def _vowel_table(counts, keys, tot):
    """Each cell `n (% of that column's VOWEL total)`; the final row gives the vowel
    total and its share of speech segments."""
    lines = [f"{'vowel':<7}" + "".join(_col(k).rjust(W) for k in keys)]
    lines.append("-" * len(lines[0]))
    for ph in VOWEL_PHON:
        cells = "".join(_cell(counts[k][ph][0], tot[k]["n_vowel"]).rjust(W) for k in keys)
        lines.append(f"{ph:<7}{cells}")
    lines.append("-" * len(lines[0]))
    lines.append(f"{'VOW tot':<7}" + "".join(
        _cell(tot[k]["n_vowel"], tot[k]["n_speech"]).rjust(W) for k in keys)
        + "   [% of speech]")
    return "\n".join(lines)


def plot_vowels(counts, keys, path):
    x = np.arange(len(VOWEL_PHON))
    ncol = len(keys)
    w = 0.8 / ncol
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, k in enumerate(keys):
        vals = [counts[k][p][0] for p in VOWEL_PHON]
        ax.bar(x + (j - (ncol - 1) / 2) * w, vals, w, label=f"{k[0]}/{k[1]}")
    ax.set_xticks(x); ax.set_xticklabels(VOWEL_PHON)
    ax.set_ylabel("gold segment count"); ax.set_xlabel("vowel (ARPABET)")
    ax.set_title("Glossectomy vowel distribution — per speaker x condition (gold segments)")
    ax.legend(ncol=ncol, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[dist] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-fps", type=float, default=MIN_FPS,
                    help="exclude utts whose SOURCE video is below this native fps "
                         "(or is missing); 0 = whole corpus, no gate")
    ap.add_argument("--suffix", default=None,
                    help="appended to output filenames; defaults to '_minfps<N>' when "
                         "the gate is on, '' otherwise")
    ap.add_argument("--from-textgrids", action="store_true",
                    help="enumerate utterances from the TextGrids + source videos rather "
                         "than the manifests -- required while resampled_video/ is missing, "
                         "and the only correct source for spk3/post after the filename fix")
    args = ap.parse_args()
    sfx = args.suffix if args.suffix is not None else (
        f"_minfps{args.min_fps:g}" if args.min_fps else "")

    os.makedirs(OUT_DIR, exist_ok=True)
    src = "textgrids+source videos" if args.from_textgrids else "manifests"
    print(f"[dist] utterance source: {src}")
    counts, n_utt, dropped = (collect_from_textgrids(args.min_fps) if args.from_textgrids
                              else collect(args.min_fps))
    if args.min_fps:
        n_drop = sum(len(v) for v in dropped.values())
        print(f"\n=== native-fps gate: min_fps={args.min_fps:g} -> "
              f"{sum(n_utt.values())} utts kept, {n_drop} excluded ===")
        for k in sorted(dropped, key=lambda k: (CONDS.index(k[0]), k[1])):
            why = Counter(w for _, w in dropped[k])
            print(f"  {_col(k):<16} -{len(dropped[k]):>3} utts  "
                  f"({', '.join(f'{n}x {w}' for w, n in why.most_common())})  "
                  f"-> {n_utt[k]} remain")
        for k in sorted(n_utt, key=lambda k: (CONDS.index(k[0]), k[1])):
            if k not in dropped:
                print(f"  {_col(k):<16} unchanged   -> {n_utt[k]} remain")
    keys = _keys(counts)
    tot = _totals(counts, keys)
    tbl, keys = _phoneme_table(counts, n_utt, tot)
    print("\n=== Phoneme-instance distribution (gold segments; * = vowel) ===")
    print("    cell = count (% of that column's speech segments, sil excluded)")
    print(tbl)
    print("\n=== Vowel-only distribution ===")
    print("    cell = count (% of that column's vowel segments)")
    print(_vowel_table(counts, keys, tot))
    groups = _group_summary(counts, keys)
    print("\n=== Manner-group counts and shares (sil-excluded) ===")
    for k in keys:
        gd = groups[_col(k)]
        print(f"{_col(k):<16} " + "  ".join(
            f"{c[:4]}={gd[c]['n']:>5} ({gd[c]['pct']:>4.1f}%)" for c in CLASS_ORDER)
            + f"   speech={tot[k]['n_speech']}")

    # long-form CSV (group split into speaker + session so spk2/post1|post2 are explicit)
    def _split(cond, group):
        return group.split("/") if "/" in group else (group, cond)
    csv_path = os.path.join(OUT_DIR, f"gloss_phoneme_dist{sfx}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "group", "speaker", "session", "phoneme", "class",
                    "is_vowel", "count", "pct_speech", "pct_vowels", "dur_s",
                    "pct_dur_speech", "n_speech_total", "n_vowel_total", "n_utt"])
        for k in keys:
            cond, group = k
            spk, sess = _split(cond, group)
            t = tot[k]
            for ph in P.ARPABET:
                n, d = counts[k][ph]
                if n == 0:
                    continue
                is_v = ph in VOWSET
                # sil has no place in the speech denominator -> report it against all
                den = t["n_all"] if ph == "sil" else t["n_speech"]
                dden = t["dur_all"] if ph == "sil" else t["dur_speech"]
                w.writerow([
                    cond, group, spk, sess, ph, PHON_CLASS.get(ph, "?"), int(is_v), n,
                    round(100.0 * n / den, 3) if den else "",
                    round(100.0 * n / t["n_vowel"], 3) if (is_v and t["n_vowel"]) else "",
                    round(d, 3),
                    round(100.0 * d / dden, 3) if dden else "",
                    t["n_speech"], t["n_vowel"], n_utt[k]])
    print(f"[dist] wrote {csv_path}")

    # speaker x condition view: spk2's two post sessions are SUMMED into one `post`,
    # so every speaker has exactly pre/post. `by_session` above keeps them split.
    spk_cond = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    spk_utt = defaultdict(int)
    spk_sess = defaultdict(list)
    for k in keys:
        cond, group = k
        spk = group.split("/")[0]
        spk_utt[(spk, cond)] += n_utt[k]
        spk_sess[(spk, cond)].append(group)
        for ph in P.ARPABET:
            n, d = counts[k][ph]
            spk_cond[(spk, cond)][ph][0] += n
            spk_cond[(spk, cond)][ph][1] += d
    by_spk = {}
    for (spk, cond) in sorted(spk_cond, key=lambda x: (x[0], CONDS.index(x[1]))):
        c = spk_cond[(spk, cond)]
        nsp = sum(c[p][0] for p in SPEECH)
        nvow = sum(c[p][0] for p in VOWEL_PHON)
        nsil = c["sil"][0]
        by_spk.setdefault(spk, {})[cond] = {
            "n_utt": spk_utt[(spk, cond)],
            "sessions": sorted(spk_sess[(spk, cond)]),
            "n_segments_all": nsp + nsil,
            "n_segments_speech": nsp,
            "n_segments_sil": nsil,
            "n_segments_vowel": nvow,
            "n_segments_consonant": nsp - nvow,
            "dur_speech_s": round(sum(c[p][1] for p in SPEECH), 3),
            # the requested payload: per-phoneme sample counts
            "phonemes": {ph: c[ph][0] for ph in P.ARPABET if c[ph][0]},
            "phonemes_pct_speech": {
                ph: round(100.0 * c[ph][0] / nsp, 3)
                for ph in SPEECH if c[ph][0] and nsp},
            "manner": {cl: sum(c[p][0] for p in SPEECH
                               if PHON_CLASS.get(p, "?") == cl) for cl in CLASS_ORDER},
        }

    summary = {
        "min_fps": args.min_fps,
        "utterance_source": src,
        "by_speaker_condition": by_spk,
        "excluded_utts": {_col(k): [{"utt_id": u, "why": w} for u, w in v]
                          for k, v in dropped.items()},
        "n_utt": {_col(k): n_utt[k] for k in keys},
        "totals": {_col(k): tot[k] for k in keys},
        "manner_group_counts_and_shares": groups,
        # n = instances; pct_speech = share of that column's sil-excluded segments
        # (for `sil` itself, share of ALL segments); pct_vowels = share of its vowels
        "phoneme_counts": {
            _col(k): {ph: {
                "n": counts[k][ph][0],
                "pct_speech": round(100.0 * counts[k][ph][0] / (
                    tot[k]["n_all"] if ph == "sil" else tot[k]["n_speech"]), 2)
                    if (tot[k]["n_all"] if ph == "sil" else tot[k]["n_speech"]) else None,
                "dur_s": round(counts[k][ph][1], 3),
            } for ph in P.ARPABET if counts[k][ph][0]}
            for k in keys},
        "vowel_counts": {
            _col(k): {ph: {
                "n": counts[k][ph][0],
                "pct_vowels": round(100.0 * counts[k][ph][0] / tot[k]["n_vowel"], 2)
                    if tot[k]["n_vowel"] else None,
                "pct_speech": round(100.0 * counts[k][ph][0] / tot[k]["n_speech"], 2)
                    if tot[k]["n_speech"] else None,
            } for ph in VOWEL_PHON} for k in keys},
    }
    jp = os.path.join(OUT_DIR, f"gloss_phoneme_dist{sfx}.json")
    json.dump(summary, open(jp, "w"), indent=2)
    print(f"[dist] wrote {jp}")

    plot_vowels(counts, keys, os.path.join(OUT_DIR, f"gloss_vowel_dist{sfx}.png"))


if __name__ == "__main__":
    main()
