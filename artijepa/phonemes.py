"""Phoneme inventory, time alignment, and sequence metrics (Arti-JEPA eval).

Replaces the dropped stimulus-group probe with phoneme prediction -- the real
articulatory test of whether the JEPA video representation encodes speech
content. Two evals share this module:

  * **Task 2 (gold):** hand-annotated ARPABET phonemes with timestamps for one
    held-out OOD speaker (`/data1/span_data/usc_lss`; 104x104 video @ 99 fps,
    16 kHz audio, 684 utterances).
  * **Task 1 (pseudo):** phoneme labels from an audio model (wav2vec2 / WavLM CTC)
    run on the 75-speaker corpus's paired audio (`audio_phoneme.py`).

**Alignment is done entirely in SECONDS, hence frame-rate agnostic** -- this is
the key to handling the 99-fps OOD speaker and the 83.28-fps training corpus with
one code path. A clip is resampled onto the `target_fps` grid that feeds the
encoder (e.g. 25 fps), so output-frame *f* sits at time `f / target_fps`. With
`tubelet_size = 2` the encoder emits one temporal token per 2 frames, so token
*j* spans `[2j, 2j+2) / target_fps` seconds. We label each temporal token by the
phoneme covering its time-centre. (Audio models emit ~50 Hz / 20 ms frames -> two
per 25-fps video frame, four per 80 ms token; this routine subsumes that.)
"""

import bisect
import json

import numpy as np

# ARPABET inventory observed in usc_lss (data-driven, 41 symbols incl. silence).
# Fixed canonical order for reproducible label indices.
ARPABET = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey",
    "f", "g", "h", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy",
    "p", "r", "s", "sh", "sil", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh",
]
PHON2IDX = {p: i for i, p in enumerate(ARPABET)}
IDX2PHON = {i: p for p, i in PHON2IDX.items()}
SIL = "sil"
SIL_IDX = PHON2IDX[SIL]
IGNORE_INDEX = -100          # tokens outside the utterance / padded clips
NUM_PHONEMES = len(ARPABET)


# --------------------------------------------------------------------------- #
# loading + per-token alignment
# --------------------------------------------------------------------------- #
def load_gold_segments(json_path):
    """Read a usc_lss phoneme json -> [(phoneme, start_s, end_s), ...] sorted."""
    segs = []
    for s in json.load(open(json_path)):
        segs.append((s["phoneme"].lower(), float(s["start"]), float(s["end"])))
    segs.sort(key=lambda x: x[1])
    return segs


def token_center_times(n_tokens, tubelet, target_fps, clip_start_frame=0):
    """Centre time (s) of each temporal token, in the utterance timeline.

    Token j covers output frames [clip_start_frame + j*tubelet,
    + (j+1)*tubelet); its centre frame is + tubelet/2.
    """
    j = np.arange(n_tokens, dtype=np.float64)
    return (clip_start_frame + j * tubelet + tubelet / 2.0) / float(target_fps)


def phoneme_at(segments, t, starts=None):
    """Phoneme (str) of the segment covering time t, or None if outside."""
    if not segments:
        return None
    if starts is None:
        starts = [s[1] for s in segments]
    i = bisect.bisect_right(starts, t) - 1          # last seg with start <= t
    if i < 0:
        return segments[0][0] if t >= segments[0][1] - 1e-9 else None
    ph, s0, s1 = segments[i]
    if s0 - 1e-9 <= t < s1 + 1e-9:
        return ph
    # t past the last segment end -> outside
    return None


def segments_to_token_labels(segments, n_tokens, tubelet, target_fps,
                             clip_start_frame=0, outside=IGNORE_INDEX):
    """Per temporal-token phoneme indices for one clip (length n_tokens).

    Tokens whose centre falls outside any annotated segment get ``outside``
    (default IGNORE_INDEX); silence maps to SIL_IDX like any other phoneme.
    """
    times = token_center_times(n_tokens, tubelet, target_fps, clip_start_frame)
    starts = [s[1] for s in segments]
    out = np.full(n_tokens, outside, dtype=np.int64)
    for k, t in enumerate(times):
        ph = phoneme_at(segments, float(t), starts)
        if ph is not None:
            out[k] = PHON2IDX.get(ph, outside)
    return out


def reference_sequence(segments, drop_sil=True):
    """Gold phoneme-index sequence for an utterance (one symbol per segment)."""
    seq = [PHON2IDX[p] for p, _, _ in segments if p in PHON2IDX]
    if drop_sil:
        seq = [i for i in seq if i != SIL_IDX]
    return seq


# --------------------------------------------------------------------------- #
# sequence metrics
# --------------------------------------------------------------------------- #
def collapse_sequence(idx_seq, drop=(SIL_IDX,), ignore=IGNORE_INDEX):
    """Frame/token label stream -> phoneme sequence: merge runs, drop sil/ignore."""
    out, prev = [], object()
    drop = set(drop) | {ignore}
    for i in idx_seq:
        i = int(i)
        if i == prev:
            continue
        prev = i
        if i not in drop:
            out.append(i)
    return out


def edit_distance(ref, hyp):
    """Levenshtein distance between two index sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def phoneme_error_rate(ref_seq, hyp_seq):
    """PER = Levenshtein(ref, hyp) / len(ref). Returns (per, n_ref)."""
    n = len(ref_seq)
    if n == 0:
        return (0.0 if len(hyp_seq) == 0 else 1.0), 0
    return edit_distance(ref_seq, hyp_seq) / n, n


def cohen_kappa(y_true, y_pred, num_classes=NUM_PHONEMES, ignore=IGNORE_INDEX):
    """Chance-corrected per-token agreement on valid (non-ignore) tokens."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    m = (y_true != ignore) & (y_pred != ignore)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) == 0:
        return 0.0
    po = float((yt == yp).mean())
    nt = np.bincount(yt, minlength=num_classes).astype(np.float64) / len(yt)
    npd = np.bincount(yp, minlength=num_classes).astype(np.float64) / len(yp)
    pe = float((nt * npd).sum())
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0


def frame_accuracy(y_true, y_pred, ignore=IGNORE_INDEX):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    m = (y_true != ignore)
    return float((y_true[m] == y_pred[m]).mean()) if m.any() else 0.0
