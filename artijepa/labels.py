"""Stimulus taxonomy for the USC 75-Speaker Speech MRI database.

Filenames look like:  sub001_2drt_06_rainbow_r1_video.mp4
                      sub001_2drt_12_picture1_video.mp4

We derive two weak labels (used only for the held-out probe during T-SSL, never
for the self-supervised objective itself):

  * stimulus_group : coarse 9-way articulation/elicitation type
  * stimulus_name  : fine 21-way per-stimulus name

These are the only readily-available labels in this corpus (it ships no dense
segmentation / landmark / phoneme annotations), so they serve purely as a cheap
collapse-sanity probe -- see Arti-JEPA-Plans.md B.3.
"""

import re

# Coarse grouping: maps a fine stimulus name (minus trailing _rN repeat index)
# onto a speech-elicitation category.
GROUP_RULES = [
    ("vcv", "vcv"),  # vowel-consonant-vowel sequences
    ("bvt", "bvt"),  # bilabial/velar/etc. articulation task
    ("shibboleth", "shibboleth"),
    ("rainbow", "read_passage"),
    ("grandfather", "read_passage"),
    ("northwind", "read_passage"),
    ("postures", "postures"),
    ("picture", "spontaneous"),  # picture description (spontaneous speech)
    ("topic", "spontaneous"),  # topic monologue (spontaneous speech)
]

GROUPS = sorted({g for _, g in GROUP_RULES})
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUPS)}

_PREFIX_RE = re.compile(r"^sub(\d+)_2drt_(\d+)_(.+?)(?:_r(\d+))?$")


def parse_prefix(prefix: str):
    """Parse 'sub001_2drt_06_rainbow_r1' -> (subject, index, base_name, repeat).

    Returns a dict or None if the prefix does not match the expected pattern.
    """
    m = _PREFIX_RE.match(prefix)
    if not m:
        return None
    subj_num, task_idx, base, repeat = m.groups()
    return {
        "subject": f"sub{subj_num}",
        "task_index": task_idx,
        "stimulus_name": base,  # e.g. rainbow, picture1, vcv2
        "repeat": int(repeat) if repeat is not None else 0,
    }


def stimulus_to_group(stimulus_name: str) -> str:
    """Map a fine stimulus name (e.g. 'picture3', 'vcv2') to its coarse group."""
    name = stimulus_name.lower()
    for token, group in GROUP_RULES:
        if name.startswith(token):
            return group
    return "other"


def filename_to_prefix(filename: str) -> str:
    """sub001_2drt_06_rainbow_r1_video.mp4 -> sub001_2drt_06_rainbow_r1."""
    stem = filename
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    if stem.endswith("_video"):
        stem = stem[: -len("_video")]
    return stem
