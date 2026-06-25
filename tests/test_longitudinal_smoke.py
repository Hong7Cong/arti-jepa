#!/usr/bin/env python
"""Zero-dependency smoke test for the longitudinal corpus plumbing.

Covers the parts that need no GPU and no real video decode:
  1. parse_avi_name -- subject/day + n_frames/fps from the clip filename tokens
  2. merge_manifests -- column reconciliation + subject-collision (leakage) guard

The full clip-decode path is already covered by test_smoke.py (RTMRIVideoDataset
is corpus-agnostic, so a longitudinal row tiles/loads exactly like a 75-speaker
row once it is in the manifest).

Usage:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_longitudinal_smoke.py
"""

import csv
import os
import sys
import tempfile

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({info})" if info else ""))
    return cond


def test_parse():
    from artijepa.build_manifest_longitudinal import parse_avi_name
    p = ("/data/longitudinal/ID07/D39.0/video/"
         "usc_disc_20240418_122021_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_"
         "bitr_recon_narms02_13_sl06mm_nframes0510_lt0.080_ls0.000_lw0.000_"
         "FOV240mm_sRes2.31_tRes12.20_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi")
    subj, day, n, fps = parse_avi_name(p)
    check("parse subject dir", subj == "ID07", subj)
    check("parse day dir", day == "D39.0", day)
    check("parse n_frames", n == 510, n)
    check("parse fps from tRes", fps is not None and abs(fps - 81.967) < 0.01,
          round(fps, 3) if fps else fps)
    # short-named clip: no tokens -> n_frames/fps None (build() will decord-probe it)
    s, d, n2, f2 = parse_avi_name("/data/longitudinal/ID03/D1.0/video/"
                                  "usc_disc_20250306_094109.avi")
    check("short name -> needs probe", n2 is None and f2 is None, f"{n2},{f2}")
    check("short name still has subject/day", s == "ID03" and d == "D1.0", f"{s},{d}")


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_merge():
    from artijepa.merge_manifests import merge
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.csv")
        b = os.path.join(d, "b.csv")
        out = os.path.join(d, "combined.csv")
        # speaker75-like (has split) and longitudinal-like (extra column dropped order)
        _write_csv(a, ["path", "subject", "split"],
                   [{"path": "x.mp4", "subject": "sub001", "split": "train"},
                    {"path": "y.mp4", "subject": "sub002", "split": "val"}])
        _write_csv(b, ["path", "subject", "split", "n_frames"],
                   [{"path": "z.avi", "subject": "longi_ID01", "split": "train",
                     "n_frames": "510"}])
        rows = merge([a, b], out)
        check("merge row count", len(rows) == 3, len(rows))
        with open(out) as f:
            rr = list(csv.DictReader(f))
        check("union columns", set(rr[0].keys()) == {"path", "subject", "split",
                                                     "n_frames"}, list(rr[0].keys()))
        # missing column filled blank for the speaker75 rows
        check("missing cell blank", rr[0]["n_frames"] == "", repr(rr[0]["n_frames"]))
        splits = {r["split"] for r in rr}
        check("splits preserved", splits == {"train", "val"}, splits)

        # leakage guard: same subject in both inputs -> non-strict warns + merges,
        # strict aborts (SystemExit).
        _write_csv(b, ["path", "subject", "split"],
                   [{"path": "z.avi", "subject": "sub001", "split": "train"}])
        rows2 = merge([a, b], out)              # should warn, not raise
        check("overlap merges when non-strict", len(rows2) == 3, len(rows2))
        try:
            merge([a, b], out, strict=True)
            check("strict aborts on overlap", False, "no SystemExit")
        except SystemExit:
            check("strict aborts on overlap", True)


def main():
    print("== longitudinal smoke test ==")
    print("[1] parse_avi_name"); test_parse()
    print("[2] merge_manifests"); test_merge()
    n_pass = sum(ok for _, ok in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILED:", [n for n, ok in results if not ok])
        sys.exit(1)
    print("ALL LONGITUDINAL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
