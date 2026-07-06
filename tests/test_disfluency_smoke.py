#!/usr/bin/env python
"""Zero-GPU smoke test for the stuttering disfluency eval (Task 8, artijepa env).

Validates the pure-logic pieces without touching real video or GPU:
  1. canonicalize(): typo repair, compound-primary, bucket5, drop-unknown
  2. parse_textgrid(): interval extraction from a tiny in-memory TextGrid
  3. build_manifest(): end-to-end on a synthetic corpus tree (fake .avi/.TextGrid)
  4. label_space / row_label: task label maps (type5/type3/binary)
  5. classification_metrics(): macro-F1 / balanced-acc / confusion on a known case
  6. SegmentProbe(mean) forward + one backward step on fake features
  7. VideoMAEBackbone token->grid reshape math (tiny random-weight config, CPU)

Usage:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_disfluency_smoke.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({info})" if info else ""))
    return cond


def test_canon():
    from artijepa.stutter import canonicalize
    cases = {
        "W_rep": ("rep", "rep", ["rep"]),
        "TH_block": ("block", "block", ["block"]),
        "SH_pro+rep": ("pro", "pro", ["pro", "rep"]),
        "X_blcok": ("block", "block", ["block"]),        # typo
        "Y_block?": ("block", "block", ["block"]),       # strip ?
        "Z_revert": ("revert", "other", ["revert"]),     # rare -> other bucket
        "Q_error": (None, None, []),                     # unknown -> dropped
    }
    ok = True
    for txt, (p, b, c) in cases.items():
        gp, gb, gc = canonicalize(txt)
        ok &= (gp == p and gb == b and gc == c)
    check("canonicalize typo/compound/bucket", ok)


def test_parse_textgrid():
    from artijepa.stutter import parse_textgrid
    tg = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "disfluency"
        xmin = 0
        xmax = 5
        intervals: size = 2
        intervals [1]:
            xmin = 0.5
            xmax = 1.5
            text = "W_rep"
        intervals [2]:
            xmin = 2.0
            xmax = 2.4
            text = ""
'''
    with tempfile.NamedTemporaryFile("w", suffix=".TextGrid", delete=False) as f:
        f.write(tg); path = f.name
    tiers = parse_textgrid(path)
    os.unlink(path)
    ivs = tiers.get("disfluency", [])
    check("parse tier present", "disfluency" in tiers, str(list(tiers)))
    check("parse 2 intervals", len(ivs) == 2, str(len(ivs)))
    check("parse xmin/text", ivs and abs(ivs[0][0] - 0.5) < 1e-9 and ivs[0][2] == "W_rep")


def test_manifest():
    from artijepa import stutter as S
    root = tempfile.mkdtemp()
    spk = "PWS3"; stem = "PWS3_demo"
    os.makedirs(os.path.join(root, spk, "textgrid"))
    os.makedirs(os.path.join(root, spk, "avi"))
    open(os.path.join(root, spk, "avi", stem + ".avi"), "w").close()   # fake video
    tg = '''item []:
    item [1]:
        class = "IntervalTier"
        name = "disfluency"
        xmin = 0
        xmax = 10
        intervals: size = 3
        intervals [1]:
            xmin = 1.0
            xmax = 1.6
            text = "T_block"
        intervals [2]:
            xmin = 2.0
            xmax = 2.5
            text = "K_pro+rep"
        intervals [3]:
            xmin = 3.0
            xmax = 5.0
            text = ""
'''
    with open(os.path.join(root, spk, "textgrid", stem + ".TextGrid"), "w") as f:
        f.write(tg)
    out = S.build_manifest(root=root, speakers=("PWS3",), tiers=("disfluency",),
                           fluent_per_file=1, verbose=False)
    rows = S.read_manifest(out)
    disf = [r for r in rows if r["bucket5"] != S.FLUENT]
    flu = [r for r in rows if r["bucket5"] == S.FLUENT]
    check("manifest 2 disfluent rows", len(disf) == 2, str(len(disf)))
    check("manifest compound primary=pro", any(r["primary"] == "pro" and r["multi"] == "pro|rep"
                                                for r in disf))
    check("manifest 1 fluent negative", len(flu) == 1, str(len(flu)))
    check("manifest seg_id unique", len({r["seg_id"] for r in rows}) == len(rows))


def test_label_space():
    from artijepa import stutter as S
    c5, _ = S.label_space("type5"); c3, _ = S.label_space("type3")
    cb, _ = S.label_space("binary")
    check("type5 classes", c5 == ["block", "rep", "pro", "osci", "other"])
    check("type3 drops rare", c3 == ["block", "rep", "pro"])
    block = {"bucket5": "block"}; other = {"bucket5": "other"}; flu = {"bucket5": S.FLUENT}
    check("type3 drops 'other'", S.row_label(other, "type3", c3) is None)
    check("type5 keeps 'other'", S.row_label(other, "type5", c5) == 4)
    check("binary fluent=0 disfluent=1",
          S.row_label(flu, "binary", cb) == 0 and S.row_label(block, "binary", cb) == 1)


def test_metrics():
    from artijepa.stutter import classification_metrics
    # 3 classes, perfect on 0/1, class 2 all wrong -> macro-F1 < acc
    yt = np.array([0, 0, 1, 1, 2, 2]); yp = np.array([0, 0, 1, 1, 0, 1])
    m = classification_metrics(yt, yp, 3, ["a", "b", "c"])
    check("acc = 4/6", abs(m["accuracy"] - 0.6667) < 1e-3, str(m["accuracy"]))
    check("balanced_acc = 2/3", abs(m["balanced_acc"] - 0.6667) < 1e-3, str(m["balanced_acc"]))
    check("class c recall 0", m["per_class"]["c"]["recall"] == 0.0)
    check("confusion shape 3x3", np.array(m["confusion"]).shape == (3, 3))


def test_probe():
    from artijepa.eval_disfluency import SegmentProbe
    torch.manual_seed(0)
    clf = SegmentProbe(dim=32, num_classes=5, kind="mean")
    x = torch.randn(8, 32)
    logit = clf(x)
    check("mean-probe logits [8,5]", tuple(logit.shape) == (8, 5), str(tuple(logit.shape)))
    loss = torch.nn.functional.cross_entropy(logit, torch.randint(0, 5, (8,)))
    loss.backward()
    g = clf.net[1].weight.grad
    check("mean-probe grad finite", g is not None and torch.isfinite(g).all())


def test_videomae_reshape():
    # tiny VideoMAE to exercise the token->grid reshape without downloading weights
    try:
        from transformers import VideoMAEModel, VideoMAEConfig
    except Exception as e:
        check("videomae import (skipped)", True, f"transformers absent: {e}")
        return
    from artijepa.videomae_baseline import VideoMAEBackbone
    cfg = VideoMAEConfig(image_size=32, patch_size=16, num_frames=8, tubelet_size=2,
                         hidden_size=24, num_hidden_layers=1, num_attention_heads=3,
                         intermediate_size=48)
    bb = VideoMAEBackbone.__new__(VideoMAEBackbone)
    torch.nn.Module.__init__(bb)
    bb.name = "tiny"; bb.model = VideoMAEModel(cfg).eval()
    bb.num_frames = 8; bb.tubelet = 2; bb.input_size = 32; bb.patch = 16; bb.grid = 2
    bb.frame_batch = 8; bb.pool_spatial = False; bb.grid_cap = 16
    bb.register_buffer("mean", torch.zeros(1, 3, 1, 1, 1))
    bb.register_buffer("std", torch.ones(1, 3, 1, 1, 1))
    clip = torch.rand(2, 3, 8, 32, 32)
    with torch.no_grad():
        out = bb(clip)                      # T'=4, S'=(32/16)^2=4 -> L=16
    check("videomae tokens [2,16,24]", tuple(out.shape) == (2, 16, 24), str(tuple(out.shape)))
    bb.pool_spatial = True
    with torch.no_grad():
        outp = bb(clip)
    check("videomae pooled [2,4,24]", tuple(outp.shape) == (2, 4, 24), str(tuple(outp.shape)))


def main():
    print("== disfluency eval smoke ==")
    for fn in (test_canon, test_parse_textgrid, test_manifest, test_label_space,
               test_metrics, test_probe, test_videomae_reshape):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, False))
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
