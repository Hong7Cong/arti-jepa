#!/usr/bin/env python
"""Zero-dependency smoke test for the Arti-JEPA pipeline (run with the artijepa env).

Validates, on real rtMRI data but with a tiny ViT on CPU:
  1. label parsing / manifest build / subject-disjoint splits
  2. RTMRIVideoDataset clip shapes for resize(256) and pad(96), temporal interp
  3. MaskCollator token-grid bookkeeping
  4. one full T-SSL step (encoder -> predictor -> L1 loss -> EMA) + diagnostics

Usage:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_smoke.py
"""

import csv
import os
import sys
import tempfile

import numpy as np
import torch

DATA_ROOT = os.environ.get("ARTI_DATA_ROOT", "/scratch1/hongn/speaker75")
OUT = os.environ.get("ARTI_OUT", "/scratch1/hongn/artijepa")

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({info})" if info else ""))
    return cond


def test_labels():
    from artijepa.labels import filename_to_prefix, parse_prefix, stimulus_to_group
    p = parse_prefix(filename_to_prefix("sub012_2drt_06_rainbow_r2_video.mp4"))
    check("parse subject", p["subject"] == "sub012", p["subject"])
    check("parse stimulus", p["stimulus_name"] == "rainbow", p["stimulus_name"])
    check("parse repeat", p["repeat"] == 2)
    check("group read_passage", stimulus_to_group("rainbow") == "read_passage")
    check("group spontaneous", stimulus_to_group("picture3") == "spontaneous")


def build_smoke_manifest():
    from artijepa.build_manifest import build
    from artijepa.splits import assign_splits
    man = os.path.join(OUT, "smoke_manifest.csv")
    split = os.path.join(OUT, "smoke_manifest_split.csv")
    os.makedirs(OUT, exist_ok=True)
    rows = build(DATA_ROOT, man, probe=False)
    check("manifest non-empty", len(rows) > 100, f"{len(rows)} rows")
    subjects = {r["subject"] for r in rows}
    check("multiple subjects", len(subjects) >= 10, f"{len(subjects)} subjects")
    mapping = assign_splits(subjects, 0.12, 0.12, seed=0)
    with open(man) as f:
        rr = list(csv.DictReader(f))
    for r in rr:
        r["split"] = mapping[r["subject"]]
    with open(split, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
        w.writeheader(); w.writerows(rr)
    # subject-disjointness
    by_split = {}
    for r in rr:
        by_split.setdefault(r["split"], set()).add(r["subject"])
    disjoint = (by_split.get("train", set()) & by_split.get("test", set())) == set()
    check("splits subject-disjoint", disjoint)
    return split


def test_dataset(split):
    from artijepa.rtmri_dataset import PreprocConfig, RTMRIVideoDataset
    # resize 256 / 32 frames
    cfg = PreprocConfig(spatial_size=256, spatial_mode="resize", frames_per_clip=32,
                        augment=True)
    ds = RTMRIVideoDataset(split, "train", cfg, seed=0)
    clips, label, idx = ds[0]
    c = clips[0]
    check("resize clip shape", tuple(c.shape) == (3, 32, 256, 256), tuple(c.shape))
    check("clip_indices len == fpc", len(idx[-1]) == 32, len(idx[-1]))
    check("channels replicated", torch.allclose(c[0], c[1]) and torch.allclose(c[1], c[2]))
    check("finite values", torch.isfinite(c).all().item())
    # pad 96 / 16 frames
    cfg2 = PreprocConfig(spatial_size=96, spatial_mode="pad", frames_per_clip=16,
                        augment=False)
    ds2 = RTMRIVideoDataset(split, "train", cfg2, seed=0)
    c2 = ds2[0][0][0]
    check("pad clip shape", tuple(c2.shape) == (3, 16, 96, 96), tuple(c2.shape))
    # temporal interpolation: source indices should span ~ (N-1)*ratio native frames
    s = ds2._sample_source_indices(3000, 83.28, np.random.default_rng(0))
    span = s.max() - s.min()
    expect = (16 - 1) * (83.28 / 50.0)
    check("temporal interp span", abs(span - expect) < 2.0, f"span={span:.1f} exp={expect:.1f}")
    # tile mode (full coverage @ target_fps, split into chunks) -- needs n_frames/fps
    full = os.path.join(OUT, "manifest_split.csv")
    if os.path.exists(full):
        cfgt = PreprocConfig(spatial_size=96, spatial_mode="pad", frames_per_clip=16,
                             target_fps=25.0, sampling="tile", augment=False)
        dst = RTMRIVideoDataset(full, "val", cfgt, seed=0)
        ct = dst[0][0]
        check("tile clip shape", tuple(ct[0].shape) == (3, 16, 96, 96), tuple(ct[0].shape))
        check("tile one chunk/item", len(ct) == 1, len(ct))
        check("tile index > #videos", len(dst) > len(dst.rows), f"{len(dst)}>{len(dst.rows)}")
        ri = dst.index[0][0]
        s0 = dst._tile_indices(dst.rows[ri], 0); s1 = dst._tile_indices(dst.rows[ri], 1)
        step = float(dst.rows[ri]["fps"]) / 25.0
        check("tile chunks contiguous", abs((s1[0] - s0[-1]) - step) < 1e-6,
              f"gap={s1[0]-s0[-1]:.2f} step={step:.2f}")
    return ds


def test_mask_collator(ds):
    from src.masks.multiseq_multiblock3d import MaskCollator
    from artijepa.masking import mask_config_for
    cfgs_mask = mask_config_for(256 // 16)  # 16x16 grid
    mc = MaskCollator(cfgs_mask=cfgs_mask, dataset_fpcs=[32], crop_size=256,
                      patch_size=16, tubelet_size=2)
    batch = [ds[0], ds[1]]
    fpc_collations = mc(batch)
    check("one fpc bucket", len(fpc_collations) == 1, len(fpc_collations))
    collated, m_enc, m_pred = fpc_collations[0]
    check("two mask generators", len(m_enc) == len(cfgs_mask) == 2)
    grid = (32 // 2) * (256 // 16) * (256 // 16)
    enc0, pred0 = m_enc[0], m_pred[0]
    ok = enc0.shape[1] + pred0.shape[1] <= grid and enc0.shape[1] > 0 and pred0.shape[1] > 0
    check("enc+pred tokens within grid", ok,
          f"enc={tuple(enc0.shape)} pred={tuple(pred0.shape)} grid={grid}")


def test_train_step(split):
    """One real T-SSL step with a tiny ViT on CPU via the actual trainer."""
    from artijepa.tssl_train import train
    cfg = {
        "meta": {"folder": os.path.join(OUT, "runs/smoke"), "seed": 0,
                 "dtype": "float32", "eval_freq": 1, "save_freq": 1,
                 "probe_label": "group_idx", "probe_max_batches": 2, "max_steps": 2},
        "data": {"manifest": split, "grayscale_stats": None, "spatial_size": 96,
                 "spatial_mode": "pad", "frames_per_clip": 16, "target_fps": 50.0,
                 "tubelet_size": 2, "patch_size": 16, "intensity_norm": "zscore",
                 "augment": True, "batch_size": 2, "probe_batch_size": 4,
                 "num_workers": 0, "pin_mem": False, "num_classes": 9},
        "model": {"model_name": "vit_tiny", "pretrained": False,
                  "use_activation_checkpointing": False},
        "mask": None, "loss": {"loss_exp": 1.0},
        "optimization": {"epochs": 1, "ipe": 2, "lr": 5e-4, "start_lr": 1e-4,
                         "final_lr": 1e-5, "warmup": 1, "weight_decay": 0.04,
                         "final_weight_decay": 0.04, "ema": [0.996, 1.0],
                         "ipe_scale": 1.0, "betas": [0.9, 0.999], "eps": 1e-8},
    }
    train(cfg)
    diag = os.path.join(OUT, "runs/smoke/diagnostics.jsonl")
    check("diagnostics written", os.path.exists(diag))
    check("checkpoint written", os.path.exists(os.path.join(OUT, "runs/smoke/latest.pt")))


def main():
    print("== Arti-JEPA smoke test ==")
    print("[1] labels");          test_labels()
    print("[2] manifest+splits"); split = build_smoke_manifest()
    print("[3] dataset");         ds = test_dataset(split)
    print("[4] mask collator");   test_mask_collator(ds)
    print("[5] T-SSL train step (vit_tiny, CPU)"); test_train_step(split)

    n_pass = sum(ok for _, ok in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILED:", [n for n, ok in results if not ok])
        sys.exit(1)
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
