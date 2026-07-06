"""Compute global grayscale mean/std over the TRAIN split (Arti-JEPA A.1 step 3).

These are rtMRI-specific channel-normalization stats (do NOT reuse ImageNet/
natural-video values). Computed on clips *after* per-clip intensity
standardization, so they typically land near (0, 1) -- their job is to pin a
common range that is applied at load time and reused across every experiment.

Usage:
    python -m artijepa.compute_stats \
        --manifest /data2/hongn/artijepa/manifest_split.csv \
        --out /data2/hongn/artijepa/grayscale_stats.json \
        --spatial-size 256 --max-clips 300
"""

import argparse
import json

import torch

from artijepa.rtmri_dataset import PreprocConfig, RTMRIVideoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data2/hongn/artijepa/manifest_split.csv")
    ap.add_argument("--out", default="/data2/hongn/artijepa/grayscale_stats.json")
    ap.add_argument("--spatial-size", type=int, default=256)
    ap.add_argument("--spatial-mode", default="resize")
    ap.add_argument("--frames-per-clip", type=int, default=32)
    ap.add_argument("--max-clips", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PreprocConfig(
        spatial_size=args.spatial_size,
        spatial_mode=args.spatial_mode,
        frames_per_clip=args.frames_per_clip,
        augment=False,
        random_temporal_crop=False,
        grayscale_mean=0.0,
        grayscale_std=1.0,
    )
    ds = RTMRIVideoDataset(args.manifest, split="train", cfg=cfg, seed=args.seed)
    n = min(args.max_clips, len(ds))
    print(f"Accumulating stats over {n}/{len(ds)} train clips...")

    count = 0
    s1 = 0.0   # sum
    s2 = 0.0   # sum of squares
    idxs = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
    for j, i in enumerate(idxs):
        clips, _, _ = ds[i]
        x = clips[0][0]  # one channel, [T,H,W] (channels are identical pre-norm)
        s1 += float(x.sum())
        s2 += float((x * x).sum())
        count += x.numel()
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{n}")

    mean = s1 / count
    var = max(s2 / count - mean * mean, 1e-12)
    std = var ** 0.5
    stats = {
        "mean": mean, "std": std, "n_pixels": count, "n_clips": n,
        "spatial_size": args.spatial_size, "spatial_mode": args.spatial_mode,
        "frames_per_clip": args.frames_per_clip,
    }
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"mean={mean:.5f} std={std:.5f} -> {args.out}")


if __name__ == "__main__":
    main()
