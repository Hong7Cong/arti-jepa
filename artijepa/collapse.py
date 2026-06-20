"""Representation-collapse diagnostics (Arti-JEPA B.3/F), label-free.

Collapse is *the* JEPA failure mode. During T-SSL we periodically measure, on a
frozen snapshot of the current encoder over held-out clips:

  * feature_std        -- mean per-dimension std of pooled features (-> 0 = collapse)
  * effective_rank     -- exp(entropy of normalized singular values); low = collapse
  * mean_abs_cosine    -- average |cosine| between clips (-> 1 = collapse)

These need no labels. Downstream usefulness is measured separately by the
phoneme-prediction eval (``eval_phoneme.py``); the old weak stimulus-group linear
probe was removed as not meaningful.
"""

import numpy as np
import torch


def simple_collate(batch):
    """(clips_list, label, idx) -> ([B,3,T,H,W], LongTensor[B])."""
    clips = torch.stack([b[0][0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return clips, labels


@torch.no_grad()
def extract_features(encoder, loader, device, max_batches=None, dtype=torch.float32):
    """Mean-pooled encoder features over a loader -> (feats [N,D], labels [N])."""
    encoder.eval()
    feats, labels = [], []
    for i, (clips, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        clips = clips.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32)):
            tokens = encoder.backbone(clips)          # [B, N, D]
        pooled = tokens.float().mean(dim=1)            # [B, D]
        feats.append(pooled.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def feature_diagnostics(feats: torch.Tensor):
    """Collapse metrics from a [N, D] feature matrix."""
    f = feats.float()
    std = f.std(dim=0).mean().item()
    # effective rank via singular-value entropy of the centered matrix
    fc = f - f.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(fc)
    p = sv / (sv.sum() + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    eff_rank = float(torch.exp(entropy))
    # mean absolute pairwise cosine (subsample for cost)
    n = min(len(f), 256)
    idx = torch.randperm(len(f))[:n]
    fn = torch.nn.functional.normalize(f[idx], dim=1)
    cos = (fn @ fn.t())
    off = cos[~torch.eye(n, dtype=torch.bool)]
    mean_abs_cos = off.abs().mean().item()
    return {
        "feature_std": std,
        "effective_rank": eff_rank,
        "mean_abs_cosine": mean_abs_cos,
        "dim": f.shape[1],
    }
