"""Build the V-JEPA 2 encoder+predictor for T-SSL and load pretrained weights.

Reuses the parent repo's ``app.vjepa.utils.init_video_model`` (which wraps the
ViT in ``MultiSeqWrapper`` and the predictor in ``PredictorMultiSeqWrapper`` --
the exact interface the V-JEPA training step expects).

Note: the predictor's RoPE grid is fixed at construction, so the model must be
built with ``spatial_size`` / ``frames_per_clip`` matching the clips it will see.
"""

import copy

import torch

from artijepa.checkpoint import clean_backbone_key, filtered_load


def build_models(
    device,
    model_name="vit_large",
    spatial_size=256,
    frames_per_clip=32,
    patch_size=16,
    tubelet_size=2,
    pred_depth=12,
    pred_embed_dim=384,
    pred_num_heads=12,
    num_mask_tokens=2,
    use_activation_checkpointing=True,
    use_sdpa=True,
):
    from app.vjepa.utils import init_video_model

    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=spatial_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=True,
        use_sdpa=use_sdpa,
        use_silu=False,
        use_pred_silu=False,
        wide_silu=True,
        use_rope=True,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    return encoder, predictor


def load_pretrained(encoder, predictor, ckpt_path, checkpoint_key="target_encoder",
                    verbose=True):
    """Initialise encoder (and predictor where shapes match) from a V-JEPA2 ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint_key not in ckpt:
        # fall back to whichever encoder key exists
        for k in ("target_encoder", "encoder", "ema_encoder"):
            if k in ckpt:
                checkpoint_key = k
                break
    enc_sd = clean_backbone_key(ckpt[checkpoint_key])
    n_enc, miss_enc, skip_enc = filtered_load(encoder.backbone, enc_sd)
    if verbose:
        print(f"[load] encoder<-{checkpoint_key}: loaded {n_enc} tensors, "
              f"{len(miss_enc)} missing, {len(skip_enc)} skipped")
        if miss_enc:
            print(f"        e.g. missing: {miss_enc[:4]}")

    if "predictor" in ckpt:
        pred_sd = clean_backbone_key(ckpt["predictor"])
        n_p, miss_p, skip_p = filtered_load(predictor.backbone, pred_sd)
        if verbose:
            print(f"[load] predictor: loaded {n_p} tensors, "
                  f"{len(miss_p)} missing, {len(skip_p)} skipped "
                  f"(mask_tokens reinit if grid/mask-count differs)")
    del ckpt
    return encoder, predictor


def make_target_encoder(encoder):
    """EMA target = frozen deep copy of the (loaded) context encoder."""
    target = copy.deepcopy(encoder)
    for p in target.parameters():
        p.requires_grad = False
    return target
