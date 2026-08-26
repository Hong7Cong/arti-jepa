"""Minimal example: load an Arti-JEPA encoder from a checkpoint and run it.

Arti-JEPA is a V-JEPA 2 ViT adapted to rt-MRI vocal-tract video, so loading is
two steps: (1) build the encoder at the geometry the checkpoint was trained at,
(2) copy the weights in. A checkpoint holds several state dicts (`encoder` =
the online encoder, `target_encoder` = its EMA copy, `predictor`, optimizer
state); for feature extraction you want `target_encoder`.

Everything it needs is inside the installed package -- the V-JEPA 2 subset is
vendored under `artijepa._vendor` -- so this runs from anywhere:

    pip install "git+https://github.com/Hong7Cong/arti-jepa.git"
    python examples/load_artijepa.py
"""

import torch

from artijepa.checkpoint import filtered_load
from artijepa.model import build_models

# --- settings: edit these ---------------------------------------------------
# The geometry must match what the checkpoint was trained at -- the RoPE grid is
# baked in at construction time, so a mismatch silently loads nothing.
CHECKPOINT = "/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_215.pt"
MODEL_NAME = "vit_large"
SPATIAL_SIZE = 256
FRAMES_PER_CLIP = 32
PATCH_SIZE = 16
TUBELET_SIZE = 2
BATCH_SIZE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build the encoder. `build_models` also returns the JEPA predictor, which
#    inference doesn't need.
encoder, _predictor = build_models(
    device=device,
    model_name=MODEL_NAME,
    spatial_size=SPATIAL_SIZE,
    frames_per_clip=FRAMES_PER_CLIP,
    patch_size=PATCH_SIZE,
    tubelet_size=TUBELET_SIZE,
    num_mask_tokens=1,
    use_activation_checkpointing=False,   # off for inference
)

# 2. Load the weights. In an Arti-JEPA checkpoint the keys are already
#    `backbone.*`, so they go onto the wrapper (`encoder`), not `encoder.backbone`.
#    `filtered_load` copies only shape-matching keys, so a geometry mismatch shows
#    up as a large "missing" count instead of a crash.
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
n, missing, skipped = filtered_load(encoder, ckpt["target_encoder"])
print(f"[load] epoch {ckpt.get('epoch', '?')}: {n} tensors loaded, "
      f"{len(missing)} missing, {len(skipped)} skipped")
del ckpt

# Freeze -- this example only extracts features.
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# 3. Random input in the layout the model expects: [B, C, T, H, W]. rt-MRI is
#    grayscale but the ViT takes 3 channels -- the real dataloader replicates the
#    gray frame 3x, so a random tensor stands in fine here.
clip = torch.randn(BATCH_SIZE, 3, FRAMES_PER_CLIP, SPATIAL_SIZE, SPATIAL_SIZE,
                   device=device)

# 4. Forward. Call `.backbone` for the plain ViT -- the wrapper around it expects
#    the list-of-clips + masks interface used during training. Output is the
#    temporal-major token grid [B, T'*S', D], with T' = FRAMES_PER_CLIP // TUBELET_SIZE
#    and S' = (SPATIAL_SIZE // PATCH_SIZE) ** 2.
with torch.no_grad():
    tokens = encoder.backbone(clip)

Tp = FRAMES_PER_CLIP // TUBELET_SIZE
Sp = (SPATIAL_SIZE // PATCH_SIZE) ** 2
print(f"input  {tuple(clip.shape)}")
print(f"tokens {tuple(tokens.shape)}  (T'={Tp} x S'={Sp} = {Tp * Sp}, D={tokens.shape[-1]})")

# Typical downstream use: mean-pool the tokens into one clip embedding [B, D].
# (The evals instead keep the grid and train an attentive probe over it.)
print(f"clip embedding {tuple(tokens.mean(dim=1).shape)}")
