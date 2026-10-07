"""Minimal example: load a trained consonant-identity probe (Phase 3,
`consonants` task) and run it on top of its frozen Arti-JEPA encoder.

The probe is a `ClipProbe` (artijepa/eval_phoneme_groups.py): a V-JEPA
AttentivePooler collapses each temporal step's spatial tokens to one vector,
then a bi-LSTM runs over the phoneme's variable-length temporal-token sequence
and its last hidden state (both directions) is the clip embedding fed to a
linear head -> 25-way consonant-identity logits. The checkpoint stores this
head's `state_dict` PLUS metadata (`class_names`, `dim`, `hidden`, ..., and
`encoder_spec` = the exact frozen-encoder checkpoint it was trained against),
so loading it is: (1) rebuild the encoder at the recorded geometry, (2) rebuild
the probe at the recorded architecture, (3) load both state dicts.

Everything it needs is inside the installed package -- the V-JEPA 2 subset is
vendored under `artijepa._vendor` -- so this runs from anywhere:

    pip install "git+https://github.com/Hong7Cong/arti-jepa.git"
    python examples/load_probe_consonants.py
"""

import torch

from artijepa.checkpoint import filtered_load
from artijepa.eval_phoneme_groups import load_probe
from artijepa.model import build_models

# --- settings: edit these ---------------------------------------------------
# `tssl256comb215` = the T-SSL ViT-L/256 encoder (same run as examples/load_artijepa.py).
# Other tags trained on the same task/data: `pretrained` (no T-SSL), `videomae`,
# `videomae_rtmri` -- swap the filename to try one. (`pretrained`'s `encoder_spec`
# is the literal string "pretrained", not a checkpoint path -- it needs the
# different resolve_checkpoint()/clean_backbone_key() loading branch used in
# artijepa.eval_phoneme.load_frozen_encoder, not shown in this minimal example.)
PROBE_CHECKPOINT = "/scratch1/hongn/artijepa/eval/phgroups/phgroups_tssl256comb215_s0_consonants.pt"
MODEL_NAME = "vit_large"
PATCH_SIZE = 16
TUBELET_SIZE = 2
BATCH_SIZE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the probe. `load_probe` reconstructs `ClipProbe` from the checkpoint's
#    recorded architecture (dim/hidden/layers/heads/dropout) and returns it
#    already in `eval()` mode, plus the full metadata dict.
probe, meta = load_probe(PROBE_CHECKPOINT, device=device)
class_names = meta["class_names"]
val_acc = meta["metrics"]["val"]["accuracy"]
print(f"[load] task={meta['task']} ({meta['num_classes']}-way) classes={class_names}")
print(f"[load] best_epoch={meta['best_epoch']} val_accuracy={val_acc:.3f}")

# 2. Build the encoder the probe was trained on top of, at the SAME geometry
#    (`spatial_size`/`frames_per_clip` are recorded in the checkpoint because
#    the RoPE grid is baked in at construction -- a mismatch silently loads
#    nothing, same gotcha as examples/load_artijepa.py).
encoder, _predictor = build_models(
    device=device,
    model_name=MODEL_NAME,
    spatial_size=meta["spatial_size"],
    frames_per_clip=meta["frames_per_clip"],
    patch_size=PATCH_SIZE,
    tubelet_size=TUBELET_SIZE,
    num_mask_tokens=1,
    use_activation_checkpointing=False,
)
ckpt = torch.load(meta["encoder_spec"], map_location="cpu", weights_only=False)
n, missing, skipped = filtered_load(encoder, ckpt["target_encoder"])
print(f"[load] encoder <- {meta['encoder_spec']}: {n} tensors loaded, "
      f"{len(missing)} missing, {len(skipped)} skipped")
del ckpt

encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# 3. Random input clip at the probe's native geometry: [B, C, T, H, W]. Real
#    phoneme clips are variable-length (`lens` below); here every clip in the
#    batch fills the whole window, i.e. lens == T'.
Tp = meta["frames_per_clip"] // TUBELET_SIZE
Sp = (meta["spatial_size"] // PATCH_SIZE) ** 2
clip = torch.randn(BATCH_SIZE, 3, meta["frames_per_clip"], meta["spatial_size"],
                   meta["spatial_size"], device=device)
lens = torch.full((BATCH_SIZE,), Tp, dtype=torch.long)

# 4. Forward: encoder tokens [B,T'*S',D] -> un-pooled grid [B,T',S',D] (the probe's
#    AttentivePooler needs the spatial tokens per temporal step) -> ClipProbe ->
#    class logits [B, num_classes].
with torch.no_grad():
    tok = encoder.backbone(clip).reshape(BATCH_SIZE, Tp, Sp, -1)
    logits = probe(tok, lens)

pred = logits.argmax(dim=-1)
print(f"tokens {tuple(tok.shape)}  logits {tuple(logits.shape)}")
for b in range(BATCH_SIZE):
    print(f"  clip {b}: predicted consonant = {class_names[pred[b].item()]}")
