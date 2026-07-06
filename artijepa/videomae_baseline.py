"""Frozen VideoMAE video-model baseline for the phoneme-prediction eval.

The generic **video-SSL** competitor to Arti-JEPA's video-JEPA: unlike the per-frame
image baselines in `baselines.py`, VideoMAE is a genuine **3-D tubelet** encoder, so
it answers "is the gain video-JEPA-specific, or would any pretrained video encoder
give phoneme-decodable features?". Default `MCG-NJU/videomae-large` (ViT-L, D=1024,
16 frames / tubelet 2 / patch 16 -> 8x14x14 = 1568 tokens).

Input/output contract mirrors `baselines.ImageBaselineBackbone` so this is a drop-in
for `eval_phoneme.extract`'s `encoder.backbone(clip)` call:
  clip `[B, 3, T, H, W]` in `[0, 1]` (grayscale x3, minmax; no rtMRI channel-norm)
    -> pool_spatial=True : `[B, T', D]`         (mean over the S'=196 spatial tokens)
    -> pool_spatial=False: `[B, T'*S', D]` temporal-major, so `extract`'s
       `reshape(B, T', N//T', D)` recovers `[B, T', S', D]` for the spatial probes.
Native geometry (224 px, **exactly 16 frames**, tubelet 2) is VideoMAE's "best shot"
and is enforced by `load_frozen_encoder` (it overrides the config's spatial_size /
frames_per_clip / tubelet_size).

transformers 5.x references a torch>=2.7 fp8 dtype (`float8_e8m0fnu`) that is absent
in this env's torch 2.6, which breaks the whole modeling import (same class of
failure that made the image baselines avoid `transformers`). We alias the missing
dtype BEFORE importing -- inference never touches the fp8-quantization path -- and
import the model class directly from its submodule to bypass the broken lazy loader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- shim: transformers 5.x wants torch>=2.7 fp8 dtypes absent in torch 2.6 --------
for _n in ("float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if not hasattr(torch, _n):
        setattr(torch, _n, torch.float8_e5m2)   # placeholder; fp8-quant path unused
from transformers.models.videomae.modeling_videomae import VideoMAEModel  # noqa: E402

VIDEOMAE_LARGE = "MCG-NJU/videomae-large"
# VideoMAEImageProcessor defaults = ImageNet normalization.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _restore_attn_biases(model, model_name):
    """transformers 5.x refactored VideoMAE self-attention to standard
    `query/key/value.bias`, but the MCG-NJU checkpoint stores the ORIGINAL VideoMAE
    scheme: learned `q_bias` + `v_bias`, with the key bias fixed at 0 (BEiT-style).
    `from_pretrained` therefore leaves all three biases zero-initialised (`q_bias`
    norm ~18 is silently dropped) -> the encoder is NOT the true pretrained model.
    Restore the trained q/v biases from the raw checkpoint; keep key bias = 0."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    raw = load_file(hf_hub_download(model_name, "model.safetensors"))
    pfx = "videomae.encoder.layer"
    n = 0
    for i, layer in enumerate(model.encoder.layer):
        qb = raw.get(f"{pfx}.{i}.attention.attention.q_bias")
        vb = raw.get(f"{pfx}.{i}.attention.attention.v_bias")
        if qb is None or vb is None:
            continue
        attn = layer.attention.attention
        with torch.no_grad():
            attn.query.bias.copy_(qb)
            attn.value.bias.copy_(vb)
            attn.key.bias.zero_()
        n += 1
    if n != len(model.encoder.layer):
        raise RuntimeError(f"VideoMAE bias restore: fixed {n}/{len(model.encoder.layer)} "
                           "layers -- checkpoint key scheme changed, refusing to ship a "
                           "half-loaded encoder")
    return n


class VideoMAEBackbone(nn.Module):
    """HF VideoMAEModel -> per temporal-token features on V-JEPA's token grid."""

    def __init__(self, model_name=VIDEOMAE_LARGE, pool_spatial=True, grid_cap=16, **_):
        super().__init__()
        self.name = model_name
        self.model = VideoMAEModel.from_pretrained(model_name)
        _restore_attn_biases(self.model, model_name)   # fix q/v biases dropped by from_pretrained
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        c = self.model.config
        self.input_size = int(c.image_size)          # 224
        self.num_frames = int(c.num_frames)          # 16 (fixed by pos-embeddings)
        self.tubelet = int(c.tubelet_size)           # 2
        self.patch = int(c.patch_size)               # 16
        self._embed_dim = int(c.hidden_size)         # 1024
        self.gh = self.input_size // self.patch      # 14 (grid side)
        self.pool_spatial = bool(pool_spatial)
        self.grid_cap = int(grid_cap)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    @property
    def embed_dim(self):
        return self._embed_dim

    @torch.no_grad()
    def forward(self, clip):
        B, C, T, H, W = clip.shape
        assert C == 3, f"expected 3ch clip, got {C}"
        assert T == self.num_frames, \
            f"VideoMAE needs exactly {self.num_frames} frames, got {T}"
        # per-frame resize to native input + ImageNet normalization
        x = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T,3,H,W]
        if H != self.input_size or W != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bicubic", align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        x = x.reshape(B, T, C, self.input_size, self.input_size)  # VideoMAE wants [B,T,C,H,W]
        tok = self.model(x).last_hidden_state                     # [B, T'*gh*gw, D] temporal-major
        Tp = T // self.tubelet                                    # 8 temporal tokens
        D = tok.shape[-1]
        S = tok.shape[1] // Tp                                     # gh*gw = 196
        if self.pool_spatial:                                     # native pooled -> [B,T',D]
            return tok.reshape(B, Tp, S, D).mean(2)
        if self.gh > self.grid_cap:                               # cap grid (14<=16 -> untouched)
            g = tok.reshape(B * Tp, self.gh, self.gh, D).permute(0, 3, 1, 2)  # [B*Tp,D,gh,gh]
            g = F.adaptive_avg_pool2d(g, (self.grid_cap, self.grid_cap))
            return g.flatten(2).transpose(1, 2).reshape(B, Tp * self.grid_cap ** 2, D)
        return tok                                                # [B, T'*S', D] temporal-major


class VideoMAEEncoder(nn.Module):
    """Wrapper exposing `.backbone(clip)` -- drop-in for the V-JEPA encoder object in
    `eval_phoneme.extract`."""

    def __init__(self, model_name=VIDEOMAE_LARGE, pool_spatial=True, grid_cap=16, **_):
        super().__init__()
        self.backbone = VideoMAEBackbone(model_name, pool_spatial=pool_spatial,
                                         grid_cap=grid_cap)
