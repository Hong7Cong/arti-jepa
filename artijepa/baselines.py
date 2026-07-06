"""Frozen image-encoder baselines for the phoneme-prediction eval (Plan Part C).

These are pretrained **2-D image** encoders (OpenAI CLIP, SigLIP, DINOv2,
supervised ViT-L, ResNet-50) used to answer: is the V-JEPA video-SSL gain
*JEPA-specific*, or would any pretrained vision encoder give similarly
phoneme-decodable features? They are applied **per frame**, then consecutive
frame-pairs are averaged so the output sits on V-JEPA's temporal token grid
(tubelet 2). That makes them drop-in for the *same* per-temporal-token probe and
the *same* phoneme labels -- only the encoder changes.

Input contract (matches `usc_lss` / `rtmri_dataset` under `intensity_norm:minmax`
with no `grayscale_stats`): clip `[B, 3, T, H, W]` in `[0, 1]`, grayscale
replicated x3. The adapter applies the model's *own* ImageNet/web mean+std and
(if needed) resizes to the model's native input size -- "each baseline its best
shot". It returns `[B, T', D]` with `T' = T // tubelet`, so in `eval_phoneme.py`
the spatial-pool reshape is a no-op (`N == T'`).

All five come from **timm** (this env's `transformers` 5.x lazy model-class
imports are broken under torch 2.6 -- same failure as the audio CTC path -- so we
deliberately avoid it). timm 1.0.27 has no DINOv3 ViT-L (only ViT-7B / ConvNeXt),
so the DINO baseline is DINOv2 ViT-L/14.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# short alias -> timm model name (.pretrained tag). Verified present in timm 1.0.27.
BASELINES = {
    "clip":   "vit_large_patch14_clip_224.openai",          # OpenAI CLIP-L/14   D=1024 @224
    "siglip": "vit_large_patch16_siglip_256.webli",         # SigLIP-L/16        D=1024 @256
    "dinov2": "vit_large_patch14_dinov2.lvd142m",           # DINOv2 ViT-L/14    D=1024 @518
    "dinov3": "vit_large_patch16_dinov3.lvd1689m",          # DINOv3 ViT-L/16    D=1024 (LVD-1689M)
    "vitl":   "vit_large_patch16_224.augreg_in21k_ft_in1k",  # Google supervised ViT-L/16 (AugReg IN21k->IN1k) D=1024 @224
    "resnet": "resnet50.a1_in1k",                           # ResNet-50 (IN1k)   D=2048 @224
}


class ImageBaselineBackbone(nn.Module):
    """timm image encoder -> per temporal-token features.

    `forward(clip)` takes `clip [B,3,T,H,W]` in `[0,1]` and returns either
      * `pool_spatial=True`  (default heads): the model's native POOLED embedding,
        tubelet-pooled over time -> `[B, T', D]` (S'=1; the spatial-pool reshape in
        `eval_phoneme.extract` is then a no-op). This is the original baseline.
      * `pool_spatial=False` (spatial-aware heads tcn_spatial/attentive): the per-frame
        PATCH-token grid, tubelet-pooled over time and flattened TEMPORAL-MAJOR to
        `[B, T'*S', D]`, so `extract`'s `reshape(B,T',N//T',D)` recovers `[B,T',S',D]`
        -- the fair-fight analogue of V-JEPA's per-frame spatial tokens. The grid side
        is capped at `grid_cap` via adaptive-avg-pool so DINOv2's 37x37@518px stays
        tractable (CLIP/SigLIP 16x16, ViT-L/16 14x14, ResNet 7x7 are <=cap -> untouched).

    Frames run in sub-batches of `frame_batch` to bound GPU memory (DINOv2's native
    518 px is heavy).
    """

    def __init__(self, model_name, tubelet_size=2, frame_batch=64,
                 pool_spatial=True, grid_cap=16):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        self.name = BASELINES.get(model_name, model_name)
        # num_classes=0 -> the model's pooled image embedding [B, D] (CLS / GAP /
        # attn-pool, whatever the model uses), uniform across ViT and ResNet.
        self.model = timm.create_model(self.name, pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        dc = resolve_data_config({}, model=self.model)
        self.input_size = int(dc["input_size"][-1])
        self.interp = dc.get("interpolation", "bicubic")
        self.register_buffer("mean", torch.tensor(dc["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(dc["std"]).view(1, 3, 1, 1))
        self.tubelet = int(tubelet_size)
        self.frame_batch = int(frame_batch)
        self.pool_spatial = bool(pool_spatial)
        self.grid_cap = int(grid_cap)
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 0))

    @property
    def embed_dim(self):
        return int(self.model.num_features)

    def _frame_grid(self, x):
        """[b,3,H,W] -> per-frame patch-token grid [b, S', D] (S' capped at grid_cap^2)."""
        feat = self.model.forward_features(x)
        if feat.ndim == 4:                       # CNN feature map [b, D, h, w]
            b, D, h, w = feat.shape
        else:                                    # ViT tokens [b, prefix+S', D]
            feat = feat[:, self.num_prefix:, :]  # drop CLS/reg tokens -> [b, S', D]
            b, S, D = feat.shape
            h = w = int(round(S ** 0.5))
            feat = feat.transpose(1, 2).reshape(b, D, h, w)
        if max(h, w) > self.grid_cap:            # downsample large grids (dinov2@518)
            feat = F.adaptive_avg_pool2d(feat, (self.grid_cap, self.grid_cap))
        return feat.flatten(2).transpose(1, 2)   # [b, S', D]

    @torch.no_grad()
    def forward(self, clip):
        B, C, T, H, W = clip.shape
        assert C == 3, f"expected 3ch clip, got {C}"
        x = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T,3,H,W]
        if H != self.input_size or W != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bicubic", align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        Tp = T // self.tubelet
        if self.pool_spatial:                                        # native pooled emb
            embs = [self.model(x[i:i + self.frame_batch])
                    for i in range(0, x.shape[0], self.frame_batch)]
            emb = torch.cat(embs, 0)                                 # [B*T, D]
            emb = emb.reshape(B, T, emb.shape[-1])
            # average tubelet-size consecutive frames -> V-JEPA temporal token grid
            return emb.reshape(B, Tp, self.tubelet, -1).mean(2)      # [B, T', D]
        grids = [self._frame_grid(x[i:i + self.frame_batch])
                 for i in range(0, x.shape[0], self.frame_batch)]
        g = torch.cat(grids, 0)                                      # [B*T, S', D]
        S, D = g.shape[1], g.shape[2]
        g = g.reshape(B, T, S, D).reshape(B, Tp, self.tubelet, S, D).mean(2)  # [B,T',S',D]
        return g.reshape(B, Tp * S, D)             # temporal-major [B, T'*S', D]


class BaselineEncoder(nn.Module):
    """Wrapper exposing `.backbone(clip)` so it is a drop-in for the V-JEPA encoder
    object in `eval_phoneme.extract`. Returns `[B,T',D]` (pool_spatial=True; the
    extract spatial-pool reshape collapses to identity) or the temporal-major
    `[B,T'*S',D]` grid (pool_spatial=False) for the spatial-aware probes."""

    def __init__(self, model_name, tubelet_size=2, frame_batch=64,
                 pool_spatial=True, grid_cap=16):
        super().__init__()
        self.backbone = ImageBaselineBackbone(
            model_name, tubelet_size=tubelet_size, frame_batch=frame_batch,
            pool_spatial=pool_spatial, grid_cap=grid_cap)
