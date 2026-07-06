"""VideoMAE ViT-L baseline for the disfluency / segment-classification eval.

A **video** counterpart to the per-frame image baselines in `baselines.py`. Where
those apply a 2-D encoder frame-by-frame, VideoMAE ingests the whole clip with a
3-D (tubelet) patch embedding, so it is the natural "generic video-SSL" competitor
to Arti-JEPA's video-JEPA features. Two use modes:

  * **frozen**  (`VideoMAEEncoder`): drop-in for the V-JEPA encoder object in the
    extractor -- `.backbone(clip)` returns temporal-major tokens `[B, L, D]`
    (`pool_spatial=True` -> `[B, T', D]`; `False` -> `[B, T'*S', D]`), so the same
    attentive segment probe trains on cached features.
  * **finetune** (`VideoMAEClassifier`): the encoder is trainable and a small
    attentive/mean head is learned end-to-end on the clips.

Loaded from HuggingFace `transformers` (5.x works here for VideoMAE; only the
lazy audio model-classes were broken). Default `MCG-NJU/videomae-large` (Kinetics
SSL-pretrained, D=1024, 16 frames, tubelet 2, patch 16 -> 8x14x14 tokens).
Input contract matches the image baselines: clip `[B, 3, T, H, W]` in `[0, 1]`
(minmax, grayscale x3); this adapter resizes to 224, samples the model's frame
count, and applies ImageNet mean/std.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_VIDEOMAE = "MCG-NJU/videomae-large"      # ViT-L, D=1024, 16f
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _select_frames(x, n_frames):
    """[B,T,C,H,W] -> [B,n_frames,C,H,W] by evenly spaced nearest-frame indexing."""
    T = x.shape[1]
    if T == n_frames:
        return x
    idx = torch.linspace(0, T - 1, n_frames, device=x.device).round().long()
    return x.index_select(1, idx)


class VideoMAEBackbone(nn.Module):
    """Frozen VideoMAE encoder -> per temporal-token features.

    `forward(clip [B,3,T,H,W] in [0,1])` returns, mirroring `ImageBaselineBackbone`:
      * `pool_spatial=True`  -> mean over spatial tokens: `[B, T', D]`.
      * `pool_spatial=False` -> temporal-major grid `[B, T'*S', D]` (S' capped at
        `grid_cap^2` via adaptive-avg-pool) for the spatial/attentive probe.
    VideoMAE has no CLS/prefix token, so the `T'*S'` tokens map cleanly to the
    `[B, T', S', D]` grid the extractor expects.
    """

    def __init__(self, model_name=DEFAULT_VIDEOMAE, frame_batch=8,
                 pool_spatial=True, grid_cap=16):
        super().__init__()
        from transformers import VideoMAEModel
        self.name = model_name
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        c = self.model.config
        self.num_frames = int(c.num_frames)
        self.tubelet = int(c.tubelet_size)
        self.input_size = int(c.image_size)
        self.patch = int(c.patch_size)
        self.grid = self.input_size // self.patch            # spatial grid side (e.g. 14)
        self.frame_batch = int(frame_batch)
        self.pool_spatial = bool(pool_spatial)
        self.grid_cap = int(grid_cap)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1))

    @property
    def embed_dim(self):
        return int(self.model.config.hidden_size)

    def _preprocess(self, clip):
        """[B,3,T,H,W] in [0,1] -> normalized [B, n_frames, 3, 224, 224]."""
        B, C, T, H, W = clip.shape
        assert C == 3, f"expected 3ch clip, got {C}"
        clip = (clip - self.mean.to(clip.dtype)) / self.std.to(clip.dtype)
        x = clip.permute(0, 2, 1, 3, 4)                      # [B,T,3,H,W]
        x = _select_frames(x, self.num_frames)
        if H != self.input_size or W != self.input_size:
            x = x.reshape(-1, C, H, W)
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bicubic", align_corners=False)
            x = x.reshape(B, self.num_frames, C, self.input_size, self.input_size)
        return x

    def _tokens(self, x):
        """[B,n_frames,3,S,S] -> grid tokens [B, T', S', D]."""
        h = self.model(pixel_values=x).last_hidden_state     # [B, T'*grid^2, D]
        Bx, L, D = h.shape
        Tp = self.num_frames // self.tubelet
        S = L // Tp                                          # grid^2
        h = h.reshape(Bx, Tp, S, D)
        side = int(round(S ** 0.5))
        if side > self.grid_cap:                             # downsample the spatial grid
            g = h.reshape(Bx * Tp, side, side, D).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(g, (self.grid_cap, self.grid_cap))
            h = g.flatten(2).transpose(1, 2).reshape(Bx, Tp, self.grid_cap ** 2, D)
        return h                                             # [B, T', S', D]

    @torch.no_grad()
    def forward(self, clip):
        x = self._preprocess(clip)
        outs = [self._tokens(x[i:i + self.frame_batch])
                for i in range(0, x.shape[0], self.frame_batch)]
        h = torch.cat(outs, 0)                               # [B, T', S', D]
        B, Tp, S, D = h.shape
        if self.pool_spatial:
            return h.mean(2)                                 # [B, T', D]
        return h.reshape(B, Tp * S, D)                       # temporal-major [B, T'*S', D]


class VideoMAEEncoder(nn.Module):
    """Wrapper exposing `.backbone(clip)` so it is a drop-in for the V-JEPA encoder
    object in the extractor (same role as `baselines.BaselineEncoder`)."""

    def __init__(self, model_name=DEFAULT_VIDEOMAE, frame_batch=8,
                 pool_spatial=True, grid_cap=16):
        super().__init__()
        self.backbone = VideoMAEBackbone(model_name, frame_batch=frame_batch,
                                         pool_spatial=pool_spatial, grid_cap=grid_cap)


class VideoMAEClassifier(nn.Module):
    """End-to-end fine-tunable VideoMAE segment classifier.

    The VideoMAE encoder is trainable; tokens are pooled to a clip vector by an
    attentive pooler (`pool='attentive'`, mirrors the frozen probe) or mean
    (`pool='mean'`) and a linear head predicts the disfluency class. `freeze_encoder`
    turns it into a linear/attentive probe on the raw (un-cached) features.
    """

    def __init__(self, num_classes, model_name=DEFAULT_VIDEOMAE, pool="attentive",
                 heads=8, freeze_encoder=False, dropout=0.0):
        super().__init__()
        self.backbone = VideoMAEBackbone(model_name, pool_spatial=(pool == "mean"),
                                         grid_cap=16)
        # re-enable grads on the encoder for fine-tuning (backbone froze them)
        self.freeze_encoder = bool(freeze_encoder)
        for p in self.backbone.model.parameters():
            p.requires_grad = not self.freeze_encoder
        dim = self.backbone.embed_dim
        self.pool = pool
        if pool == "attentive":
            from src.models.attentive_pooler import AttentivePooler
            self.pooler = AttentivePooler(num_queries=1, embed_dim=dim,
                                          num_heads=heads, mlp_ratio=4.0, depth=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def _encode(self, clip):
        x = self.backbone._preprocess(clip)
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            h = self.backbone._tokens(x)                     # [B,T',S',D]
        B, Tp, S, D = h.shape
        if self.pool == "mean":
            return h.reshape(B, Tp * S, D).mean(1)           # [B,D]
        q = self.pooler(h.reshape(B, Tp * S, D)).squeeze(1)  # [B,D]
        return q

    def forward(self, clip):
        return self.head(self.drop(self._encode(clip)))      # [B, num_classes]
