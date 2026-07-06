"""Audio conditioning for AC-JEPA-audio (plans_aucjepa.md §3.4-3.5).

Three responsibilities:

1. **Temporal alignment / pooling** (`pool_audio_to_tokens`): the encoder's
   ``T'`` temporal tokens live at the tubelet rate -- token *j* covers output
   frames ``[clip_start + j*tubelet, + (j+1)*tubelet)``. The cached audio stream
   is ~50 Hz. We mean-pool the audio frames whose timestamps fall in each token's
   window -> one vector per temporal token, plus a validity mask for tail tokens
   with no audio. Alignment is in SECONDS via the exact same window edges that
   ``phonemes.token_center_times`` uses, so any video fps works.

2. **Normalization** (`normalize_audio`): per-dim z-score with corpus stats
   (from the offline cache's ``meta.json``) so the absolute (state) and delta
   (action) projection heads see comparable scales.

3. **State/action + the predictor adapter** (`to_state_action`,
   `AudioConditionedPredictor`): ``state = e[t]`` (absolute, "which tract shape")
   and ``action = e[t+1]-e[t]`` (delta, "how the tract moves"), each with its own
   trainable ``Linear(A, D)`` -- these live INSIDE the reused
   ``src.models.ac_predictor.VisionTransformerPredictorAC`` (its ``state_encoder``
   / ``action_encoder``), so the AC predictor is reused verbatim (``add_tokens=2``)
   and the projections serialize with the predictor.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. temporal alignment / pooling  (numpy; runs in the dataloader)
# --------------------------------------------------------------------------- #
def pool_audio_to_tokens(feats, audio_rate, n_tok, tubelet, target_fps,
                         clip_start_frame=0, pool="mean"):
    """Average-pool a ``[T_audio, A]`` audio stream onto ``n_tok`` temporal tokens.

    Token j spans output-frame window ``[clip_start + j*tubelet,
    + (j+1)*tubelet)`` -> seconds ``[f_lo, f_hi)/target_fps`` -> audio indices
    ``[f_lo/target_fps * audio_rate, f_hi/... * audio_rate)``. Returns
    ``(e[n_tok, A] float32, valid[n_tok] bool)``; empty-window tokens (clip tail
    past the audio end) are zero with ``valid=False``. A token whose window is
    empty but whose centre still lies inside the audio falls back to the nearest
    single frame (keeps interior tokens valid under rounding).
    """
    feats = np.asarray(feats, dtype=np.float32)
    T_audio, A = feats.shape
    e = np.zeros((n_tok, A), dtype=np.float32)
    valid = np.zeros(n_tok, dtype=bool)
    for j in range(n_tok):
        f_lo = clip_start_frame + j * tubelet
        f_hi = clip_start_frame + (j + 1) * tubelet
        i_lo = int(np.ceil(f_lo / target_fps * audio_rate - 1e-6))
        i_hi = int(np.ceil(f_hi / target_fps * audio_rate - 1e-6))   # exclusive
        a, b = max(0, i_lo), min(T_audio, i_hi)
        if b > a:
            e[j] = feats[a:b].mean(0)
            valid[j] = True
        else:                                                        # empty window
            c = int(round((f_lo + tubelet / 2.0) / target_fps * audio_rate))
            if 0 <= c < T_audio:
                e[j] = feats[c]
                valid[j] = True
    return e, valid


def normalize_audio(e, mean, std, eps=1e-6):
    """Per-dim z-score with corpus stats (``meta.json`` mean/std)."""
    return (e - np.asarray(mean, np.float32)) / (np.asarray(std, np.float32) + eps)


# --------------------------------------------------------------------------- #
# 2. state / action construction  (torch; batched in the train loop)
# --------------------------------------------------------------------------- #
def to_state_action(e, use_action=True):
    """``e[B, T', A]`` -> (state ``[B,T',A]`` absolute, action ``[B,T'-1,A]`` delta).

    Audio embeddings are (approximately) Euclidean, so the V-JEPA 2-AC manifold
    ``poses_to_diffs`` collapses to plain subtraction (no SO(3) correction).

    ``use_action=False`` -> **state-only** conditioning (no delta pathway). We shift
    the audio forward one token so that when the predictor emits frame *t* (from
    context position *t-1*) the last context slot carries the SYNCHRONOUS audio
    ``e[t]`` -- otherwise simply dropping the action would leave frame *t* seeing
    only ``e[t-1]`` (a 1-token / ~20 ms lag). The last slot is padded with ``e[-1]``.
    The action is zeroed rather than removed, so ``add_tokens`` stays 2 and the AC
    predictor + its checkpoint are reused verbatim (the action token degenerates to
    a constant bias). This is the fair state-vs-state+action ablation.
    """
    if not use_action:
        state = torch.cat([e[:, 1:], e[:, -1:]], dim=1)   # [B,T',A]  state[p]=e[p+1]
        action = torch.zeros_like(e[:, 1:])               # [B,T'-1,A] action off
        return state, action
    state = e
    action = e[:, 1:] - e[:, :-1]
    return state, action


# --------------------------------------------------------------------------- #
# 3. audio-conditioned predictor (wraps vit_ac_predictor verbatim)
# --------------------------------------------------------------------------- #
class AudioConditionedPredictor(nn.Module):
    """Thin adapter over ``VisionTransformerPredictorAC`` (parent repo, unmodified).

    The stock ``state_encoder`` / ``action_encoder`` (``Linear(A, D)``) ARE the
    trainable audio projections -- ``state=e[t]``, ``action=e[t+1]-e[t]`` share dim
    ``A``, so they drop in unchanged (``add_tokens=2``). Frame-causal attention +
    3-axis RoPE come for free. Owns ``forward_predictions`` (teacher-forced + AR
    rollout), mirroring ``app/vjepa_droid/train.py``.

    ``cond_mode`` selects HOW the audio reaches the predictor (all three reuse the
    same ``forward_predictions`` rollout + the same A->D state/action encoders):

    * ``concat`` (default) -- the stock path: per frame, prepend the (action,state)
      audio tokens to that frame's H*W visual tokens and let self-attention mix them
      (``add_tokens=2``, ``VisionTransformerPredictorAC.forward`` verbatim).
    * ``film`` -- per-block, per-channel Feature-wise Linear Modulation. The audio
      embedding emits ``(gamma, beta)`` per frame that scale/shift the visual tokens
      before each block (adaLN-zero style; zero-init -> starts at identity). No audio
      tokens are added, so the blocks run ``action_tokens=0``.
    * ``cross_attn`` -- each block is followed by a cross-attention sublayer where the
      visual tokens (queries) attend the per-frame audio embeddings (keys/values),
      frame-causal, gated by a zero-init scalar so audio starts as a no-op.

    ``film``/``cross_attn`` add fresh trainable params (the predictor is trained from
    scratch every run, so no checkpoint key compatibility is required); ``concat`` is
    untouched. See ``_forward_conditioned``.
    """

    def __init__(self, img_size, patch_size, num_frames, tubelet_size, embed_dim,
                 action_embed_dim, pred_embed_dim=384, depth=12, num_heads=12,
                 use_rope=True, frame_causal=True, use_activation_checkpointing=False,
                 use_extrinsics=False, spk_dim=None, normalize_reps=True,
                 cond_mode="concat", cross_attn_heads=None):
        super().__init__()
        from src.models.ac_predictor import vit_ac_predictor
        self.backbone = vit_ac_predictor(
            img_size=img_size, patch_size=patch_size, num_frames=num_frames,
            tubelet_size=tubelet_size, embed_dim=embed_dim,
            predictor_embed_dim=pred_embed_dim, action_embed_dim=action_embed_dim,
            depth=depth, num_heads=num_heads, is_frame_causal=frame_causal,
            use_rope=use_rope, use_sdpa=True, use_silu=False, wide_silu=True,
            use_extrinsics=use_extrinsics,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        # speaker -> extrinsics slot: the stock extrinsics_encoder is hardwired to
        # action_embed_dim-1; since we own the adapter, give it its own Linear.
        if use_extrinsics and spk_dim is not None:
            self.backbone.extrinsics_encoder = nn.Linear(spk_dim, pred_embed_dim)
        self.use_extrinsics = use_extrinsics
        gh = img_size // patch_size if isinstance(img_size, int) else img_size[0] // patch_size
        gw = img_size // patch_size if isinstance(img_size, int) else img_size[1] // patch_size
        self.tokens_per_frame = gh * gw
        self.normalize_reps = normalize_reps

        assert cond_mode in ("concat", "film", "cross_attn"), \
            f"cond_mode={cond_mode!r} not in (concat, film, cross_attn)"
        self.cond_mode = cond_mode
        if cond_mode != "concat":
            from src.models.utils.modules import build_action_block_causal_attention_mask
            D = pred_embed_dim
            # frame-causal self-attn mask over the H*W visual tokens per frame ONLY
            # (no prepended audio tokens -> add_tokens=0); sliced to N per call.
            Tmax = num_frames // tubelet_size
            self.register_buffer(
                "cond_attn_mask",
                build_action_block_causal_attention_mask(Tmax, gh, gw, add_tokens=0),
                persistent=False)
            if cond_mode == "film":
                # one per-block FiLM generator; zero-init -> gamma=beta=0 = identity,
                # audio effect is learned from scratch (adaLN-zero).
                self.film = nn.ModuleList(nn.Linear(D, 2 * D) for _ in range(depth))
                for lin in self.film:
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
            else:  # cross_attn
                nh = int(cross_attn_heads or num_heads)
                self.xattn = nn.ModuleList(
                    nn.MultiheadAttention(D, nh, batch_first=True) for _ in range(depth))
                self.xattn_norm = nn.ModuleList(nn.LayerNorm(D) for _ in range(depth))
                # zero-init gate -> cross-attn contributes nothing at start, learned on.
                self.xattn_gate = nn.ParameterList(
                    nn.Parameter(torch.zeros(1)) for _ in range(depth))

    def forward(self, x, actions, states, extrinsics=None):
        return self._predict(x, actions, states, extrinsics)

    def _predict(self, z, a, s, e=None):
        """Dispatch one predictor call on ``cond_mode`` (concat = stock backbone)."""
        if self.cond_mode == "concat":
            return self.backbone(z, a, s, e)
        return self._forward_conditioned(z, a, s, e)

    def _xattn_block_mask(self, N, hw, T, device):
        """Bool ``[N, T]`` for ``nn.MultiheadAttention`` (True = FORBIDDEN): a query
        token at frame ``t = n // hw`` may attend audio frames ``0..t`` (frame-causal,
        mirrors the self-attention mask). Row ``t`` always keeps its own frame, so no
        query is fully masked (no softmax NaN)."""
        q_frame = torch.arange(N, device=device) // hw            # [N]
        a_frame = torch.arange(T, device=device)                  # [T]
        return a_frame[None, :] > q_frame[:, None]                # [N, T]

    def _forward_conditioned(self, x, actions, states, extrinsics=None):
        """FiLM / cross-attention audio conditioning (no per-frame audio tokens).

        Mirrors ``VisionTransformerPredictorAC.forward`` but injects the WavLM audio
        via feature-wise modulation (``film``) or cross-attention (``cross_attn``)
        instead of concatenating action/state tokens. Self-attention runs with
        ``action_tokens=0`` (plain frame-causal spatial-RoPE over the H*W visual
        tokens); the A->D ``state_encoder``/``action_encoder`` are reused to build the
        per-frame audio embedding. Both injection paths are zero-initialised, so
        training starts as a pure visual predictor and must LEARN to use the audio.
        """
        bb = self.backbone
        x = bb.predictor_embed(x)                                 # [B, N, D]
        B, N, D = x.size()
        H, W = bb.grid_height, bb.grid_width
        hw = H * W
        T = N // hw
        audio_emb = bb.state_encoder(states) + bb.action_encoder(actions)   # [B, T, D]
        attn_mask = self.cond_attn_mask[:N, :N].to(x.device, non_blocking=True)
        xmask = (self._xattn_block_mask(N, hw, T, x.device)
                 if self.cond_mode == "cross_attn" else None)
        for i, blk in enumerate(bb.predictor_blocks):
            if self.cond_mode == "film":
                g, b = self.film[i](audio_emb).chunk(2, dim=-1)   # [B,T,D] each
                g = g.repeat_interleave(hw, dim=1)                # [B,N,D]
                b = b.repeat_interleave(hw, dim=1)
                x = (1.0 + g) * x + b
            x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=0)
            if self.cond_mode == "cross_attn":
                q = self.xattn_norm[i](x)
                ctx, _ = self.xattn[i](q, audio_emb, audio_emb, attn_mask=xmask,
                                       need_weights=False)
                x = x + self.xattn_gate[i] * ctx
        x = bb.predictor_norm(x)
        x = bb.predictor_proj(x)
        return x

    def _step(self, z, a, s, e=None):
        z = self._predict(z, a, s, e)
        if self.normalize_reps:
            z = F.layer_norm(z, (z.size(-1),))
        return z

    def _ar_rollout(self, h, states, actions, start, n_steps, extrinsics=None):
        """Autoregressive rollout: seed with ``start`` REAL frames (0..start-1),
        then predict ``n_steps`` further frames using only audio + own predictions.

        Predicting frame n needs context frames 0..n-1 with ``states[:, :n]`` /
        ``actions[:, :n]`` (the AC predictor outputs the next frame per input
        frame; we keep the last). Returns the predicted frames
        ``[B, n_steps*HW, D]`` (frames ``start..start+n_steps-1``).
        """
        hw = self.tokens_per_frame
        _z = h[:, : start * hw]                       # real context prefix
        for n in range(start, start + n_steps):
            _a, _s = actions[:, :n], states[:, :n]
            _e = extrinsics[:, :n] if (extrinsics is not None) else None
            _z_nxt = self._step(_z, _a, _s, _e)[:, -hw:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        return _z[:, start * hw:]

    def forward_predictions(self, h, states, actions, auto_steps=2,
                            ctx_frames=None, extrinsics=None):
        """World-model rollout. Returns (z_tf, z_ar) + the AR start frame.

        Args:
          h:       [B, T'*HW, D] frozen target tokens (already layer-normed).
          states:  [B, T',  A]   absolute audio per frame.
          actions: [B, T'-1, A]  audio delta per consecutive frame.
          auto_steps: droid-style AR depth from frame 0 (used iff ctx_frames None).
          ctx_frames: if set, the AR branch is the plan's §4 context-prefix rollout
            -- seed with ``ctx_frames`` real frames and predict ALL remaining
            frames from audio only (makes audio causally necessary).
        z_tf = teacher-forced next-frame over the whole clip (predicts frames
        1..T'-1). z_ar = the AR branch (predicts frames ``ar_start``..T'-1 when
        ctx_frames is set, else frames 1..auto_steps).
        """
        hw = self.tokens_per_frame
        Tp = h.size(1) // hw
        ext_tf = extrinsics[:, :-1] if (extrinsics is not None) else None
        # -- teacher forced: context = frames 0..T'-2, predict frames 1..T'-1
        z_tf = self._step(h[:, :-hw], actions, states[:, :-1], ext_tf)
        # -- autoregressive branch
        if ctx_frames is not None:
            ar_start = int(ctx_frames)
            z_ar = self._ar_rollout(h, states, actions, ar_start, Tp - ar_start, extrinsics)
        else:
            ar_start = 1
            z_ar = self._ar_rollout(h, states, actions, ar_start, max(1, auto_steps), extrinsics)
        return z_tf, z_ar, ar_start


# --------------------------------------------------------------------------- #
# loss (masked L1 in layer-normed feature space, plan §3.6)
# --------------------------------------------------------------------------- #
def rollout_l1(z, h, hw, frame_valid=None, loss_exp=1.0, start_frame=1):
    """L1 between predicted tokens ``z`` and the matching future target frames.

    ``z`` predicts frames ``start_frame..start_frame+n_pred-1``; the targets are
    ``h[:, start_frame*hw : start_frame*hw + z.size(1)]``. If ``frame_valid[B,T']``
    is given, frames whose target audio is invalid (clip tail) are dropped.
    """
    n_pred = z.size(1) // hw
    s = start_frame * hw
    h_t = h[:, s : s + z.size(1)]
    diff = torch.abs(z - h_t) ** loss_exp / loss_exp           # [B, n_pred*hw, D]
    if frame_valid is None:
        return diff.mean()
    fv = frame_valid[:, start_frame : start_frame + n_pred].float()   # [B, n_pred]
    w = fv.repeat_interleave(hw, dim=1).unsqueeze(-1)          # [B, n_pred*hw, 1]
    denom = w.sum() * z.size(-1)
    return (diff * w).sum() / denom.clamp_min(1.0)
