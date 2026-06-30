"""Articulator conditioning for AC-JEPA (aucjepa_plans_new.md §1.2-§2a).

The *articulator-conditioned* analogue of the (now trashed) acoustic ``audio_cond``:
the conditioning signal is the **6-D articulator vector** ``arti-6`` (constriction
degree at six places of articulation) instead of a 768-D WavLM embedding. Everything
else is the V-JEPA 2-AC recipe, reused verbatim:

  * ``state  = arti[t]``           absolute articulator config  (6-D)
  * ``action = arti[t+1]-arti[t]`` articulator delta = the "gesture" (6-D)

Both feed the stock ``VisionTransformerPredictorAC`` ``state_encoder`` /
``action_encoder`` (``Linear(6, D)``) so the AC predictor is reused unmodified
(``add_tokens=2``); frame-causal attention + 3-axis RoPE come for free.

Responsibilities (mirror the trashed audio module, A=6):
  1. ``pool_arti_to_tokens`` -- align a native-rate ``[T_arti, 6]`` stream onto the
     encoder's ``T'`` temporal tokens (seconds-based windows; any fps).
  2. ``normalize_arti`` -- per-dim z-score with corpus stats (cache ``meta.json``).
  3. ``to_state_action`` / ``ArtiConditionedPredictor`` -- build state/action and
     run the teacher-forced + autoregressive world-model rollout.

The forward model trained here is the frozen ``P`` the planner (``acjepa_plan``)
rolls out at test time.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical 6-D articulator order (the .mat field order in usc_lss/articulators).
# FREEZE this order -- the cache, the predictor's 6-D state/action, the phoneme
# articulatory-target table (Energy 3) and the planner must all agree on it.
ARTICULATORS = ["Bilabial", "Alveolar", "Palatal", "Velum", "Pharyngeal", "Larynx"]
ARTI_DIM = len(ARTICULATORS)


# --------------------------------------------------------------------------- #
# 1. temporal alignment / pooling  (numpy; runs in the dataloader)
# --------------------------------------------------------------------------- #
def pool_arti_to_tokens(feats, arti_rate, n_tok, tubelet, target_fps,
                        clip_start_frame=0, pool="mean"):
    """Average-pool an ``[T_arti, 6]`` articulator stream onto ``n_tok`` temporal tokens.

    Token j spans output-frame window ``[clip_start + j*tubelet, +(j+1)*tubelet)``
    -> seconds ``[f_lo, f_hi)/target_fps`` -> arti indices ``[.. * arti_rate)``.
    Returns ``(e[n_tok, 6] float32, valid[n_tok] bool)``; empty-window tokens (clip
    tail past the arti end) are zero with ``valid=False``. A token whose window is
    empty but whose centre still lies inside the stream falls back to the nearest
    single frame (keeps interior tokens valid under rounding). Identical window
    math to ``phonemes.token_center_times`` so alignment matches the phoneme eval.
    """
    feats = np.asarray(feats, dtype=np.float32)
    T_arti, A = feats.shape
    e = np.zeros((n_tok, A), dtype=np.float32)
    valid = np.zeros(n_tok, dtype=bool)
    for j in range(n_tok):
        f_lo = clip_start_frame + j * tubelet
        f_hi = clip_start_frame + (j + 1) * tubelet
        i_lo = int(np.ceil(f_lo / target_fps * arti_rate - 1e-6))
        i_hi = int(np.ceil(f_hi / target_fps * arti_rate - 1e-6))   # exclusive
        a, b = max(0, i_lo), min(T_arti, i_hi)
        if b > a:
            e[j] = feats[a:b].mean(0)
            valid[j] = True
        else:                                                       # empty window
            c = int(round((f_lo + tubelet / 2.0) / target_fps * arti_rate))
            if 0 <= c < T_arti:
                e[j] = feats[c]
                valid[j] = True
    return e, valid


def normalize_arti(e, mean, std, eps=1e-6):
    """Per-dim z-score with corpus stats (``meta.json`` mean/std)."""
    return (e - np.asarray(mean, np.float32)) / (np.asarray(std, np.float32) + eps)


# --------------------------------------------------------------------------- #
# 2. state / action construction  (torch; batched in the train loop & planner)
# --------------------------------------------------------------------------- #
def to_state_action(e):
    """``e[B, T', 6]`` -> (state ``[B,T',6]`` absolute, action ``[B,T'-1,6]`` delta).

    Articulator coordinates are Euclidean, so the V-JEPA 2-AC manifold
    ``poses_to_diffs`` collapses to plain subtraction (no SO(3) correction).
    """
    state = e
    action = e[:, 1:] - e[:, :-1]
    return state, action


def state_from_actions(s_k, actions):
    """Absolute states from a seed state ``s_k[B,6]`` + an action sequence
    ``actions[B,T-1,6]`` -> states ``[B,T,6]`` (cumulative sum). Used by the
    planner: CEM searches over actions, the predictor needs the absolute states.
    """
    s0 = s_k.unsqueeze(1)                                   # [B,1,6]
    states = torch.cat([s0, s0 + torch.cumsum(actions, dim=1)], dim=1)
    return states


# --------------------------------------------------------------------------- #
# 3. articulator-conditioned predictor (wraps vit_ac_predictor verbatim, A=6)
# --------------------------------------------------------------------------- #
class ArtiConditionedPredictor(nn.Module):
    """Thin adapter over ``VisionTransformerPredictorAC`` (parent repo, unmodified).

    The stock ``state_encoder`` / ``action_encoder`` (``Linear(6, D)``) ARE the
    trainable articulator projections -- ``state=arti[t]``, ``action=arti[t+1]-arti[t]``
    share dim ``A=6`` so they drop in unchanged (``add_tokens=2``). Frame-causal
    attention + 3-axis RoPE come for free. Owns ``forward_predictions``
    (teacher-forced + AR rollout) and the planner-facing ``rollout``.
    """

    def __init__(self, img_size, patch_size, num_frames, tubelet_size, embed_dim,
                 action_embed_dim=ARTI_DIM, pred_embed_dim=384, depth=12, num_heads=12,
                 use_rope=True, frame_causal=True, use_activation_checkpointing=False,
                 use_extrinsics=False, spk_dim=None, normalize_reps=True):
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

    def forward(self, x, actions, states, extrinsics=None):
        return self.backbone(x, actions, states, extrinsics)

    def _step(self, z, a, s, e=None):
        z = self.backbone(z, a, s, e)
        if self.normalize_reps:
            z = F.layer_norm(z, (z.size(-1),))
        return z

    def _ar_rollout(self, h, states, actions, start, n_steps, extrinsics=None):
        """Autoregressive rollout: seed with ``start`` REAL frames (0..start-1),
        then predict ``n_steps`` further frames using only arti + own predictions.

        Predicting frame n needs context frames 0..n-1 with ``states[:, :n]`` /
        ``actions[:, :n]`` (the AC predictor outputs the next frame per input frame;
        keep the last). Returns the predicted frames ``[B, n_steps*HW, D]`` (frames
        ``start..start+n_steps-1``). This is the routine the planner reuses to score
        a candidate action sequence entirely in latent space (no pixel decode).
        """
        hw = self.tokens_per_frame
        _z = h[:, : start * hw]                            # real context prefix
        for n in range(start, start + n_steps):
            _a, _s = actions[:, :n], states[:, :n]
            _e = extrinsics[:, :n] if (extrinsics is not None) else None
            _z_nxt = self._step(_z, _a, _s, _e)[:, -hw:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        return _z[:, start * hw:]

    def forward_predictions(self, h, states, actions, auto_steps=2,
                            ctx_frames=None, extrinsics=None):
        """World-model rollout. Returns (z_tf, z_ar, ar_start).

        Args:
          h:       [B, T'*HW, D] frozen target tokens (already layer-normed).
          states:  [B, T',  6]   absolute arti per frame.
          actions: [B, T'-1, 6]  arti delta per consecutive frame.
          auto_steps: droid-style AR depth from frame 0 (used iff ctx_frames None).
          ctx_frames: if set, the AR branch is the plan §4 context-prefix rollout --
            seed ``ctx_frames`` real frames and predict ALL remaining frames from
            arti only (makes the articulators causally necessary).
        z_tf = teacher-forced next-frame over the whole clip (frames 1..T'-1).
        z_ar = the AR branch (frames ``ar_start``..T'-1 when ctx_frames is set, else
        frames 1..auto_steps).
        """
        hw = self.tokens_per_frame
        Tp = h.size(1) // hw
        ext_tf = extrinsics[:, :-1] if (extrinsics is not None) else None
        z_tf = self._step(h[:, :-hw], actions, states[:, :-1], ext_tf)
        if ctx_frames is not None:
            ar_start = int(ctx_frames)
            z_ar = self._ar_rollout(h, states, actions, ar_start, Tp - ar_start, extrinsics)
        else:
            ar_start = 1
            z_ar = self._ar_rollout(h, states, actions, ar_start, max(1, auto_steps), extrinsics)
        return z_tf, z_ar, ar_start

    @torch.no_grad()
    def rollout(self, z_seed, states, actions, ctx_frames, extrinsics=None):
        """Planner-facing pure rollout: seed ``ctx_frames`` real latent frames in
        ``z_seed[B, ctx_frames*HW, D]`` and roll out the rest of the clip from
        ``states``/``actions`` only. Returns the FULL predicted latent (seed +
        rolled), ``[B, T'*HW, D]``, with ``T' = states.size(1)``. The caller scores
        it with an energy (``acjepa_energy``).
        """
        hw = self.tokens_per_frame
        Tp = states.size(1)
        z_pred = self._ar_rollout(z_seed, states, actions, ctx_frames,
                                  Tp - ctx_frames, extrinsics)         # future frames
        return torch.cat([z_seed[:, : ctx_frames * hw], z_pred], dim=1)


# --------------------------------------------------------------------------- #
# loss (masked L1 in layer-normed feature space, plan §3.6)
# --------------------------------------------------------------------------- #
def rollout_l1(z, h, hw, frame_valid=None, loss_exp=1.0, start_frame=1):
    """L1 between predicted tokens ``z`` and the matching future target frames.

    ``z`` predicts frames ``start_frame..start_frame+n_pred-1``; targets are
    ``h[:, start_frame*hw : start_frame*hw + z.size(1)]``. If ``frame_valid[B,T']``
    is given, frames whose target arti is invalid (clip tail) are dropped.
    """
    n_pred = z.size(1) // hw
    s = start_frame * hw
    h_t = h[:, s : s + z.size(1)]
    diff = torch.abs(z - h_t) ** loss_exp / loss_exp
    if frame_valid is None:
        return diff.mean()
    fv = frame_valid[:, start_frame : start_frame + n_pred].float()
    w = fv.repeat_interleave(hw, dim=1).unsqueeze(-1)
    denom = w.sum() * z.size(-1)
    return (diff * w).sum() / denom.clamp_min(1.0)
