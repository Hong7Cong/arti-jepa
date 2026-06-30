"""Energy functions for AC-JEPA planning (aucjepa_plans_new.md §1, M3).

The forward model ``P`` predicts a CONTINUOUS latent ``ẑ``; the planning goal ``p*``
is a DISCRETE phoneme. These functions are the **bridge** -- they score how well a
rolled-out latent (or arti trajectory) realises a target phoneme. Pure functions on
``(rollout, p*)``; the planner (``acjepa_plan``) minimises them with CEM.

Three energies (plan §1; default = #1):
  1. ``energy_classifier_nll``   -- NLL under a frozen phoneme head ``C: latent->K``.
     Uses ``ẑ``, tolerates coarticulation, probabilistic. RECOMMENDED.
  2. ``energy_prototype``        -- L1 to a per-phoneme mean latent prototype ``μ_p``.
     Keeps the V-JEPA 2-AC L1 form; assumes one canonical latent per phoneme.
  3. ``energy_arti_target``      -- L1 of the final arti STATE to a canonical 6-D
     config ``s*_p``. No video model needed; cheap sanity baseline / regulariser.

All operate on a final-frame (or per-subgoal-frame) slice of the rollout so they
batch over the CEM population ``M`` (the leading dim).
"""

import numpy as np
import torch
import torch.nn.functional as F

from artijepa.phonemes import NUM_PHONEMES


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def frame_slice(z, frame, hw):
    """Latent tokens of ``frame`` from ``z[B, T'*hw, D]`` -> ``[B, hw, D]``."""
    return z[:, frame * hw : (frame + 1) * hw]


def pool_frame(z, frame, hw):
    """Mean-pooled latent of ``frame`` -> ``[B, D]`` (the head/proto input)."""
    return frame_slice(z, frame, hw).mean(dim=1)


# --------------------------------------------------------------------------- #
# Energy 1 -- classifier NLL  (recommended)
# --------------------------------------------------------------------------- #
def energy_classifier_nll(z, classifier, p_star, hw, frame=-1):
    """``ℰ = -log softmax(C(pool(ẑ_frame)))[p*]`` per candidate -> ``[B]``.

    ``classifier`` maps ``[B, D] -> [B, K]`` logits (a frozen phoneme head). ``frame``
    defaults to the last frame of the rollout. ``p_star`` is an int (single target)
    or a per-candidate ``[B]`` long tensor.
    """
    Tp = z.size(1) // hw
    f = Tp - 1 if frame < 0 else frame
    logits = classifier(pool_frame(z, f, hw))                 # [B, K]
    logp = F.log_softmax(logits, dim=-1)
    if not torch.is_tensor(p_star):
        p_star = torch.full((z.size(0),), int(p_star), device=z.device, dtype=torch.long)
    return -logp.gather(1, p_star.view(-1, 1)).squeeze(1)     # [B]


# --------------------------------------------------------------------------- #
# Energy 2 -- phoneme prototype  (L1 form of V-JEPA 2-AC)
# --------------------------------------------------------------------------- #
def energy_prototype(z, mu_p, hw, frame=-1, layer_norm=True):
    """``ℰ = || LN(ẑ_frame) - μ_{p*} ||_1`` per candidate -> ``[B]``.

    ``mu_p`` is the target phoneme's prototype, ``[hw, D]`` (or ``[D]``, broadcast
    over the spatial tokens). Match the LN convention used to BUILD the prototypes.
    """
    Tp = z.size(1) // hw
    f = Tp - 1 if frame < 0 else frame
    zf = frame_slice(z, f, hw)                                # [B, hw, D]
    if layer_norm:
        zf = F.layer_norm(zf, (zf.size(-1),))
    if mu_p.dim() == 1:
        mu_p = mu_p.view(1, 1, -1)
    elif mu_p.dim() == 2:
        mu_p = mu_p.unsqueeze(0)
    return (zf - mu_p).abs().mean(dim=(1, 2))                 # [B]


# --------------------------------------------------------------------------- #
# Energy 3 -- articulatory target  (no learned head; interpretable)
# --------------------------------------------------------------------------- #
def energy_arti_target(states, s_star, frame=-1):
    """``ℰ = || ŝ_frame - s*_{p*} ||_1`` per candidate -> ``[B]``.

    ``states[B, T', 6]`` are the (cumulative) absolute arti states the planner rolls;
    ``s_star`` is the target phoneme's canonical 6-D config (``[6]``). Plans purely in
    articulator space -- the Articulatory-Phonology gestural-target view.
    """
    Tp = states.size(1)
    f = Tp - 1 if frame < 0 else frame
    sf = states[:, f]                                         # [B, 6]
    if s_star.dim() == 1:
        s_star = s_star.view(1, -1)
    return (sf - s_star).abs().mean(dim=1)                    # [B]


# --------------------------------------------------------------------------- #
# building the bridges from labelled data (M2)
# --------------------------------------------------------------------------- #
def build_arti_targets(arti, labels, num_classes=NUM_PHONEMES, ignore=-100):
    """Per-phoneme canonical 6-D config ``s*_p`` = mean arti over frames labelled p.

    ``arti[N, 6]`` and ``labels[N]`` are flattened frame-level arti + phoneme idx
    (use the cached arti-6 + gold/pseudo labels). Returns ``[K, 6]`` (NaN rows for
    phonemes with no frames -- caller should mask those goals out).
    """
    arti = np.asarray(arti, np.float64)
    labels = np.asarray(labels)
    out = np.full((num_classes, arti.shape[1]), np.nan, np.float32)
    for p in range(num_classes):
        m = labels == p
        if m.any():
            out[p] = arti[m].mean(0)
    return out


def build_prototypes(feats, labels, num_classes=NUM_PHONEMES, layer_norm=True):
    """Per-phoneme mean (layer-normed) latent prototype ``μ_p``.

    ``feats[N, hw, D]`` are per-frame pooled-by-token latents with phoneme ``labels``
    ``[N]``. Returns ``[K, hw, D]`` (NaN rows for empty phonemes).
    """
    feats = torch.as_tensor(feats, dtype=torch.float32)
    if layer_norm:
        feats = F.layer_norm(feats, (feats.size(-1),))
    labels = torch.as_tensor(labels)
    K, _, _ = num_classes, feats.size(1), feats.size(2)
    out = torch.full((K, feats.size(1), feats.size(2)), float("nan"))
    for p in range(K):
        m = labels == p
        if m.any():
            out[p] = feats[m].mean(0)
    return out
