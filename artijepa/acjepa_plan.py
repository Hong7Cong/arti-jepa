"""CEM / receding-horizon planner for AC-JEPA (aucjepa_plans_new.md §3, M3-M5).

Pure INFERENCE. Given a frozen encoder ``E``, a frozen articulator world-model ``P``
(``acjepa_train``), and a goal phoneme ``p*`` (or a phoneme string), search for a
sequence of articulator actions ``a^{1:T}`` whose rolled-out video latent minimises a
phoneme energy (``acjepa_energy``). The whole search runs in LATENT space -- no pixel
decoding (the JEPA advantage). This is V-JEPA 2-AC's CEM/MPC recipe with the robot
end-effector replaced by the 6-D articulator vector and the goal image replaced by a
phoneme.

Sketch (single goal):
    seed: ctx_frames real frames of a clip -> z_seed (latent), arti_seed (states)
    CEM over a^{1:T} (T x 6 Gaussians):
      sample M candidates -> build full (states, actions) -> P.rollout -> energy
      keep the top-k lowest-energy -> refit (mu, sigma); repeat n_iter
    return argmin action seq (+ rolled latent / arti trajectory)

The CEM loop here is artifact-free and unit-testable with any ``energy_fn``; the CLI
wires it to a trained ``P`` + a chosen energy bridge.
"""

import argparse

import numpy as np
import torch

from artijepa.arti_cond import state_from_actions


# --------------------------------------------------------------------------- #
# building the full (states, actions) for a candidate population
# --------------------------------------------------------------------------- #
def assemble_sequences(arti_seed, cem_actions, ctx_frames):
    """Splice real seed arti with CEM future actions -> full (states, actions).

    Args:
      arti_seed:  [T', 6] real arti for ONE seed clip (only frames < ctx_frames are
                  used as the real prefix; the rest is overwritten by the plan).
      cem_actions:[M, T, 6] candidate future action deltas, T = T' - ctx_frames.
      ctx_frames: number of real seed frames.
    Returns (states [M, T', 6], actions [M, T'-1, 6]) ready for ``P.rollout``.
    """
    M = cem_actions.size(0)
    Tp = arti_seed.size(0)
    dev = cem_actions.device
    seed = arti_seed.to(dev).unsqueeze(0).expand(M, -1, -1)            # [M,T',6]
    # real prefix states (frames 0..ctx_frames-1) + future from cumulative actions
    s_k = seed[:, ctx_frames - 1]                                      # [M,6]
    fut_states = state_from_actions(s_k, cem_actions)[:, 1:]           # [M,T,6] frames ctx..T'-1
    states = torch.cat([seed[:, :ctx_frames], fut_states], dim=1)      # [M,T',6]
    # actions: real deltas for indices 0..ctx_frames-2, CEM for ctx_frames-1..T'-2
    real_act = seed[:, 1:ctx_frames] - seed[:, :ctx_frames - 1]        # [M, ctx-1, 6]
    actions = torch.cat([real_act, cem_actions], dim=1)               # [M, T'-1, 6]
    return states, actions


# --------------------------------------------------------------------------- #
# CEM
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cem_plan(P, z_seed, arti_seed, energy_fn, ctx_frames, horizon, hw,
             M=256, top_k=32, n_iter=8, sigma0=0.3, a_clip=None, seed=0,
             return_traj=True):
    """Cross-Entropy-Method search over future arti actions ``a^{1:horizon}``.

    Args:
      P:         frozen ArtiConditionedPredictor.
      z_seed:    [1, ctx_frames*hw, D] real latent prefix (one clip).
      arti_seed: [T', 6] real arti for that clip.
      energy_fn: (z_full[M,T'*hw,D], states[M,T',6]) -> energy[M]   (lower = better).
      a_clip:    per-step velocity bound (abs) on each action dim, or None.
    Returns dict with best action seq, energy, and (optional) rolled latent / states.
    """
    dev = z_seed.device
    D = z_seed.size(-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = torch.zeros(horizon, 6)
    sigma = torch.full((horizon, 6), float(sigma0))
    z_seed_M = z_seed.expand(M, -1, -1)
    best = {"energy": float("inf"), "actions": None}
    for it in range(n_iter):
        noise = torch.randn(M, horizon, 6, generator=g)
        cand = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise)          # [M,horizon,6]
        if a_clip is not None:
            cand = cand.clamp(-a_clip, a_clip)
        cand = cand.to(dev)
        states, actions = assemble_sequences(arti_seed, cand, ctx_frames)
        z_full = P.rollout(z_seed_M, states, actions, ctx_frames)      # [M,T'*hw,D]
        E = energy_fn(z_full, states)                                  # [M]
        order = torch.argsort(E)
        elite = cand[order[:top_k]].cpu()                             # [k,horizon,6]
        mu = elite.mean(0)
        sigma = elite.std(0).clamp_min(1e-3)
        e_best = float(E[order[0]])
        if e_best < best["energy"]:
            best = {"energy": e_best, "actions": cand[order[0]].cpu().clone()}
    out = {"energy": best["energy"], "actions": best["actions"], "mu": mu, "sigma": sigma}
    if return_traj:
        states, actions = assemble_sequences(arti_seed, best["actions"].to(dev).unsqueeze(0),
                                             ctx_frames)
        out["states"] = states[0].cpu()                               # [T',6]
        out["z"] = P.rollout(z_seed, states, actions, ctx_frames)[0].cpu()
    return out


# --------------------------------------------------------------------------- #
# energy builders (wire a goal phoneme to one of the three energies)
# --------------------------------------------------------------------------- #
def make_energy(kind, hw, *, classifier=None, p_star=None, mu_p=None, s_star=None,
                lam_arti=0.0, arti_target=None):
    """Return an ``energy_fn(z_full, states) -> [M]`` for the configured energy.

    kind: 'classifier' (needs classifier + p_star), 'prototype' (needs mu_p),
    'arti_target' (needs s_star). ``lam_arti`` adds the §1 Energy-3 regulariser
    (needs ``arti_target`` = s*_{p*}) to the classifier/prototype energy.
    """
    from artijepa import acjepa_energy as EN

    def energy_fn(z_full, states):
        if kind == "classifier":
            E = EN.energy_classifier_nll(z_full, classifier, p_star, hw)
        elif kind == "prototype":
            E = EN.energy_prototype(z_full, mu_p, hw)
        elif kind == "arti_target":
            E = EN.energy_arti_target(states, s_star)
        else:
            raise ValueError(f"unknown energy kind {kind!r}")
        if lam_arti and arti_target is not None and kind != "arti_target":
            E = E + lam_arti * EN.energy_arti_target(states, arti_target)
        return E

    return energy_fn


# --------------------------------------------------------------------------- #
# seeding from a real clip
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_seed(encoder, clip, ctx_frames, hw):
    """Encode a clip [1,3,T,H,W] -> layer-normed latent; return the ctx prefix."""
    import torch.nn.functional as F
    h = encoder.backbone(clip)
    h = F.layer_norm(h, (h.size(-1),))
    return h[:, : ctx_frames * hw]


def load_world_model(cfg, ckpt, device):
    """Frozen encoder + ArtiConditionedPredictor with trained weights from ``ckpt``."""
    from artijepa.acjepa_train import build_frozen_encoder
    from artijepa.arti_cond import ArtiConditionedPredictor

    data, model_c, pred_c = cfg["data"], cfg["model"], cfg.get("predictor", {})
    encoder = build_frozen_encoder(cfg, device)
    P = ArtiConditionedPredictor(
        img_size=data["spatial_size"], patch_size=data.get("patch_size", 16),
        num_frames=data["frames_per_clip"], tubelet_size=data.get("tubelet_size", 2),
        embed_dim=encoder.embed_dim, action_embed_dim=int(cfg.get("arti", {}).get("dim", 6)),
        pred_embed_dim=pred_c.get("pred_embed_dim", 384),
        depth=pred_c.get("pred_depth", 12), num_heads=pred_c.get("pred_num_heads", 12),
        use_rope=pred_c.get("use_rope", True), frame_causal=pred_c.get("frame_causal", True),
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)["predictor"]
    P.load_state_dict(state)
    P.eval()
    for p in P.parameters():
        p.requires_grad_(False)
    return encoder, P


def main():
    ap = argparse.ArgumentParser(description="CEM articulator planning toward a goal phoneme")
    ap.add_argument("--config", required=True, help="planner config (acjepa_plan_*.yaml)")
    ap.add_argument("--world-model", required=True, help="trained P checkpoint (latest.pt)")
    ap.add_argument("--target", required=True, help="goal phoneme (ARPABET, e.g. 'm') "
                    "or comma-separated string for a sequence goal")
    ap.add_argument("--seed-clip", type=int, default=0, help="val-set item to seed from")
    args = ap.parse_args()

    from artijepa.tssl_train import load_config, _preproc_from_cfg
    from artijepa.acjepa_dataset import RTMRIArtiDataset
    from artijepa.acjepa_train import _cache_dir
    from artijepa.arti_cond import to_state_action
    from artijepa.phonemes import PHON2IDX
    from artijepa import acjepa_energy as EN

    cfg = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, P = load_world_model(cfg, args.world_model, device)
    hw = P.tokens_per_frame
    cem = cfg.get("cem", {})
    plan_c = cfg.get("plan", {})
    ctx_frames = int(plan_c.get("ctx_frames", 4))
    Tp = cfg["data"]["frames_per_clip"] // cfg["data"].get("tubelet_size", 2)
    horizon = int(plan_c.get("horizon", Tp - ctx_frames))

    # -- seed from a real val clip
    d = cfg["data"]
    pc = _preproc_from_cfg(d, augment=False, random_temporal_crop=False, sampling="tile")
    ds = RTMRIArtiDataset(d["manifest"], split="val", cfg=pc, cache_dir=_cache_dir(cfg),
                          seed=0, normalize=cfg.get("arti", {}).get("normalize", "zscore"))
    item = ds[args.seed_clip]
    clip = item["clip"].unsqueeze(0).to(device)
    arti_seed = item["arti"].float()                                  # [T',6]
    z_seed = encode_seed(encoder, clip, ctx_frames, hw)

    p_star = PHON2IDX[args.target.split(",")[0].strip().lower()]
    kind = plan_c.get("energy", "arti_target")
    # Energy 3 is runnable now (needs an arti-target table); classifier/prototype
    # need the M2 phoneme head / prototypes (load paths here when available).
    s_star = None
    if kind == "arti_target":
        tgt_path = plan_c.get("arti_target_table")
        assert tgt_path, "energy=arti_target needs plan.arti_target_table (build via " \
                         "acjepa_energy.build_arti_targets on cached arti + labels)"
        table = torch.as_tensor(np.load(tgt_path), dtype=torch.float32, device=device)
        s_star = table[p_star]
    energy_fn = make_energy(kind, hw, p_star=p_star, s_star=s_star,
                            lam_arti=float(plan_c.get("lambda_arti", 0.0)))

    out = cem_plan(P, z_seed, arti_seed, energy_fn, ctx_frames, horizon, hw,
                   M=int(cem.get("M", 256)), top_k=int(cem.get("top_k", 32)),
                   n_iter=int(cem.get("n_iter", 8)), sigma0=float(cem.get("sigma0", 0.3)),
                   a_clip=cem.get("a_clip"), seed=cfg["meta"].get("seed", 0))
    print(f"[plan] target={args.target!r} (idx {p_star})  best energy={out['energy']:.4f}")
    print(f"[plan] planned arti trajectory [T',6]:\n{out['states'].numpy().round(3)}")


if __name__ == "__main__":
    main()
