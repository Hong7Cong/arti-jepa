#!/usr/bin/env python
"""Zero-GPU smoke test for Articulator-Conditioned JEPA (artijepa env, no transformers).

Validates on a tiny ViT on CPU with FAKE arti-6:
  1. pool_arti_to_tokens alignment (window pooling + validity mask)
  2. to_state_action / state_from_actions shapes + round-trip
  3. ArtiConditionedPredictor forward + forward_predictions + rollout shapes
  4. one full step: FROZEN encoder -> loss.backward() asserts ALL encoder grads None
     and predictor (incl. state/action arti encoders) grads finite
  5. energies (classifier NLL / prototype / arti-target) shapes + monotonicity
  6. planner: assemble_sequences splice + cem_plan converges on a closed-form arti target

Usage:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_acjepa_smoke.py
"""

import sys

import numpy as np
import torch

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({info})" if info else ""))
    return cond


def test_pooling():
    from artijepa.arti_cond import pool_arti_to_tokens
    T_arti, A, n_tok = 10, 6, 8
    feats = np.tile(np.arange(T_arti, dtype=np.float32)[:, None], (1, A))
    e, valid = pool_arti_to_tokens(feats, arti_rate=50.0, n_tok=n_tok, tubelet=2,
                                   target_fps=50.0, clip_start_frame=0)
    check("pool shape", e.shape == (n_tok, A) and valid.shape == (n_tok,), f"{e.shape}")
    check("token0 = mean(0,1)=0.5", abs(e[0, 0] - 0.5) < 1e-5, f"{e[0,0]}")
    check("token4 = mean(8,9)=8.5", abs(e[4, 0] - 8.5) < 1e-5, f"{e[4,0]}")
    check("tail tokens invalid", (not valid[5]) and valid[4], f"valid={valid.tolist()}")
    e2, _ = pool_arti_to_tokens(feats, 50.0, 2, 2, 50.0, clip_start_frame=4)
    check("clip_start offset", abs(e2[0, 0] - 4.5) < 1e-5, f"{e2[0,0]}")


def test_state_action():
    from artijepa.arti_cond import to_state_action, state_from_actions
    e = torch.randn(2, 8, 6)
    s, a = to_state_action(e)
    check("state shape == e", s.shape == (2, 8, 6))
    check("action shape T-1", a.shape == (2, 7, 6))
    check("action = diff", torch.allclose(a, e[:, 1:] - e[:, :-1]))
    # cumulative reconstruction: state_from_actions(s0, diffs) == original states
    rebuilt = state_from_actions(e[:, 0], a)
    check("state_from_actions round-trip", torch.allclose(rebuilt, e, atol=1e-5))


def _build(device="cpu", img=96, patch=16, frames=16, tub=2, A=6):
    from artijepa.arti_cond import ArtiConditionedPredictor
    from artijepa.model import build_models
    encoder, _ = build_models(device=torch.device(device), model_name="vit_tiny",
                              spatial_size=img, frames_per_clip=frames, patch_size=patch,
                              tubelet_size=tub, num_mask_tokens=1,
                              use_activation_checkpointing=False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    pred = ArtiConditionedPredictor(
        img_size=img, patch_size=patch, num_frames=frames, tubelet_size=tub,
        embed_dim=encoder.embed_dim, action_embed_dim=A, pred_embed_dim=64,
        depth=2, num_heads=4, use_rope=True, frame_causal=True).to(device)
    return encoder, pred, A


def test_predictor_forward():
    torch.manual_seed(0)
    enc, pred, A = _build()
    B, frames, img, patch, tub = 2, 16, 96, 16, 2
    Tp, hw = frames // tub, (img // patch) ** 2
    clip = torch.randn(B, 3, frames, img, img)
    with torch.no_grad():
        h = enc.backbone(clip)
    D = h.size(-1)
    check("encoder tokens = T'*HW", h.shape == (B, Tp * hw, D), f"{tuple(h.shape)}")
    check("tokens_per_frame", pred.tokens_per_frame == hw, pred.tokens_per_frame)
    arti = torch.randn(B, Tp, A)
    from artijepa.arti_cond import to_state_action
    state, action = to_state_action(arti)
    z_tf, z_ar, ar0 = pred.forward_predictions(h, state, action, auto_steps=2)
    check("z_tf shape", z_tf.shape == (B, (Tp - 1) * hw, D), f"{tuple(z_tf.shape)}")
    check("z_ar shape (auto_steps)", z_ar.shape == (B, 2 * hw, D) and ar0 == 1)
    K = 5
    _, z_ar_c, ar0c = pred.forward_predictions(h, state, action, ctx_frames=K)
    check("ctx-prefix z_ar shape", z_ar_c.shape == (B, (Tp - K) * hw, D) and ar0c == K)
    # planner-facing full rollout: seed + future == whole clip
    z_full = pred.rollout(h[:, : K * hw], state, action, ctx_frames=K)
    check("rollout full shape", z_full.shape == (B, Tp * hw, D), f"{tuple(z_full.shape)}")


def test_frozen_step():
    torch.manual_seed(0)
    enc, pred, A = _build()
    from artijepa.arti_cond import rollout_l1, to_state_action
    import torch.nn.functional as F
    B, frames, img, patch, tub = 2, 16, 96, 16, 2
    Tp, hw = frames // tub, (img // patch) ** 2
    clip = torch.randn(B, 3, frames, img, img)
    arti = torch.randn(B, Tp, A)
    valid = torch.ones(B, Tp, dtype=torch.bool)
    with torch.no_grad():
        h = enc.backbone(clip)
        h = F.layer_norm(h, (h.size(-1),))
    state, action = to_state_action(arti)
    z_tf, z_ar, ar0 = pred.forward_predictions(h, state, action, auto_steps=2, ctx_frames=4)
    loss = rollout_l1(z_tf, h, hw, valid, start_frame=1) + rollout_l1(z_ar, h, hw, valid, start_frame=ar0)
    loss.backward()
    check("loss finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")
    enc_grads = [p.grad for p in enc.parameters() if p.grad is not None]
    check("ALL encoder grads None (frozen)", len(enc_grads) == 0, f"{len(enc_grads)} non-None")
    se = pred.backbone.state_encoder.weight.grad
    ae = pred.backbone.action_encoder.weight.grad
    check("state_encoder grad finite", se is not None and torch.isfinite(se).all().item())
    check("action_encoder grad finite", ae is not None and torch.isfinite(ae).all().item())


def test_energies():
    from artijepa import acjepa_energy as EN
    B, Tp, hw, D, K = 4, 8, 9, 16, 41
    z = torch.randn(B, Tp * hw, D)
    clf = torch.nn.Linear(D, K)
    e1 = EN.energy_classifier_nll(z, clf, p_star=3, hw=hw)
    check("classifier NLL shape", e1.shape == (B,) and torch.isfinite(e1).all())
    mu = torch.randn(hw, D)
    e2 = EN.energy_prototype(z, mu, hw)
    check("prototype shape", e2.shape == (B,) and torch.isfinite(e2).all())
    states = torch.randn(B, Tp, 6)
    s_star = states[0, -1].clone()                  # candidate 0 sits exactly on target
    e3 = EN.energy_arti_target(states, s_star)
    check("arti-target shape", e3.shape == (B,))
    check("arti-target zero at target", float(e3[0]) < 1e-6, f"{float(e3[0]):.4g}")
    # build_arti_targets averages per label
    arti = np.array([[1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0]], np.float32)
    labels = np.array([5, 5, 7])
    tbl = EN.build_arti_targets(arti, labels, num_classes=K)
    check("build_arti_targets mean", np.allclose(tbl[5], 2.0) and np.allclose(tbl[7], 0.0))


def test_planner():
    from artijepa.acjepa_plan import assemble_sequences, cem_plan
    # assemble: real prefix preserved, future states = cumsum of actions from s_k
    Tp, ctx, M = 8, 3, 5
    arti_seed = torch.arange(Tp * 6, dtype=torch.float32).reshape(Tp, 6)
    cem_actions = torch.zeros(M, Tp - ctx, 6)        # zero actions -> states hold s_k
    states, actions = assemble_sequences(arti_seed, cem_actions, ctx)
    check("assemble states shape", states.shape == (M, Tp, 6))
    check("assemble actions shape", actions.shape == (M, Tp - 1, 6))
    check("real prefix preserved", torch.allclose(states[0, :ctx], arti_seed[:ctx]))
    check("zero-action future holds s_k",
          torch.allclose(states[0, ctx:], arti_seed[ctx - 1].expand(Tp - ctx, 6)))

    # CEM should drive a planned arti state toward a target (closed-form Energy-3).
    # Dummy predictor: rollout returns zeros (energy ignores z); energy = arti target.
    hw, D = 4, 8
    target = torch.full((6,), 5.0)

    class DummyP:
        tokens_per_frame = hw

        def rollout(self, z_seed, states, actions, ctx_frames, extrinsics=None):
            return torch.zeros(states.size(0), states.size(1) * hw, D)

    def energy_fn(z_full, states):
        return (states[:, -1] - target).abs().mean(dim=1)

    z_seed = torch.zeros(1, ctx * hw, D)
    out = cem_plan(DummyP(), z_seed, arti_seed, energy_fn, ctx_frames=ctx,
                   horizon=Tp - ctx, hw=hw, M=256, top_k=16, n_iter=24, sigma0=1.0, seed=0)
    final_state = out["states"][-1]
    check("CEM reduces arti-target energy", out["energy"] < 0.3, f"E={out['energy']:.3f}")
    check("CEM final state near target", torch.allclose(final_state, target, atol=0.6),
          f"{final_state.numpy().round(2)}")


def main():
    print("== Articulator-Conditioned JEPA smoke test ==")
    print("[1] arti pooling/alignment"); test_pooling()
    print("[2] state/action");           test_state_action()
    print("[3] predictor forward");      test_predictor_forward()
    print("[4] frozen-encoder step");    test_frozen_step()
    print("[5] energies");               test_energies()
    print("[6] planner CEM");            test_planner()
    n_pass = sum(ok for _, ok in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILED:", [n for n, ok in results if not ok])
        sys.exit(1)
    print("ALL AC-JEPA (articulator) SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
