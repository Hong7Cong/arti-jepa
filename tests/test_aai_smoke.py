#!/usr/bin/env python
"""Zero-GPU smoke test for AC-JEPA-audio (run in the artijepa env, no transformers).

Validates on a tiny ViT on CPU with FAKE audio:
  1. pool_audio_to_tokens alignment (window pooling + validity mask)
  2. to_state_action shapes
  3. AudioConditionedPredictor forward + forward_predictions shapes
  4. one full step: FROZEN encoder -> loss.backward() asserts ALL encoder grads
     are None and predictor (incl. state/action audio encoders) grads are finite
     -- the core "encoder frozen, predictor learns" guarantee.

Usage:
    cd /project2/shrikann_35/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_aucjepa_smoke.py
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
    from artijepa.audio_cond import pool_audio_to_tokens
    # feats[i,:] = i ; 50 Hz audio, 50 fps grid, tubelet 2 -> token j = mean(2j, 2j+1)
    T_audio, A, n_tok = 10, 4, 8
    feats = np.tile(np.arange(T_audio, dtype=np.float32)[:, None], (1, A))
    e, valid = pool_audio_to_tokens(feats, audio_rate=50.0, n_tok=n_tok, tubelet=2,
                                    target_fps=50.0, clip_start_frame=0)
    check("pool shape", e.shape == (n_tok, A) and valid.shape == (n_tok,), f"{e.shape}")
    check("token0 = mean(0,1)=0.5", abs(e[0, 0] - 0.5) < 1e-5, f"{e[0,0]}")
    check("token4 = mean(8,9)=8.5", abs(e[4, 0] - 8.5) < 1e-5, f"{e[4,0]}")
    check("tail tokens invalid (>=audio end)", (not valid[5]) and valid[4],
          f"valid={valid.tolist()}")
    # clip_start offset shifts the window
    e2, _ = pool_audio_to_tokens(feats, 50.0, 2, 2, 50.0, clip_start_frame=4)
    check("clip_start offset", abs(e2[0, 0] - 4.5) < 1e-5, f"{e2[0,0]}")


def test_state_action():
    from artijepa.audio_cond import to_state_action
    e = torch.randn(2, 8, 16)
    s, a = to_state_action(e)
    check("state shape == e", s.shape == (2, 8, 16))
    check("action shape T-1", a.shape == (2, 7, 16))
    check("action = diff", torch.allclose(a, e[:, 1:] - e[:, :-1]))


def _build(device="cpu", img=96, patch=16, frames=16, tub=2, A=32, cond_mode="concat"):
    from artijepa.audio_cond import AudioConditionedPredictor
    from artijepa.model import build_models
    encoder, _ = build_models(device=torch.device(device), model_name="vit_tiny",
                              spatial_size=img, frames_per_clip=frames, patch_size=patch,
                              tubelet_size=tub, num_mask_tokens=1,
                              use_activation_checkpointing=False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    pred = AudioConditionedPredictor(
        img_size=img, patch_size=patch, num_frames=frames, tubelet_size=tub,
        embed_dim=encoder.embed_dim, action_embed_dim=A, pred_embed_dim=64,
        depth=2, num_heads=4, use_rope=True, frame_causal=True,
        cond_mode=cond_mode).to(device)
    return encoder, pred, A


def test_predictor_forward():
    torch.manual_seed(0)
    enc, pred, A = _build()
    B, frames, img, patch, tub = 2, 16, 96, 16, 2
    Tp = frames // tub                                 # 8 temporal tokens
    hw = (img // patch) ** 2                           # 36 tokens/frame
    clip = torch.randn(B, 3, frames, img, img)
    with torch.no_grad():
        h = enc.backbone(clip)                         # [B, Tp*hw, D]
    D = h.size(-1)
    check("encoder tokens = T'*HW", h.shape == (B, Tp * hw, D), f"{tuple(h.shape)}")
    check("tokens_per_frame", pred.tokens_per_frame == hw, pred.tokens_per_frame)
    audio = torch.randn(B, Tp, A)
    from artijepa.audio_cond import to_state_action
    state, action = to_state_action(audio)
    # direct forward over T context frames
    z = pred(h[:, : (Tp - 1) * hw], action, state[:, :-1])
    check("direct forward shape", z.shape == (B, (Tp - 1) * hw, D), f"{tuple(z.shape)}")
    # rollout: droid-style (auto_steps from frame 0)
    z_tf, z_ar, ar0 = pred.forward_predictions(h, state, action, auto_steps=2)
    check("z_tf shape (T'-1 frames)", z_tf.shape == (B, (Tp - 1) * hw, D), f"{tuple(z_tf.shape)}")
    check("z_ar shape (auto_steps frames)", z_ar.shape == (B, 2 * hw, D) and ar0 == 1,
          f"{tuple(z_ar.shape)} ar0={ar0}")
    # rollout: context-prefix (plan §4) -- seed K real frames, predict the rest
    K = 5
    _, z_ar_c, ar0c = pred.forward_predictions(h, state, action, ctx_frames=K)
    check("ctx-prefix z_ar shape (T'-K frames)",
          z_ar_c.shape == (B, (Tp - K) * hw, D) and ar0c == K, f"{tuple(z_ar_c.shape)} ar0={ar0c}")


def test_frozen_step():
    """The keystone guarantee: encoder grads None, predictor grads finite."""
    torch.manual_seed(0)
    enc, pred, A = _build()
    from artijepa.audio_cond import rollout_l1, to_state_action
    B, frames, img, patch, tub = 2, 16, 96, 16, 2
    Tp, hw = frames // tub, (img // patch) ** 2
    clip = torch.randn(B, 3, frames, img, img)
    audio = torch.randn(B, Tp, A)
    valid = torch.ones(B, Tp, dtype=torch.bool)
    import torch.nn.functional as F
    with torch.no_grad():
        h = enc.backbone(clip)
        h = F.layer_norm(h, (h.size(-1),))
    state, action = to_state_action(audio)
    z_tf, z_ar, ar0 = pred.forward_predictions(h, state, action, auto_steps=2, ctx_frames=4)
    loss = rollout_l1(z_tf, h, hw, valid, start_frame=1) + rollout_l1(z_ar, h, hw, valid, start_frame=ar0)
    loss.backward()
    check("loss finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")
    enc_grads = [p.grad for p in enc.parameters() if p.grad is not None]
    check("ALL encoder grads None (frozen)", len(enc_grads) == 0, f"{len(enc_grads)} non-None")
    # predictor: state/action audio encoders + a block must have finite grads
    se = pred.backbone.state_encoder.weight.grad
    ae = pred.backbone.action_encoder.weight.grad
    check("state_encoder grad finite", se is not None and torch.isfinite(se).all().item())
    check("action_encoder grad finite", ae is not None and torch.isfinite(ae).all().item())
    n_pred_grad = sum(1 for p in pred.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_pred_req = sum(1 for p in pred.parameters() if p.requires_grad)
    check("most predictor params have finite grad", n_pred_grad >= n_pred_req - 2,
          f"{n_pred_grad}/{n_pred_req}")


def test_cond_modes():
    """film / cross_attn conditioning: same rollout shapes as concat, encoder stays
    frozen, and the mode-specific audio params (film / xattn / gate) receive grad."""
    import torch.nn.functional as F
    from artijepa.audio_cond import rollout_l1, to_state_action
    B, frames, img, patch, tub = 2, 16, 96, 16, 2
    Tp, hw = frames // tub, (img // patch) ** 2
    for mode in ("film", "cross_attn"):
        torch.manual_seed(0)
        enc, pred, A = _build(cond_mode=mode)
        check(f"[{mode}] cond_mode set", pred.cond_mode == mode)
        clip = torch.randn(B, 3, frames, img, img)
        audio = torch.randn(B, Tp, A)
        valid = torch.ones(B, Tp, dtype=torch.bool)
        with torch.no_grad():
            h = F.layer_norm(enc.backbone(clip), (enc.embed_dim,))
        D = h.size(-1)
        state, action = to_state_action(audio)
        # ctx=1 (the new run's setting): seed 1 real frame, predict the other T'-1
        z_tf, z_ar, ar0 = pred.forward_predictions(h, state, action, ctx_frames=1)
        check(f"[{mode}] z_tf shape", z_tf.shape == (B, (Tp - 1) * hw, D), f"{tuple(z_tf.shape)}")
        check(f"[{mode}] ctx=1 z_ar shape", z_ar.shape == (B, (Tp - 1) * hw, D) and ar0 == 1,
              f"{tuple(z_ar.shape)} ar0={ar0}")
        loss = (rollout_l1(z_tf, h, hw, valid, start_frame=1)
                + rollout_l1(z_ar, h, hw, valid, start_frame=ar0))
        loss.backward()
        check(f"[{mode}] loss finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")
        check(f"[{mode}] encoder frozen (no grads)",
              sum(1 for p in enc.parameters() if p.grad is not None) == 0)
        if mode == "film":
            g = pred.film[0].weight.grad
            check("[film] FiLM generator grad finite",
                  g is not None and torch.isfinite(g).all().item())
        else:
            xg = pred.xattn_gate[0].grad
            xq = pred.xattn[0].in_proj_weight.grad
            check("[cross_attn] gate + attn grads finite",
                  xg is not None and torch.isfinite(xg).all().item()
                  and xq is not None and torch.isfinite(xq).all().item())


def test_concat_unchanged():
    """concat mode must call the stock backbone verbatim (no new params added)."""
    _, pred_c, _ = _build(cond_mode="concat")
    has_extra = any(hasattr(pred_c, a) for a in ("film", "xattn", "xattn_gate"))
    check("[concat] no film/xattn modules added", not has_extra)


def main():
    print("== AC-JEPA-audio smoke test ==")
    print("[1] audio pooling/alignment"); test_pooling()
    print("[2] state/action");            test_state_action()
    print("[3] predictor forward");       test_predictor_forward()
    print("[4] frozen-encoder step");     test_frozen_step()
    print("[5] film/cross_attn modes");   test_cond_modes()
    print("[6] concat unchanged");        test_concat_unchanged()
    n_pass = sum(ok for _, ok in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILED:", [n for n, ok in results if not ok])
        sys.exit(1)
    print("ALL AC-JEPA SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
