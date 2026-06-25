"""Acoustic-Conditioned JEPA training loop (plans_aucjepa.md §6).

Forked from ``tssl_train.py`` but: the V-JEPA 2 ViT-L encoder is **frozen** (no EMA,
target == context == the frozen encoder), and the *only* trainable parameters are
an **audio-conditioned predictor** (``src.models.ac_predictor`` reused verbatim,
wrapped by ``audio_cond.AudioConditionedPredictor``) plus its state/action audio
projections. Objective = temporal world-model: encode the whole clip (frozen) ->
predict next-frame tokens conditioned on per-frame WavLM audio (``state=e[t]``,
``action=e[t+1]-e[t]``), teacher-forced + ``auto_steps`` autoregressive (the droid
``forward_predictions`` recipe). Loss = L1 in layer-normed feature space.

Run:
    source dev_artiJEPA/scripts/_env.sh
    python -m artijepa.aucjepa_train --config dev_artiJEPA/configs/aucjepa_vitl_128.yaml
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from artijepa.audio_cond import AudioConditionedPredictor, rollout_l1, to_state_action
from artijepa.aucjepa_dataset import RTMRIAudioDataset, collate
from artijepa.checkpoint import clean_backbone_key, filtered_load, resolve_checkpoint
from artijepa.model import build_models
from artijepa.tssl_train import _preproc_from_cfg, load_config


def init_opt_predictor_only(predictor, iterations_per_epoch, start_lr, ref_lr,
                            warmup, num_epochs, wd, final_wd, final_lr,
                            mixed_precision, ipe_scale=1.0, betas=(0.9, 0.999),
                            eps=1e-8):
    """AdamW + cosine schedules over the predictor's trainable params ONLY.

    The frozen encoder is not handed to the optimizer at all (avoids bloating
    optimizer state with frozen tensors). Mirrors ``app.vjepa.utils.init_opt``
    (no-WD on biases / 1-D params).
    """
    from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

    named = [(n, p) for n, p in predictor.named_parameters() if p.requires_grad]
    param_groups = [
        {"params": [p for n, p in named if ("bias" not in n) and (p.ndim != 1)]},
        {"params": [p for n, p in named if ("bias" in n) or (p.ndim == 1)],
         "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=wd, final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def build_frozen_encoder(cfg, device):
    """Build the ViT-L encoder, load (domain-adapted) weights, freeze it."""
    data, model_c = cfg["data"], cfg["model"]
    encoder, _unused_pred = build_models(
        device=device, model_name=model_c.get("model_name", "vit_large"),
        spatial_size=data["spatial_size"], frames_per_clip=data["frames_per_clip"],
        patch_size=data.get("patch_size", 16), tubelet_size=data.get("tubelet_size", 2),
        num_mask_tokens=1, use_activation_checkpointing=False)
    del _unused_pred                                   # masked predictor unused here
    # -- load encoder weights only (NOT the masked predictor)
    ckpt_path = resolve_checkpoint(model_c.get("model_name", "vit_large"),
                                   model_c.get("checkpoint"))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = model_c.get("checkpoint_key", "target_encoder")
    if key not in ck:
        for k in ("target_encoder", "encoder", "ema_encoder"):
            if k in ck:
                key = k
                break
    n, miss, skip = filtered_load(encoder.backbone, clean_backbone_key(ck[key]))
    print(f"[aucjepa] encoder<-{os.path.basename(ckpt_path)}:{key} loaded {n} "
          f"tensors, {len(miss)} missing, {len(skip)} skipped")
    del ck
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


def build_audio_loader(cfg, split, audio_dir, shuffle, drop_last):
    d = cfg["data"]
    pc = _preproc_from_cfg(d, augment=(d.get("augment", True) and split == "train"),
                           random_temporal_crop=False, sampling="tile")
    ds = RTMRIAudioDataset(d["manifest"], split=split, cfg=pc, audio_dir=audio_dir,
                           seed=cfg["meta"]["seed"],
                           normalize=cfg.get("audio", {}).get("normalize", "zscore"))
    nw = d.get("num_workers", 4)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=d["batch_size"], shuffle=shuffle, drop_last=drop_last,
        num_workers=nw, pin_memory=d.get("pin_mem", True),
        # persistent_workers defaults FALSE: keeping workers alive across epochs lets
        # their anon/shmem memory creep until the job's cgroup --mem cap OOM-kills them
        # (seen at epoch 5, --mem=32G). Respawn per epoch releases it; cost is amortised
        # over ipe steps. Re-enable via data.persistent_workers if RAM is plentiful.
        persistent_workers=bool(d.get("persistent_workers", False)) and nw > 0,
        collate_fn=collate)
    return ds, loader


@torch.no_grad()
def run_diagnostics(encoder, predictor, loader, device, dtype, mixed, auto_steps,
                    hw, ctx_frames=None, max_batches=8):
    """Audio-conditioned future-pred L1 with REAL vs SHUFFLED audio (plan §10).

    A large real-vs-shuffled gap = the predictor actually uses the acoustics. The
    gap is measured on the AUTOREGRESSIVE branch (where audio is necessary -- the
    teacher-forced branch can lean on the visible past frames). Also reports the
    teacher-forced vs AR loss levels.
    """
    predictor.eval()
    real_tf, real_ar, shuf_ar = [], [], []
    for i, (clips, audio, valid) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True).float()
        valid = valid.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
            h = encoder.backbone(clips)
            h = F.layer_norm(h, (h.size(-1),))
            st, ac = to_state_action(audio)
            z_tf, z_ar, ar0 = predictor.forward_predictions(h, st, ac, auto_steps, ctx_frames)
            real_tf.append(float(rollout_l1(z_tf, h, hw, valid, start_frame=1)))
            real_ar.append(float(rollout_l1(z_ar, h, hw, valid, start_frame=ar0)))
            # shuffled audio across the batch (break the video<->audio pairing)
            perm = torch.randperm(audio.size(0), device=device)
            st_s, ac_s = to_state_action(audio[perm])
            _, z_ar_s, _ = predictor.forward_predictions(h, st_s, ac_s, auto_steps, ctx_frames)
            shuf_ar.append(float(rollout_l1(z_ar_s, h, hw, valid, start_frame=ar0)))
    predictor.train()
    if not real_tf:
        return {}
    m = lambda v: float(np.mean(v))  # noqa: E731
    return {"val_tf_l1": m(real_tf), "val_ar_l1": m(real_ar),
            "val_shuf_ar_l1": m(shuf_ar), "audio_gap": m(shuf_ar) - m(real_ar)}


def train(cfg):
    meta, data, opt_c, loss_c, model_c = (
        cfg["meta"], cfg["data"], cfg["optimization"], cfg["loss"], cfg["model"])
    audio_c, pred_c = cfg.get("audio", {}), cfg.get("predictor", {})
    folder = meta["folder"]
    os.makedirs(folder, exist_ok=True)
    seed = meta.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    which = meta.get("dtype", "float32").lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(which, torch.float32)
    mixed = dtype != torch.float32 and device.type == "cuda"
    print(f"[aucjepa] device={device} dtype={dtype} mixed={mixed}")

    audio_dir = audio_c["cache_dir"]
    A = int(audio_c.get("dim", 768))
    auto_steps = int(opt_c.get("auto_steps", 2))
    # context-prefix rollout (plan §4): seed ctx_frames real frames, predict the
    # rest from audio only -> makes audio causally necessary. None -> droid AR.
    ctx_frames = cfg.get("ctx_frames")
    objective = cfg.get("objective", "temporal")
    assert objective == "temporal", \
        f"objective={objective!r} not implemented; only 'temporal' (masked is a TODO)"

    # -- data
    ssl_ds, loader = build_audio_loader(cfg, "train", audio_dir, shuffle=True, drop_last=True)
    try:
        _, monitor_loader = build_audio_loader(cfg, "val", audio_dir, shuffle=False, drop_last=False)
    except (ValueError, KeyError):
        monitor_loader = None
    ipe = opt_c.get("ipe") or len(loader)
    epochs = opt_c["epochs"]
    bs = data.get("batch_size", 1)
    eff = opt_c.get("effective_batch")
    accum_steps = max(1, int(opt_c.get("accum_steps") or (round(eff / bs) if eff else 1)))
    oue = max(1, ipe // accum_steps)
    print(f"[aucjepa] {len(ssl_ds)} train clips, ipe={ipe}, epochs={epochs}; "
          f"batch={bs} x accum={accum_steps} -> eff_batch={bs*accum_steps}, {oue} oue")

    # -- models: frozen encoder + trainable AC predictor
    encoder = build_frozen_encoder(cfg, device)
    predictor = AudioConditionedPredictor(
        img_size=data["spatial_size"], patch_size=data.get("patch_size", 16),
        num_frames=data["frames_per_clip"], tubelet_size=data.get("tubelet_size", 2),
        embed_dim=encoder.embed_dim, action_embed_dim=A,
        pred_embed_dim=pred_c.get("pred_embed_dim", 384),
        depth=pred_c.get("pred_depth", 12), num_heads=pred_c.get("pred_num_heads", 12),
        use_rope=pred_c.get("use_rope", True), frame_causal=pred_c.get("frame_causal", True),
        use_activation_checkpointing=model_c.get("use_activation_checkpointing", False),
    ).to(device)
    hw = predictor.tokens_per_frame
    n_train = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"[aucjepa] trainable predictor params {n_train/1e6:.1f}M; "
          f"frozen encoder params {n_enc/1e6:.1f}M; tokens/frame={hw}, A={A}, "
          f"auto_steps={auto_steps}, ctx_frames={ctx_frames}")

    optimizer, scaler, scheduler, wd_scheduler = init_opt_predictor_only(
        predictor, iterations_per_epoch=oue, start_lr=opt_c["start_lr"],
        ref_lr=opt_c["lr"], warmup=opt_c["warmup"], num_epochs=epochs,
        wd=float(opt_c["weight_decay"]), final_wd=float(opt_c["final_weight_decay"]),
        final_lr=opt_c["final_lr"], mixed_precision=mixed,
        ipe_scale=opt_c.get("ipe_scale", 1.0),
        betas=tuple(opt_c.get("betas", (0.9, 0.999))), eps=float(opt_c.get("eps", 1e-8)))

    loss_exp = loss_c.get("loss_exp", 1.0)
    max_steps = meta.get("max_steps")
    log_path = os.path.join(folder, "train_log.csv")
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:
        log_f.write("epoch,itr,loss,jloss,sloss,lr,wd,iter_ms\n")

    # -- resume (predictor + opt + scaler + epoch; encoder is frozen/reproducible)
    start_epoch = 0
    resume_path = meta.get("resume")
    if resume_path and os.path.exists(resume_path):
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        predictor.load_state_dict(rck["predictor"])
        optimizer.load_state_dict(rck["opt"])
        if scaler is not None and rck.get("scaler") is not None:
            scaler.load_state_dict(rck["scaler"])
        start_epoch = int(rck["epoch"])
        for _ in range(start_epoch * oue):
            scheduler.step(); wd_scheduler.step()
        print(f"[aucjepa] RESUMED {os.path.basename(resume_path)} @ epoch {start_epoch} "
              f"({start_epoch*oue} opt-updates fast-forwarded); -> epoch {epochs}")

    global_step = start_epoch * ipe
    new_lr, new_wd = 0.0, 0.0
    for epoch in range(start_epoch, epochs):
        predictor.train()
        running = rjl = rsl = 0.0
        n_seen = 0
        for itr, (clips, audio, valid) in enumerate(loader):
            if itr >= ipe:
                break
            t0 = time.time()
            clips = clips.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True).float()
            valid = valid.to(device, non_blocking=True)

            if itr % accum_steps == 0:
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()
            is_boundary = (itr + 1) % accum_steps == 0 or itr + 1 == ipe

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
                with torch.no_grad():                          # frozen perceptual target
                    h = encoder.backbone(clips)                # [B, T'*HW, D]
                    h = F.layer_norm(h, (h.size(-1),))
                state, action = to_state_action(audio)         # [B,T',A], [B,T'-1,A]
                z_tf, z_ar, ar0 = predictor.forward_predictions(
                    h, state, action, auto_steps, ctx_frames)
                jloss = rollout_l1(z_tf, h, hw, valid, loss_exp, start_frame=1)
                sloss = rollout_l1(z_ar, h, hw, valid, loss_exp, start_frame=ar0)
                loss = jloss + sloss

            loss_b = loss / accum_steps
            if mixed:
                scaler.scale(loss_b).backward()
            else:
                loss_b.backward()
            if is_boundary:
                if mixed:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            lval, jl, sl = float(loss), float(jloss), float(sloss)
            assert not np.isnan(lval), "loss is NaN (instability -- lower lr / auto_steps)"
            running += lval; rjl += jl; rsl += sl; n_seen += 1
            dt = (time.time() - t0) * 1000.0
            log_f.write(f"{epoch+1},{itr},{lval:.5f},{jl:.5f},{sl:.5f},"
                        f"{new_lr:.3e},{new_wd:.3e},{dt:.0f}\n")
            if itr % 10 == 0:
                log_f.flush()
                print(f"[e{epoch+1} {itr}/{ipe}] loss={running/n_seen:.4f} "
                      f"(j={rjl/n_seen:.4f} s={rsl/n_seen:.4f}) lr={new_lr:.2e} {dt:.0f}ms")
            global_step += 1
            if max_steps and global_step >= max_steps:
                print(f"[aucjepa] hit max_steps={max_steps}, stopping early")
                break

        # -- diagnostics: audio real-vs-shuffled gap + TF/AR gap
        if monitor_loader is not None and (epoch + 1) % meta.get("eval_freq", 1) == 0:
            diag = run_diagnostics(encoder, predictor, monitor_loader, device, dtype,
                                   mixed, auto_steps, hw, ctx_frames=ctx_frames,
                                   max_batches=meta.get("probe_max_batches", 8))
            print(f"[e{epoch+1}] diagnostics: {diag}")
            with open(os.path.join(folder, "diagnostics.jsonl"), "a") as df:
                df.write(json.dumps({"epoch": epoch + 1, **diag}) + "\n")

        # -- checkpoint (atomic; predictor/opt/scaler/epoch -- encoder is frozen)
        if (epoch + 1) % meta.get("save_freq", 1) == 0:
            ckpt_path = os.path.join(folder, "latest.pt")
            torch.save({
                "predictor": predictor.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1, "loss": running / max(1, n_seen),
                "encoder_ckpt": model_c.get("checkpoint"),
                "encoder_key": model_c.get("checkpoint_key", "target_encoder"),
            }, ckpt_path + ".tmp")
            if os.path.exists(ckpt_path):
                os.replace(ckpt_path, ckpt_path + ".prev")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            snap = meta.get("snapshot_freq")
            if snap and (epoch + 1) % snap == 0:
                shutil.copyfile(ckpt_path, os.path.join(folder, f"epoch_{epoch+1}.pt"))
        print(f"[aucjepa] epoch {epoch+1} avg loss {running/max(1,n_seen):.4f}")
        if max_steps and global_step >= max_steps:
            break

    log_f.close()
    print("[aucjepa] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.setdefault("meta", {})["max_steps"] = args.max_steps
    if args.resume is not None:
        cfg.setdefault("meta", {})["resume"] = args.resume
    train(cfg)


if __name__ == "__main__":
    main()
