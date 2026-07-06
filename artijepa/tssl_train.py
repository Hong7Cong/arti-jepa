"""Domain-adaptive self-supervised pre-training on rtMRI (Arti-JEPA Part B, T-SSL).

Continues the V-JEPA mask-denoising objective (EMA target encoder + stop-grad +
L1 feature loss + multiblock-3D masking) on *unlabeled* rtMRI, initialised from
the pretrained V-JEPA 2 ViT-L. Single-process (no DDP/SLURM) so it runs on one
V100. Periodically logs label-free representation-collapse diagnostics (the weak
stimulus-group probe was removed; downstream eval is phoneme prediction, see
``eval_phoneme.py``).

Run:
    cd /data2/hongn/vjepa2
    PYTHONPATH=.:dev_artiJEPA python -m artijepa.tssl_train \
        --config dev_artiJEPA/configs/tssl_vitl_256.yaml
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

from artijepa.checkpoint import resolve_checkpoint
from artijepa.collapse import extract_features, feature_diagnostics, simple_collate
from artijepa.masking import mask_config_for
from artijepa.model import build_models, load_pretrained, make_target_encoder
from artijepa.rtmri_dataset import PreprocConfig, RTMRIVideoDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _preproc_from_cfg(d, augment, random_temporal_crop, sampling="crop"):
    gmean, gstd = 0.0, 1.0
    stats_path = d.get("grayscale_stats")
    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            st = json.load(f)
        gmean, gstd = st["mean"], st["std"]
    return PreprocConfig(
        target_fps=d.get("target_fps", 50.0),
        frames_per_clip=d["frames_per_clip"],
        sampling=sampling,
        spatial_mode=d.get("spatial_mode", "resize"),
        spatial_size=d["spatial_size"],
        intensity_norm=d.get("intensity_norm", "zscore"),
        grayscale_mean=gmean, grayscale_std=gstd,
        augment=augment, random_temporal_crop=random_temporal_crop,
        num_clips=1, tubelet_size=d.get("tubelet_size", 2),
    )


def build_ssl_loader(cfg, mask_collator):
    d = cfg["data"]
    pc = _preproc_from_cfg(d, augment=d.get("augment", True), random_temporal_crop=True,
                           sampling=d.get("sampling", "crop"))
    ds = RTMRIVideoDataset(d["manifest"], split="train", cfg=pc, seed=cfg["meta"]["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=d["batch_size"], shuffle=True, drop_last=True,
        num_workers=d.get("num_workers", 8), pin_memory=d.get("pin_mem", True),
        persistent_workers=d.get("num_workers", 8) > 0, collate_fn=mask_collator,
    )
    return ds, loader


def build_monitor_loader(cfg):
    """Held-out feature loader for *label-free* collapse monitoring.

    One centre clip per held-out video (``sampling='crop'``, no aug). We only need
    the pooled features to measure feature_std / effective_rank / mean_abs_cosine
    -- no labels. (Downstream phoneme evaluation lives in ``eval_phoneme.py``; the
    old weak stimulus-group probe was dropped as not meaningful.)
    """
    d = cfg["data"]
    pc = _preproc_from_cfg(d, augment=False, random_temporal_crop=False, sampling="crop")
    try:
        ds = RTMRIVideoDataset(d["manifest"], split="val", cfg=pc, seed=cfg["meta"]["seed"])
    except ValueError:
        return None
    return torch.utils.data.DataLoader(
        ds, batch_size=d.get("probe_batch_size", d["batch_size"]),
        shuffle=False, num_workers=d.get("num_workers", 8), collate_fn=simple_collate,
    )


def run_diagnostics(encoder, monitor_loader, device, max_batches, dtype):
    """Label-free representation-collapse metrics on held-out clips."""
    out = {}
    if monitor_loader is not None:
        fv, _ = extract_features(encoder, monitor_loader, device,
                                 max_batches=max_batches, dtype=dtype)
        out.update(feature_diagnostics(fv))
    encoder.train()
    return out


def train(cfg):
    meta, data, opt_c, loss_c, model_c = (
        cfg["meta"], cfg["data"], cfg["optimization"], cfg["loss"], cfg["model"],
    )
    folder = meta["folder"]
    os.makedirs(folder, exist_ok=True)
    seed = meta.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    which = meta.get("dtype", "float32").lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(which, torch.float32)
    mixed = dtype != torch.float32 and device.type == "cuda"
    print(f"[tssl] device={device} dtype={dtype} mixed={mixed}")

    # -- masks (re-tuned for the token grid unless given explicitly)
    patch = data.get("patch_size", 16)
    spatial_tokens = data["spatial_size"] // patch
    cfgs_mask = cfg.get("mask") or mask_config_for(spatial_tokens)
    print(f"[tssl] grid={spatial_tokens}x{spatial_tokens}, "
          f"{data['frames_per_clip'] // 2} temporal; {len(cfgs_mask)} mask(s)")

    from src.masks.multiseq_multiblock3d import MaskCollator
    from src.masks.utils import apply_masks

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask, dataset_fpcs=[data["frames_per_clip"]],
        crop_size=data["spatial_size"], patch_size=patch,
        tubelet_size=data.get("tubelet_size", 2),
    )

    ssl_ds, loader = build_ssl_loader(cfg, mask_collator)
    monitor_loader = build_monitor_loader(cfg)
    ipe = opt_c.get("ipe") or len(loader)
    epochs = opt_c["epochs"]
    # gradient accumulation: micro-batch = data.batch_size; effective batch =
    # batch_size * accum_steps. The lr/wd/EMA schedules step once per OPTIMIZER
    # update, so the schedule horizon is in optimizer-update units (oue/epoch).
    bs = data.get("batch_size", 1)
    eff = opt_c.get("effective_batch")
    accum_steps = max(1, int(opt_c.get("accum_steps") or (round(eff / bs) if eff else 1)))
    oue = max(1, ipe // accum_steps)            # optimizer updates per epoch
    print(f"[tssl] {len(ssl_ds)} train clips, ipe={ipe} micro-batches, epochs={epochs}; "
          f"batch={bs} x accum={accum_steps} -> eff_batch={bs*accum_steps}, "
          f"{oue} optimizer-updates/epoch")

    # -- model
    n_masks = len(cfgs_mask)  # one mask-token per (fpc) seq; >=1 is enough
    encoder, predictor = build_models(
        device=device, model_name=model_c.get("model_name", "vit_large"),
        spatial_size=data["spatial_size"], frames_per_clip=data["frames_per_clip"],
        patch_size=patch, tubelet_size=data.get("tubelet_size", 2),
        num_mask_tokens=n_masks,
        use_activation_checkpointing=model_c.get("use_activation_checkpointing", True),
    )
    if model_c.get("pretrained", True) and not meta.get("resume"):
        # init from the V-JEPA2 pretrained weights (skipped when resuming -- the
        # resume checkpoint below overwrites encoder/predictor anyway)
        ckpt = resolve_checkpoint(model_c.get("model_name", "vit_large"),
                                  model_c.get("checkpoint"))
        load_pretrained(encoder, predictor, ckpt,
                        model_c.get("checkpoint_key", "target_encoder"))
    target_encoder = make_target_encoder(encoder)

    # -- optimizer / schedulers (reuse repo recipe machinery)
    from app.vjepa.utils import init_opt
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=False, encoder=encoder, predictor=predictor,
        wd=float(opt_c["weight_decay"]), final_wd=float(opt_c["final_weight_decay"]),
        start_lr=opt_c["start_lr"], ref_lr=opt_c["lr"], final_lr=opt_c["final_lr"],
        iterations_per_epoch=oue, warmup=opt_c["warmup"], num_epochs=epochs,
        ipe_scale=opt_c.get("ipe_scale", 1.0), mixed_precision=mixed,
        betas=tuple(opt_c.get("betas", (0.9, 0.999))), eps=float(opt_c.get("eps", 1e-8)),
    )
    ema = opt_c.get("ema", [0.998, 1.0])
    total_steps = int(oue * epochs * opt_c.get("ipe_scale", 1.0)) + 1
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / total_steps
                          for i in range(total_steps + 1))

    loss_exp = loss_c.get("loss_exp", 1.0)
    max_steps = meta.get("max_steps")
    log_path = os.path.join(folder, "train_log.csv")
    log_f = open(log_path, "a")
    if os.path.getsize(log_path) == 0:
        log_f.write("epoch,itr,loss,lr,wd,ema_m,iter_ms\n")

    # -- resume from a checkpoint (continue training after an allocation expires).
    # Restores encoder/predictor/target/optimizer/scaler and fast-forwards the
    # lr / wd / EMA-momentum schedules to the resumed step, so training picks up
    # exactly where it stopped. Use the SAME config (epochs/ipe) so the schedule
    # horizon matches.
    start_epoch = 0
    resume_path = meta.get("resume")
    if resume_path:
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(rck["encoder"])
        predictor.load_state_dict(rck["predictor"])
        target_encoder.load_state_dict(rck["target_encoder"])
        optimizer.load_state_dict(rck["opt"])
        if scaler is not None and rck.get("scaler") is not None:
            scaler.load_state_dict(rck["scaler"])
        start_epoch = int(rck["epoch"])
        skip = start_epoch * oue                    # schedulers step per optimizer update
        for _ in range(skip):                       # fast-forward lr/wd schedules
            scheduler.step(); wd_scheduler.step()
        momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / total_steps
                              for i in range(skip, total_steps + 1))
        print(f"[tssl] RESUMED {os.path.basename(resume_path)} @ epoch {start_epoch} "
              f"({skip} opt-updates fast-forwarded); continuing to epoch {epochs}")

    global_step = start_epoch * ipe
    new_lr, new_wd, m = 0.0, 0.0, ema[0]            # last-known values for logging
    for epoch in range(start_epoch, epochs):
        encoder.train(); predictor.train()
        running = 0.0
        n_seen = 0
        for itr, sample in enumerate(loader):
            if itr >= ipe:
                break
            t0 = time.time()

            clips, masks_enc, masks_pred = [], [], []
            for fpc_sample in sample:
                udata, m_enc, m_pred = fpc_sample
                clips += [udata[0][0].to(device, non_blocking=True)]
                masks_enc += [[m.to(device, non_blocking=True) for m in m_enc]]
                masks_pred += [[m.to(device, non_blocking=True) for m in m_pred]]

            # advance lr/wd once per accumulation group (= per optimizer update)
            if itr % accum_steps == 0:
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()
            is_boundary = (itr + 1) % accum_steps == 0 or itr + 1 == ipe

            def forward_target(c):
                with torch.no_grad():
                    h = target_encoder(c)
                    return [F.layer_norm(hi, (hi.size(-1),)) for hi in h]

            def forward_context(c):
                z = encoder(c, masks_enc)
                return predictor(z, masks_enc, masks_pred)

            def loss_fn(z, h):
                h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
                loss, n = 0.0, 0
                for zi, hi in zip(z, h):
                    for zij, hij in zip(zi, hi):
                        loss = loss + torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                        n += 1
                return loss / n

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
                h = forward_target(clips)
                z = forward_context(clips)
                loss = loss_fn(z, h)

            # accumulate scaled grads; only step the optimizer on the group boundary
            loss_b = loss / accum_steps
            if mixed:
                scaler.scale(loss_b).backward()
            else:
                loss_b.backward()
            if is_boundary:
                if mixed:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # EMA momentum update of target encoder (once per optimizer update)
                m = next(momentum_scheduler)
                with torch.no_grad():
                    pk = list(target_encoder.parameters())
                    pq = list(encoder.parameters())
                    torch._foreach_mul_(pk, m)
                    torch._foreach_add_(pk, pq, alpha=1 - m)

            lval = float(loss)
            assert not np.isnan(lval), "loss is NaN (collapse / instability)"
            running += lval
            n_seen += 1
            dt = (time.time() - t0) * 1000.0
            log_f.write(f"{epoch+1},{itr},{lval:.5f},{new_lr:.3e},{new_wd:.3e},{m:.5f},{dt:.0f}\n")
            if itr % 10 == 0:
                log_f.flush()
                print(f"[e{epoch+1} {itr}/{ipe}] loss={running/n_seen:.4f} "
                      f"lr={new_lr:.2e} m={m:.4f} {dt:.0f}ms")
            global_step += 1
            if max_steps and global_step >= max_steps:
                print(f"[tssl] hit max_steps={max_steps}, stopping early")
                break

        # -- diagnostics (label-free collapse monitoring)
        if monitor_loader is not None and (epoch + 1) % meta.get("eval_freq", 1) == 0:
            diag = run_diagnostics(encoder, monitor_loader, device,
                                   max_batches=meta.get("probe_max_batches", 20),
                                   dtype=dtype)
            print(f"[e{epoch+1}] diagnostics: {diag}")
            with open(os.path.join(folder, "diagnostics.jsonl"), "a") as df:
                df.write(json.dumps({"epoch": epoch + 1, **diag}) + "\n")

        # -- checkpoint (resumable; atomic write so a mid-save kill can't corrupt it)
        if (epoch + 1) % meta.get("save_freq", 1) == 0:
            ckpt_path = os.path.join(folder, "latest.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1, "loss": running / max(1, n_seen),
            }, ckpt_path + ".tmp")
            if os.path.exists(ckpt_path):              # one-deep backup
                os.replace(ckpt_path, ckpt_path + ".prev")
            os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic install
            snap = meta.get("snapshot_freq")           # optional epoch_NN.pt history
            if snap and (epoch + 1) % snap == 0:
                shutil.copyfile(ckpt_path, os.path.join(folder, f"epoch_{epoch+1}.pt"))
        print(f"[tssl] epoch {epoch+1} avg loss {running/max(1, n_seen):.4f}")
        if max_steps and global_step >= max_steps:
            break

    log_f.close()
    print("[tssl] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override meta.max_steps (smoke testing)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path (e.g. runs/<name>/latest.pt) to continue "
                         "training from; restores weights/opt/scaler + schedules")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.setdefault("meta", {})["max_steps"] = args.max_steps
    if args.resume is not None:
        cfg.setdefault("meta", {})["resume"] = args.resume
    train(cfg)


if __name__ == "__main__":
    main()
