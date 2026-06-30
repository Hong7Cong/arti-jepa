"""Multi-GPU (DDP) Acoustic-Conditioned JEPA training (plans_aucjepa.md §6).

An *independent* launcher that trains the SAME model as ``aucjepa_train.py``
(frozen T-SSL ViT-L encoder + trainable audio-conditioned predictor) but across
N GPUs with ``torch.nn.parallel.DistributedDataParallel`` -- one process per GPU,
spawned by ``torchrun``. DDP is *data*-parallel: the frozen encoder is replicated
(no grad / no sync) on every rank and ONLY the AC predictor is DDP-wrapped and
gradient-synced.

Schedule semantics are preserved exactly. ``effective_batch`` is the GLOBAL
optimizer batch::

    effective_batch = batch_size (per GPU) * world_size * accum_steps

so the trainer derives ``accum_steps = effective_batch / (batch_size * world_size)``
and the per-update LR/WD trajectory is identical to the single-GPU run -- DDP just
parallelises the micro-batches for ~world_size x throughput. Checkpoints store the
*unwrapped* predictor state, so a DDP ``latest.pt`` resumes 1:1 under the
single-GPU ``aucjepa_train.py`` and vice versa.

Run (auto-detects GPUs):
    bash dev_artiJEPA/scripts/10_train_aucjepa_ddp.sh dev_artiJEPA/configs/aucjepa_vitl_256_ddp.yaml
or directly:
    torchrun --standalone --nproc_per_node=2 -m artijepa.aucjepa_train_ddp \
        --config dev_artiJEPA/configs/aucjepa_vitl_256_ddp.yaml
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from artijepa.audio_cond import AudioConditionedPredictor, rollout_l1, to_state_action
from artijepa.aucjepa_dataset import RTMRIAudioDataset, collate
# Reuse the single-GPU builders VERBATIM -- the model / optimizer / diagnostics are
# identical; only the training *loop* is re-implemented for DDP below.
from artijepa.aucjepa_train import (
    build_audio_loader,
    build_frozen_encoder,
    init_opt_predictor_only,
    run_diagnostics,
)
from artijepa.tssl_train import _preproc_from_cfg, load_config


def _ddp_env():
    """torchrun-provided rendezvous coordinates (defaults => single process)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


class _DDPPredictor(nn.Module):
    """Routes ``DDP.forward()`` -> ``AC predictor.forward_predictions`` so DDP's
    gradient-reduction hooks arm every iteration (DDP only instruments ``forward``).

    The trainable AC predictor is held as ``.pred``; checkpoints store
    ``.pred.state_dict()`` (NOT this wrapper's) so they stay interchangeable with
    the single-GPU ``aucjepa_train.py`` (keys like ``backbone.*``, no prefix).
    """

    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, h, states, actions, auto_steps, ctx_frames):
        return self.pred.forward_predictions(h, states, actions, auto_steps, ctx_frames)


def build_distributed_loader(cfg, audio_dir, rank, world_size):
    """Train loader sharded across ranks by ``DistributedSampler`` (disjoint, reshuffled
    per epoch via ``set_epoch``). Same dataset/preproc as the single-GPU path."""
    d = cfg["data"]
    pc = _preproc_from_cfg(d, augment=d.get("augment", True),
                           random_temporal_crop=False, sampling="tile")
    ds = RTMRIAudioDataset(d["manifest"], split="train", cfg=pc, audio_dir=audio_dir,
                           seed=cfg["meta"]["seed"],
                           normalize=cfg.get("audio", {}).get("normalize", "zscore"))
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    nw = d.get("num_workers", 2)
    loader = DataLoader(
        ds, batch_size=d["batch_size"], sampler=sampler, drop_last=True,
        num_workers=nw, pin_memory=d.get("pin_mem", False),
        persistent_workers=bool(d.get("persistent_workers", False)) and nw > 0,
        collate_fn=collate)
    return ds, loader, sampler


def train(cfg):
    rank, local_rank, world_size = _ddp_env()
    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # bind the device at init so collectives (incl. barrier) know the rank->GPU map
    # (silences the "devices currently unknown" warning + its potential-hang note)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)

    meta, data, opt_c, loss_c, model_c = (
        cfg["meta"], cfg["data"], cfg["optimization"], cfg["loss"], cfg["model"])
    audio_c, pred_c = cfg.get("audio", {}), cfg.get("predictor", {})
    folder = meta["folder"]
    if is_main:
        os.makedirs(folder, exist_ok=True)
    dist.barrier()

    seed = meta.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)            # identical predictor init on every rank
    which = meta.get("dtype", "float32").lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(which, torch.float32)
    mixed = dtype != torch.float32 and device.type == "cuda"

    audio_dir = audio_c["cache_dir"]
    A = int(audio_c.get("dim", 768))
    auto_steps = int(opt_c.get("auto_steps", 2))
    ctx_frames = cfg.get("ctx_frames")            # context-prefix AR rollout (plan §4)
    objective = cfg.get("objective", "temporal")
    assert objective == "temporal", \
        f"objective={objective!r} not implemented; only 'temporal' (masked is a TODO)"

    # -- data (train sharded; val monitor only on rank 0)
    ssl_ds, loader, sampler = build_distributed_loader(cfg, audio_dir, rank, world_size)
    monitor_loader = None
    if is_main:
        try:
            _, monitor_loader = build_audio_loader(cfg, "val", audio_dir,
                                                   shuffle=False, drop_last=False)
        except (ValueError, KeyError):
            monitor_loader = None

    ipe = opt_c.get("ipe") or len(loader)
    epochs = opt_c["epochs"]
    bs = data.get("batch_size", 1)
    eff = opt_c.get("effective_batch")
    # accum so that batch_size * world_size * accum == effective_batch (global)
    accum_steps = max(1, int(opt_c.get("accum_steps")
                             or (round(eff / (bs * world_size)) if eff else 1)))
    oue = max(1, ipe // accum_steps)
    if is_main:
        print(f"[aucjepa-ddp] world_size={world_size} device={device} dtype={dtype} "
              f"mixed={mixed}", flush=True)
        print(f"[aucjepa-ddp] {len(ssl_ds)} train clips, ipe={ipe}/rank, epochs={epochs}; "
              f"batch={bs}/gpu x world={world_size} x accum={accum_steps} -> "
              f"eff_batch={bs*world_size*accum_steps}, {oue} oue/epoch", flush=True)

    # -- frozen encoder: build SEQUENTIALLY across ranks. torch.load pulls the ~5 GB
    #    ckpt to CPU; N ranks loading at once would N x the host-RAM spike and trip
    #    the job's --mem cgroup (the cap that OOM-killed the 256 single-GPU run).
    encoder = None
    for r in range(world_size):
        if rank == r:
            encoder = build_frozen_encoder(cfg, device)
        dist.barrier()

    # -- trainable AC predictor (unwrapped == ``core``; DDP-wrapped == ``predictor``)
    core = AudioConditionedPredictor(
        img_size=data["spatial_size"], patch_size=data.get("patch_size", 16),
        num_frames=data["frames_per_clip"], tubelet_size=data.get("tubelet_size", 2),
        embed_dim=encoder.embed_dim, action_embed_dim=A,
        pred_embed_dim=pred_c.get("pred_embed_dim", 384),
        depth=pred_c.get("pred_depth", 12), num_heads=pred_c.get("pred_num_heads", 12),
        use_rope=pred_c.get("use_rope", True), frame_causal=pred_c.get("frame_causal", True),
        use_activation_checkpointing=model_c.get("use_activation_checkpointing", False),
    ).to(device)
    hw = core.tokens_per_frame

    # optimizer over the unwrapped predictor's params (clean names; same tensors DDP
    # will sync). Built BEFORE resume so resume can repopulate its state.
    optimizer, scaler, scheduler, wd_scheduler = init_opt_predictor_only(
        core, iterations_per_epoch=oue, start_lr=opt_c["start_lr"],
        ref_lr=opt_c["lr"], warmup=opt_c["warmup"], num_epochs=epochs,
        wd=float(opt_c["weight_decay"]), final_wd=float(opt_c["final_weight_decay"]),
        final_lr=opt_c["final_lr"], mixed_precision=mixed,
        ipe_scale=opt_c.get("ipe_scale", 1.0),
        betas=tuple(opt_c.get("betas", (0.9, 0.999))), eps=float(opt_c.get("eps", 1e-8)))

    # -- resume (predictor + opt + scaler + epoch; every rank loads the SAME file
    #    -> identical weights, then the DDP ctor broadcasts rank-0 to be doubly sure)
    start_epoch = 0
    resume_path = meta.get("resume")
    if resume_path and os.path.exists(resume_path):
        rck = torch.load(resume_path, map_location=device, weights_only=False)
        core.load_state_dict(rck["predictor"])
        optimizer.load_state_dict(rck["opt"])
        if scaler is not None and rck.get("scaler") is not None:
            scaler.load_state_dict(rck["scaler"])
        start_epoch = int(rck["epoch"])
        for _ in range(start_epoch * oue):
            scheduler.step(); wd_scheduler.step()
        if is_main:
            print(f"[aucjepa-ddp] RESUMED {os.path.basename(resume_path)} @ epoch "
                  f"{start_epoch} ({start_epoch*oue} opt-updates fast-forwarded)", flush=True)

    # DDP-wrap AFTER (any) resume so the constructor broadcasts the loaded rank-0
    # weights to all ranks. static_graph=True is REQUIRED here: the AC predictor's
    # backbone is invoked 9x/step (1 teacher-forced + 8 AR-rollout calls) and it
    # always constructs an `extrinsics_encoder` that is UNUSED when use_extrinsics is
    # False -> a plain reducer raises "parameters not used in producing loss" at the
    # first all-reduce. static_graph records the (deterministic, tile-sampled) graph
    # once: it tolerates unused params + repeated module calls + non-reentrant
    # activation ckpt, and keeps `extrinsics_encoder` trainable so the optimizer param
    # groups still match the single-GPU trainer (checkpoints stay interchangeable).
    predictor = DDP(_DDPPredictor(core).to(device), device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False,
                    static_graph=True)
    if is_main:
        n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
        n_enc = sum(p.numel() for p in encoder.parameters())
        print(f"[aucjepa-ddp] trainable predictor {n_train/1e6:.1f}M; frozen encoder "
              f"{n_enc/1e6:.1f}M; tokens/frame={hw}, A={A}, auto_steps={auto_steps}, "
              f"ctx_frames={ctx_frames}, act_ckpt={model_c.get('use_activation_checkpointing', False)}",
              flush=True)

    loss_exp = loss_c.get("loss_exp", 1.0)
    max_steps = meta.get("max_steps")

    log_f = None
    if is_main:
        log_path = os.path.join(folder, "train_log.csv")
        log_f = open(log_path, "a")
        if os.path.getsize(log_path) == 0:
            log_f.write("epoch,itr,loss,jloss,sloss,lr,wd,iter_ms\n")

    global_step = start_epoch * ipe
    new_lr, new_wd = 0.0, 0.0
    for epoch in range(start_epoch, epochs):
        predictor.train()
        sampler.set_epoch(epoch)                  # reshuffle the shard each epoch
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
            # No DDP no_sync() grad-accum: it does not compose with static_graph=True.
            # We all-reduce every micro-step instead; grads still accumulate in .grad
            # across the window and the optimizer steps only on the boundary. The extra
            # reduces are negligible (compute-bound; 23M-param predictor).
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
                with torch.no_grad():                      # frozen perceptual target
                    h = encoder.backbone(clips)            # [B, T'*HW, D]
                    h = F.layer_norm(h, (h.size(-1),))
                state, action = to_state_action(audio)     # [B,T',A], [B,T'-1,A]
                z_tf, z_ar, ar0 = predictor(h, state, action, auto_steps, ctx_frames)
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
            if is_main:
                log_f.write(f"{epoch+1},{itr},{lval:.5f},{jl:.5f},{sl:.5f},"
                            f"{new_lr:.3e},{new_wd:.3e},{dt:.0f}\n")
                if itr % 10 == 0:
                    log_f.flush()
                    print(f"[e{epoch+1} {itr}/{ipe}] loss={running/n_seen:.4f} "
                          f"(j={rjl/n_seen:.4f} s={rsl/n_seen:.4f}) lr={new_lr:.2e} "
                          f"{dt:.0f}ms", flush=True)
            global_step += 1
            if max_steps and global_step >= max_steps:
                if is_main:
                    print(f"[aucjepa-ddp] hit max_steps={max_steps}, stopping early", flush=True)
                break

        # -- diagnostics (rank 0 only; no collectives -> safe while others wait below)
        if is_main and monitor_loader is not None and (epoch + 1) % meta.get("eval_freq", 1) == 0:
            diag = run_diagnostics(encoder, core, monitor_loader, device, dtype, mixed,
                                   auto_steps, hw, ctx_frames=ctx_frames,
                                   max_batches=meta.get("probe_max_batches", 8))
            print(f"[e{epoch+1}] diagnostics: {diag}", flush=True)
            with open(os.path.join(folder, "diagnostics.jsonl"), "a") as df:
                df.write(json.dumps({"epoch": epoch + 1, **diag}) + "\n")

        # -- checkpoint (rank 0; atomic; unwrapped predictor -> single-GPU compatible)
        if is_main and (epoch + 1) % meta.get("save_freq", 1) == 0:
            ckpt_path = os.path.join(folder, "latest.pt")
            torch.save({
                "predictor": core.state_dict(),
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
        if is_main:
            print(f"[aucjepa-ddp] epoch {epoch+1} avg loss {running/max(1,n_seen):.4f}", flush=True)

        dist.barrier()                              # keep ranks in lockstep across rank-0 diag/save
        if max_steps and global_step >= max_steps:
            break

    if is_main and log_f is not None:
        log_f.close()
    if is_main:
        print("[aucjepa-ddp] done.", flush=True)
    dist.destroy_process_group()


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
