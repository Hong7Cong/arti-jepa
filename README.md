# Arti-JEPA — V-JEPA 2 on rtMRI vocal-tract video

Implementation of `Arti-JEPA-Plans.md` for the USC 75-Speaker Speech MRI corpus,
driven by `MasterPrompt.md`. This directory adds the rtMRI-specific data
engineering (Part A) and the **domain-adaptive self-supervised pre-training**
track (**T-SSL**, Part B) on top of the parent V-JEPA 2 repo — it reuses the
repo's encoder/predictor/mask machinery and never modifies it. Make sure to save/update progress in `TODO.md` files to keep track of progresses and todo things.

> **Pinned scope:** the 75-speaker corpus ships *no* dense labels, so the headline
> track is **T-SSL** — continue the V-JEPA mask-denoising objective on the
> unlabeled rtMRI, initialised from V-JEPA 2 ViT-L, with **label-free**
> representation-collapse monitoring. The downstream eval is **phoneme prediction**
> (κ + PER): Task 1 = pseudo phonemes from an audio model on the paired audio;
> Task 2 = gold phonemes for an OOD speaker (`usc_lss`). See `Arti-JEPA-Plans.md`
> B.4 / Part C and `TODO.md`. (The old weak stimulus-group probe was removed.)

## Environment

The stock `vjepa2-312` env ships `torch 2.12+cu130`, which **cannot drive this
node's Tesla V100** (driver = CUDA 12.9). Build the V100-compatible env once:

```bash
bash dev_artiJEPA/setup_env.sh          # creates conda env `artijepa` (torch 2.6+cu124)
```

Everything below assumes the repo root is on `PYTHONPATH` together with this dir
(handled by `scripts/_env.sh`):

```bash
source dev_artiJEPA/scripts/_env.sh      # activates artijepa, sets PYTHONPATH, cds to repo root
```

## Data layout

`/scratch1/hongn/speaker75/subNNN/2drt/video/*_video.mp4` — 75 subjects, 2,371
clips, **84×84 @ 83.28 fps**, mpeg4 grayscale, ~40 s each, with a paired audio
stream. `metafile_public_*.json` carries demographics + the per-clip task list.
All generated artifacts go under `/scratch1/hongn/artijepa/` (never `/project2`).

## Pipeline

```bash
# Phase 0 — data engineering (manifest -> subject-disjoint splits -> grayscale stats)
bash dev_artiJEPA/scripts/01_prepare_data.sh        # ~few min (decord probe + 300-clip stats)

# Fast end-to-end validation (CPU/GPU, ViT-tiny, 2 steps)
bash dev_artiJEPA/scripts/02_smoke.sh

# T-SSL — domain-adaptive pre-training (primary = ViT-L @ 256px / 32f)
bash dev_artiJEPA/scripts/03_train_tssl.sh                                   # primary
bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_128.yaml   # 128px ablation
bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_256.yaml --max-steps 5  # quick check
```

### Resume training from a checkpoint

`latest.pt` is saved **every epoch** (`save_freq`), written **atomically**
(`latest.pt.tmp` → renamed in, with the prior one kept as `latest.pt.prev`) and
carries encoder / predictor / target-encoder / optimizer / GradScaler + the epoch
number. So if a SLURM allocation expires mid-run, **continue from where it
stopped** with `--resume <checkpoint path>`:

```bash
# in a fresh allocation
source dev_artiJEPA/scripts/_env.sh
bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_256.yaml \
     --resume /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt
```

Resume restores all weights + optimizer + scaler and **fast-forwards** the
lr / wd / EMA-momentum schedules to the saved step, so training picks up exactly
where it left off (no re-warmup, no restart from the pretrained init). **Use the
same config** (`epochs` / `ipe`) as the original run so the schedule horizon
matches. Equivalent to the flag, you can set `meta.resume: <path>` in the YAML.
Optional `meta.snapshot_freq: N` also keeps an `epoch_NN.pt` history alongside the
rolling `latest.pt`. (If a hard kill ever lands during the rename, recover with
`mv latest.pt.tmp latest.pt`, or fall back to `latest.pt.prev`.)

## What maps to the plan

| Plan | Where |
|---|---|
| A.1 intensity norm (percentile clip + per-clip z-score, rtMRI stats) | `rtmri_dataset.py::_intensity_norm`, `compute_stats.py` |
| A.2 temporal 83.28→`target_fps` (default 25) **linear interpolation in the dataloader** | `rtmri_dataset.py::_sample_source_indices` / `_load_clip` |
| A.2 sampling: `crop` (one window/video) or **`tile`** (whole video @`target_fps` split into `frames_per_clip` chunks → full coverage) | `rtmri_dataset.py::_build_tile_index` / `_tile_indices` |
| A.3 spatial 84→256/128 (bicubic) or →96 (reflect-pad) | `rtmri_dataset.py::_spatial` |
| A.4 grayscale→3-channel replicate | `rtmri_dataset.py::_load_clip` |
| A.6 even-length clips, random temporal crop | `PreprocConfig`, `_sample_source_indices` |
| A.7 anatomically-safe aug (no hflip/rotation/colour) | `rtmri_dataset.py::_augment` |
| A.8 subject-disjoint splits | `splits.py` |
| A.9 config template | `configs/preprocess.yaml` |
| B.1 ViT-L backbone, pretrained init | `model.py`, `checkpoint.py` |
| B.3 T-SSL (EMA target, L1 feature loss, multiblock masks **re-tuned per grid**) | `tssl_train.py`, `masking.py` |
| B.3/F **label-free** representation-collapse monitoring | `collapse.py` |
| B.4/C phoneme prediction eval (κ + PER), gold OOD + pseudo | `eval_phoneme.py`, `phonemes.py`, `usc_lss.py`, `audio_phoneme.py` |
| B.4 probe heads: `linear`/`mlp`/`tcn`/`lstm`/`transformer` (mean-pool S') + **`tcn_spatial`/`attentive` (un-pooled `[B,T',S',D]` grid)** | `eval_phoneme.TokenProbe`, `scripts/06_probe_sweep.sh` / `07_probe_spatial.sh` |
| D ablations (resolution / clip length / masks / probe / with-vs-without T-SSL) | per-config YAMLs + `masking.mask_config_for` |

## Outputs

Per run under `meta.folder` (default `/scratch1/hongn/artijepa/runs/<name>`):
`train_log.csv` (loss/lr/wd/ema/step-time), `diagnostics.jsonl`
(`feature_std`, `effective_rank`, `mean_abs_cosine` per epoch — label-free, watch
these for collapse), and `latest.pt` (encoder/predictor/target/opt/scaler/epoch —
**resumable**, saved atomically each epoch with a `latest.pt.prev` backup; see
[Resume training](#resume-training-from-a-checkpoint)). Phoneme-eval results land
in `…/eval/phoneme_*.json`; cached features in `…/feat_cache/phoneme/`.

## Temporal sampling (`crop` vs `tile`)

The T-SSL configs default to **25 fps + `sampling: tile`**: each video is
resampled onto a uniform 25 Hz grid and cut into consecutive, non-overlapping
`frames_per_clip` (32-frame = 1.28 s) chunks, every chunk an independent training
clip. This gives **full temporal coverage** (mean 23.7 chunks/video → 42,415 train
clips) instead of one random 32-frame window per video (`sampling: crop`). Tile
mode reads `n_frames`/`fps` from the manifest, so build it with
`build_manifest --probe`. The phoneme eval also tiles (per-token labels reassembled
per utterance for PER); the collapse-monitor loader stays on `crop`.

Because one tile epoch is ~10k batches, the configs set a fixed
`optimization.ipe` (500) so the LR warmup/cosine schedule and the per-epoch
diagnostics/checkpoint cadence stay meaningful; bound wall-clock with `--max-steps`.

## Notes / next steps

- **fp16 vs bf16:** the V100 (Volta) has no bf16 tensor cores → T-SSL configs use
  `dtype: float16` + GradScaler; on the L40S/Ampere+ switch to `bfloat16` (the
  phoneme eval already uses bf16).
- Effective batch: bump `batch_size` + add gradient accumulation for the 256px run;
  128px already fits a larger batch.
- **Headline (UPDATED 2026-06-10, gold OOD speaker, 128px): use the SPATIAL probe.**
  The default probe mean-pools the spatial tokens — that discards *where* in the
  vocal tract the signal sits, which is the phonetic information. With the
  spatial-aware `attentive` probe (un-pooled `[B,T',S',D]` grid), T-SSL gives
  **test κ 0.327→0.475** (pretrained→+T-SSL, +0.148) — vs only 0.222→0.247 (+0.025)
  with the mean-pool `tcn`. **tssl_128 (κ 0.475) beats the best image baseline**
  (supervised ViT-L/16, 0.368) at the *same* 128px, so the earlier "baselines >
  V-JEPA" was a probe artifact, not a resolution gap. Reproduce:
  `bash scripts/07_probe_spatial.sh`. The 128px T-SSL run was collapse-free
  (effective_rank 79→102).
- **256px T-SSL is now RUNNING** on the V100 at **batch 32** (~0.39 s/clip, ~87 h /
  25k micro-steps, resumable each epoch; ~24.5/32 GB VRAM). **Activation checkpointing
  is mandatory at 256px** (no-ckpt OOMs even at bs8); bs32+ckpt is the speed/VRAM sweet
  spot (bs64 OOMs), and keeps `eff_batch=64`/`oue=250` so resume is unchanged. Launch with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Resume with
  `--resume runs/tssl_vitl_256/latest.pt`; eval intermediate checkpoints with
  `scripts/07_probe_spatial.sh dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml`.
- **Task-1 caveat:** the audio CTC labeler needs a transformers-compatible env
  (this env's transformers 5.x wants torch≥2.7); run `build_pseudo_labels` there,
  then eval reads the cached `.npy`. See `TODO.md`.
- Dense heads (segmentation/landmarks/inversion) deferred until labels exist;
  `usc_lss` ships tongue-contours/SAM-seg/kinematics for that later — Plans B.4.
```
