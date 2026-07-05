# RESULTS — AAI-JEPA (acoustic-conditioned rtMRI latent rollout)

**Run:** `aai_wavlm_256_combined_L9_ctx2`
**Config:** `configs/aai_wavlm_256_combined_L9_ctx2.yaml`
**Run folder:** `/scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx2`
**What it is:** a frozen T-SSL ViT-L rtMRI encoder + a trainable audio-conditioned predictor that,
given `ctx=2` seed latent frames, rolls out the remaining 14 of `T'=16` temporal tokens from
**WavLM audio only**. Self-supervised (target = the frozen encoder's own future token embeddings);
no arti-6, no phoneme labels. This is the **WavLM layer-9** variant (the layer −1 run's `audio_gap`
never turned on).

> Snapshot written 2026-07-05, 09:34 PDT — run in progress at **epoch 12 / 20** (11 epochs logged).
> See "Current status" at the bottom; update as it progresses.

---

## Predictor training settings

| Setting | Value |
|---|---|
| Objective | `temporal` — acoustic-conditioned autoregressive latent rollout |
| `ctx_frames` | **2** (seed 2 real latent frames, autoregress frames 2..15 from audio only) |
| Predictor kind | `ac_audio` (`AudioConditionedPredictor`), state=`e[t]`, action=`e[t+1]−e[t]` |
| Predictor depth / dim / heads | 12 / 384 / 12 |
| RoPE / frame-causal | true / true |
| Trainable params | **~23.0M** (predictor + `Linear(768→384)` state/action heads) |
| Epochs | **20** (full cosine anneal by ep20) |
| Iters/epoch (`ipe`) | 1000 |
| Effective batch | 128 (micro `batch_size=3` → grad-accum ~43) |
| LR schedule | `start_lr` 1e-4 → `lr` 5e-4 (warmup 2 ep) → cosine → `final_lr` 1e-5 |
| Weight decay | 0.04 (constant, `final_weight_decay` 0.04) |
| Optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, `ipe_scale` 1.0 |
| Loss | `rollout_l1` (teacher-forced + AR branches) in layer-normed feature space, masked by `valid`, `loss_exp` 1.0 |
| Precision | float16 + GradScaler (V100 / Volta) |
| Hardware | single **V100-PCIE-32GB** (node d14-10); ~25.3 GB used, ~3.7 s/step, ~1 h/epoch |
| eval / save freq | every epoch; `probe_max_batches` 8 (audio_gap diagnostic) |

## Dataset

| | |
|---|---|
| Manifest | `/scratch1/hongn/artijepa/manifest_combined_aai.csv` |
| Corpus | **combined rtMRI** = rtMRI-75 (`speaker75`) + longitudinal `.avi` — same pool T-SSL pretrained on; audio embedded in every video |
| Split | **speaker-disjoint** (`make_aai_split.py`): **7,812 train / 477 val / 1,192 test** (9,481 clips) |
| Labels | none — self-supervised; target = frozen encoder's own future token embeddings |
| Spatial | 256 px, `resize`, `intensity_norm=zscore`, grayscale_stats `grayscale_stats_combined.json` |
| Temporal | `frames_per_clip=32`, `target_fps=50.0`, `tubelet_size=2` → **T'=16 tokens**; `patch_size=16` → 256 tokens/frame |
| Sampling | `tile` (one (clip,chunk) per item; loader decodes video via decord) |
| Augment | true |
| Loader | `num_workers=4`, `pin_mem=false`, `persistent_workers=false` (under `--mem` cgroup) |

## Audio config (conditioning signal)

| | |
|---|---|
| Model | `microsoft/wavlm-base-plus` (frozen; features cached offline) |
| **Layer** | **9** (mid-layer; articulatory content peaks ~L9–11 vs the flat last-layer run) |
| Dim (A) | 768 |
| Cache | `/scratch1/hongn/artijepa/audio_feats/wavlm_base_plus_L9` (`.npy` + `meta.json`) |
| Native rate | 50.187 Hz (16 kHz mono input) → mean-pooled onto the T'=16 token grid |
| Pool / normalize | `mean` / per-dim `zscore` (train-split stats, over 11.2M frames) |
| speaker / online | false / false (cached, never runs at train time) |

## Frozen T-SSL JEPA config (encoder / init checkpoint)

| | |
|---|---|
| Checkpoint | `/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_60.pt` (epoch-60 snapshot, 5.1 GB) |
| Checkpoint key | `target_encoder` (loaded 292 tensors, 0 missing, 0 skipped) |
| Architecture | **ViT-L** (`vit_large`), 303.9M params — **frozen** |
| Pretraining | domain-adapted T-SSL on the same combined rtMRI corpus, 256 px / 50 fps |
| Role | supplies both the 2 seed latent frames AND the self-supervised rollout target `h` |
| Activation checkpointing | off (fastest; bs3/ctx=2 ~22.7–25 GB fits the 32 GB V100) |
| WavLM at train time | never runs (features cached) |

---

## Current status (2026-07-05, epoch 12 / 20)

Reconstruction improving cleanly; `audio_gap` (shuffled−real AR L1 = "does the rollout use the
audio") stays positive but small and noisy, peaking at epoch 7.

| epoch | val_tf_l1 | val_ar_l1 | audio_gap |
|---|---|---|---|
| 1 | 0.7449 | 0.7502 | 0.0000018 |
| 2 | 0.7078 | 0.7326 | 0.000037 |
| 3 | 0.6532 | 0.6740 | 0.00070 |
| 4 | 0.6261 | 0.6527 | 0.00224 |
| 5 | 0.6057 | 0.6340 | 0.00151 |
| 6 | 0.5840 | 0.6079 | 0.00187 |
| 7 | 0.5696 | 0.5932 | **0.00455** ← peak |
| 8 | 0.5517 | 0.5774 | 0.00321 |
| 9 | 0.5368 | 0.5667 | 0.00141 |
| 10 | 0.5267 | 0.5575 | 0.00105 |
| 11 | 0.5203 | 0.5513 | 0.00236 |

**Note:** run was interrupted at ep6 (GPU-node disconnect) and resumed from the ep5 `latest.pt`;
per-epoch checkpointing means ep6 restarted from step 0. Success criterion (aai_plans.md §5.4):
`audio_gap` clearly positive and above the shuffled control, and real-audio AR L1 beating the
no-audio baseline — the audio_gap is positive but not yet growing with the loss; watch ep16–20
under the LR anneal. Planned levers if it fades: `ctx=1`, WavLM layer sweep {6,12}, `wavlm-large`.
