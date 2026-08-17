# RESULTS — AAI-JEPA (acoustic-conditioned rtMRI latent rollout)

**What it is:** a frozen T-SSL ViT-L rtMRI encoder + a trainable audio-conditioned
predictor that, given `ctx` seed latent frames, rolls out the remaining tokens of
`T'=16` from **WavLM audio only**. Self-supervised (target = the frozen encoder's
own future token embeddings); no arti-6, no phoneme labels. All runs use **WavLM
layer-9** on the combined rtMRI corpus (the layer −1 run's `audio_gap` never turned on).

**The question:** does the rollout actually *use* the audio? Headline metric
`audio_gap = val_shuf_ar_l1 − val_ar_l1` (shuffled-audio minus real-audio AR L1;
**>0 ⇒ real audio helps**). Secondary: `val_tf_l1` (teacher-forced) and `val_ar_l1`
(autoregressive) L1 in layer-normed feature space (↓). This doc sweeps the
**conditioning variant** (ctx2 vs ctx1; FiLM vs cross-attention vs state-only).

> Updated 2026-07-12. `audio_gap` stays small & positive across variants (weak but
> non-zero audio use). FiLM was **extended 20 → 50 epochs** (COMPLETED): the longer
> anneal **improved reconstruction** (val_tf_l1 0.501 → 0.450) but **did not lift
> `audio_gap`** (converged 0.0022, ≈ its ep20 value; the 0.0201 @ ep2 peak was a
> transient re-anneal artifact) — i.e. FiLM's weak audio use is **not** schedule-
> limited. `xattn` was **stopped at ep44** (of a 100-ep extension).

---

## Run sweep — headline (WavLM-L9, combined corpus, best-of / final)

| Run (variant) | epochs | val_tf_l1 ↓ | val_ar_l1 ↓ | **audio_gap** | peak audio_gap | status |
|---|---|---|---|---|---|---|
| `…_L9_ctx2` (ctx2, ac_audio) | 12/20 | 0.517 | 0.548 | 0.0017 | 0.0046 @e7 | stopped early |
| `…_L9_ctx2_stateonly` (ctx2, state-only) | 20/20 | 0.503 | 0.536 | 0.0004 | 0.0045 @e6 | done |
| `…_L9_ctx1_film` (ctx1, **FiLM**) | **50/50** | 0.450 | 0.500 | 0.0022 | **0.0201 @e2** | done (extended 20→50) |
| `…_L9_ctx1_xattn` (ctx1, **cross-attn**) | 43/100 | **0.431** | **0.485** | 0.0019 | 0.0090 @e4 | **stopped @ ep44** |

`ctx1` (1 seed frame) forces more reliance on audio than `ctx2`. `xattn` has the
best reconstruction (trained furthest), but `audio_gap` is not larger than the
others — lower AR L1 is coming from the longer schedule, not from audio use. FiLM's
large early `audio_gap` peak is the reason to extend it.

## Checkpoints & weight locations

All under `/scratch1/hongn/artijepa/runs/<run>/`. Each saves `latest.pt` **atomically
every epoch** (encoder-frozen; contains predictor + state/action heads + optimizer +
scaler + epoch) with a one-epoch-back `latest.pt.prev` backup.

| Run | latest.pt (epoch) | named snapshot | config |
|---|---|---|---|
| `…_L9_ctx2` | ep12 | — | `configs/aai_wavlm_256_combined_L9_ctx2.yaml` |
| `…_L9_ctx2_stateonly` | ep20 | — | `configs/aai_wavlm_256_combined_L9_ctx2_stateonly.yaml` |
| `…_L9_ctx1_film` | **ep50** (done) | **`epoch_20.pt`** (frozen ep20 snapshot) | `…_L9_ctx1_film.yaml` (base 20ep), `…_film_e50.yaml` (→50) |
| `…_L9_ctx1_xattn` | ep44 | — | `…_L9_ctx1_xattn.yaml` (20ep), `…_xattn_e100.yaml` (extend) |

## How to run / resume

Env: `source dev_artiJEPA/scripts/_env.sh` (conda `artijepa` + PYTHONPATH). Trainer:
`python -m artijepa.aai_train --config <yaml> [--resume <ckpt>]`.

```bash
# Fresh run (from scratch / from the frozen T-SSL encoder init in the config):
bash dev_artiJEPA/scripts/16_train_aai.sh dev_artiJEPA/configs/aai_wavlm_256_combined_L9_ctx1_film.yaml

# Extend/resume as a self-resubmitting SLURM job (survives the 48h cap):
sbatch dev_artiJEPA/scripts/resume_aai_film_e50.sbatch      # FiLM  20 -> 50  (THIS run, job 10176135)
sbatch dev_artiJEPA/scripts/resume_aai_xattn_e100.sbatch    # xattn 20 -> 100 (currently stopped)

# monitor / cancel
squeue -u $USER
tail -f /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_film/train_log.csv
scancel <jobid>          # also stops the self-resubmit chain
```

To **extend epochs**, copy the base config, bump `optimization.epochs`, keep the same
`meta.folder`, and `--resume <folder>/latest.pt` (the cosine LR re-stretches over the
new horizon). Example: `configs/aai_wavlm_256_combined_L9_ctx1_film_e50.yaml` is the
base FiLM config with `epochs: 20 → 50`, resumed from `latest.pt @ ep20`.

---

## Shared L9 base config (below tables describe the `ctx2` run; per-variant deltas: `ctx_frames`, conditioning `film`/`xattn`/state-only)

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

## Current status (2026-07-11)

Actions this session:
- **`xattn` stopped at ep44** (`scancel` of job 10145254, which was extending it 20→100). Its
  `latest.pt` @ ep44 is preserved; reconstruction is best-of-sweep (val_tf_l1 0.431) but
  `audio_gap` (0.0019) is no larger than the shorter runs — the AR-L1 gain is schedule-driven,
  not audio-driven.
- **`film` extended 20 → 50 — COMPLETED** (job 10176135 → sacct COMPLETED 2026-07-12;
  `resume_aai_film_e50.sbatch`, resumed `latest.pt` @ ep20; ep20 preserved as `epoch_20.pt`).
  **Result:** reconstruction improved cleanly (val_tf_l1 0.5007 → **0.4502**, best 0.4497 @ ep43;
  train loss 0.897 → 0.778), but `audio_gap` **did not recover** — ep21 briefly rose (0.0064 @
  ep22) from the re-anneal then settled to a converged **0.0022** (ep45–50 mean, sd 0.0004),
  ≈ the ep20 value (0.0015) and *below* the ep1–20 mean (0.0063, inflated by the ep2 spike).
  **Conclusion:** FiLM's weak audio conditioning is **not schedule-limited** — longer training
  buys latent reconstruction, not audio use. Next levers per aai_plans.md: WavLM layer sweep
  {6,12}, `wavlm-large`, or the stronger `xattn` conditioning (resume its ep44 ckpt).

**Cross-variant read (WavLM-L9):** `audio_gap` is small & positive everywhere (~0.001–0.002
at convergence), i.e. the rollout uses the audio only weakly. Best early signal = FiLM @ ep2.
Success criterion (aai_plans.md §5.4): `audio_gap` clearly positive AND real-audio AR L1 beating
the no-audio baseline — not yet decisively met. Remaining levers if FiLM-50 doesn't lift it:
WavLM layer sweep {6,12}, `wavlm-large`, stronger conditioning (already testing FiLM vs xattn).

**FiLM `…_L9_ctx1_film` — full 50-epoch trace** (val_tf_l1 / val_ar_l1 / audio_gap):
ep2 peak audio_gap **0.0201**; ep20 (end of base run) tf_l1 0.5007 / ar_l1 0.5357 / gap 0.00153;
ep22 re-anneal gap bump 0.00636; ep50 (final) **tf_l1 0.4502 / ar_l1 0.4992 / gap 0.00247**
(shuf_ar 0.5017). Converged ep45–50: tf_l1 0.4504, gap 0.00216 ± 0.0004. Full per-epoch trace in
`diagnostics.jsonl`.
