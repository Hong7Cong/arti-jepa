# Arti-JEPA — Master (file map & setup)

## Initial settings
- **Env:** conda `artijepa` (torch 2.6+cu124, V100/L40S-compatible) — activate via
  `source dev_artiJEPA/scripts/_env.sh`. (Stock `vjepa2-312` ships torch 2.12+cu130
  which can't drive this node's GPU.)
- **Code:** develop in `/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA` ONLY. The
  parent `/project2/shrikann_35/hongn/vjepa2` is the read-only V-JEPA2 reference.
- **Data:** 75-speaker rtMRI video (unlabeled, T-SSL) at `/scratch1/hongn/speaker75`;
  longitudinal corpus at `/project2/shrikann_35/kevinyhu/data/longitudinal`; OOD gold
  speaker at `/scratch1/hongn/usc_lss`. All run artifacts under
  `/scratch1/hongn/artijepa/` (never `/project2`).
- **Downstream eval = PHONEME PREDICTION** (old stimulus-group probe dropped). Frozen
  encoder features → small per-token classifier; metrics = frame-level Cohen's κ + PER.
  "With vs without T-SSL" lift is the headline. Task 1 = pseudo phonemes (audio model on
  75-spk paired audio); Task 2 = gold phonemes for OOD `usc_lss` speaker. Alignment in
  SECONDS so any fps works (clips → 25 fps, audio ~50 Hz, 4 audio units ≈ 1 JEPA token).

## Documentation files (read these first)
| file | what it holds |
|---|---|
| `Master.md` | **this file** — setup + which-file-does-what map |
| `RESULTS.md` | **all results** — phoneme κ/PER tables, per-model train/eval settings, headlines, ablations |
| `RUNME.md` | **reproduce `RESULTS.md`** — inference-only commands on the frozen saved checkpoints |
| `TODO_pretraining.md` | plans/status for **T-SSL pretraining** (128/256/combined runs) |
| `TODO_eval.md` | plans to **evaluate** our + public models on all tasks (TBA where pending) |
| `Arti-JEPA-Plans.md` | original full project plan (Phases A–D) |
| `README.md` | repo overview |
| `TODO_acjepa.md`, `aucjepa_plans_new.md` | **separate AC-JEPA track** (articulator-conditioned JEPA + planning) |

## Code — `artijepa/`
**Data engineering**
- `build_manifest.py` — 75-speaker manifest builder.
- `build_manifest_longitudinal.py` — longitudinal `.avi` manifest (parses nframes/tRes
  from clip name, decord-probes residual).
- `merge_manifests.py` — column-reconciling concat + subject-leakage guard → `manifest_combined.csv`.
- `splits.py` — subject-disjoint train/val/test splits.
- `compute_stats.py` — grayscale mean/std stats.
- `rtmri_dataset.py` — core rtMRI video dataset (decord resample, tile/crop, aug; per-row res/fps-agnostic).
- `masking.py` — multiblock masks per token grid.
- `labels.py` — label utilities.

**T-SSL pretraining**
- `tssl_train.py` — T-SSL trainer: EMA target encoder, L1 feature loss, grad-accum, resume, collapse diagnostics.
- `model.py` — V-JEPA2 ViT-L encoder wrapper / loading.
- `checkpoint.py` — atomic save/restore (`.prev` backup, scaler+schedule).
- `collapse.py` — label-free collapse metrics (feature_std, effective_rank, mean_abs_cosine).

**Phoneme evaluation**
- `eval_phoneme.py` — freeze encoder → per-token features → probe (linear/mlp/tcn/lstm/transformer/tcn_spatial/attentive, CE/CTC) → κ + PER. `--seed`, `--tag`, `--encoder`, `--probe`, `--loss`.
- `phonemes.py` — 41-ARPABET inventory, seconds-based alignment, CTC-collapse, PER (edit dist), Cohen's κ.
- `usc_lss.py` — OOD gold manifest builder + dataset (104×104→resize, 99→25 fps, per-token gold labels, tile+pad, PER reassembly).
- `audio_phoneme.py` — Task-1 pseudo pipeline (ffmpeg 16 kHz → wav2vec2 CTC → `.npy`; `PseudoPhonemeDataset`). Model step blocked by env → decoupled.
- `baselines.py` — public image models (timm: clip/siglip/dinov2/vitl/resnet) → tubelet-pool to V-JEPA grid → same probe; emits spatial patch grid when `pool_spatial=False`.
- `videomae_baseline.py` — VideoMAE-L video baseline (transformers). Frozen `.backbone(clip)` (drop-in for the extractor) + fine-tunable `VideoMAEClassifier` (encoder+attentive/mean head).

**Disfluency evaluation (Task 8 — stuttering)**
- `stutter.py` — TextGrid parser + disfluency-type canonicalizer (typo/compound→bucket5) + manifest builder (disfluent events + fluent negatives, speaker col for LOSO) + segment dataset (uniform-sample clip per `[xmin,xmax]`) + macro-F1/balanced-acc/confusion metrics.
- `eval_disfluency.py` — segment classification: freeze encoder (V-JEPA/T-SSL / image_baseline / videomae) → cached per-segment features → attentive `SegmentProbe` → LOSO macro-F1; `--mode finetune` trains VideoMAE end-to-end. Canonical = attentive @ 256px.

**AC-JEPA track (separate — articulator-conditioned JEPA + planning)**
- `arti_cache.py` — offline articulator (+MRI frame) cache from `usc_lss/articulators/*.mat` (arti-6, 100 Hz, frame-exact).
- `arti_cond.py` — articulator conditioning (6-D arti-6 vector) wrapping the predictor.
- `acjepa_dataset.py` — rtMRI clips + frame-aligned arti-6 from the same cached session.
- `acjepa_train.py` / `acjepa_train_ddp.py` — train arti-conditioned predictor (frozen encoder); single-GPU / DDP launchers.
- `acjepa_energy.py` — energy functions bridging continuous latent ↔ discrete phoneme goal.
- `acjepa_plan.py` — CEM / receding-horizon planner (inference).

## Configs — `configs/`
- T-SSL: `tssl_vitl_128.yaml`, `tssl_vitl_256.yaml`, `tssl_vitl_256_combined.yaml` (+longitudinal).
- Eval: `eval_phoneme_usc_lss.yaml` (128), `eval_phoneme_usc_lss_256.yaml`, `eval_phoneme_usc_lss_baseline.yaml`, `eval_phoneme_pseudo.yaml`, `eval_disfluency.yaml` (Task 8, attentive @ 256px).
- Misc: `preprocess.yaml`, `smoke.yaml`.
- AC-JEPA: `acjepa_arti6_128.yaml`, `acjepa_arti6_256.yaml`, `acjepa_arti6_256_ddp.yaml`, `acjepa_plan_256.yaml`.

## Scripts — `scripts/`
- `_env.sh` — env activation (source this first).
- `01_prepare_data.sh`, `01b_prepare_longitudinal.sh` — data prep.
- `02_smoke.sh` — smoke test.
- `03_train_tssl.sh` — T-SSL training (`--resume <ckpt>`, `--max-steps`).
- `04_eval_phoneme.sh` — phoneme eval (`--encoder`, `--tag`, `--probe`).
- `05_eval_baselines.sh` — public-model baselines.
- `15_build_stutter_manifest.sh` — build the disfluency manifest (`--fluent-per-file N`).
- `16_eval_disfluency.sh` — disfluency-type eval (`--encoder`/`--model videomae`/`--mode finetune`).
- `06_probe_sweep.sh` — head×loss×encoder grid (cached features).
- `07_probe_spatial.sh` — spatial-aware probes (tcn_spatial, attentive).
- `11_build_arti_cache.sh`, `12_train_acjepa.sh`, `13_train_acjepa_ddp.sh`, `14_plan_acjepa.sh` — AC-JEPA track.

## Tests — `tests/`
- `test_smoke.py`, `test_longitudinal_smoke.py`, `test_acjepa_smoke.py`.
