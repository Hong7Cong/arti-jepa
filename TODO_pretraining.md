# Arti-JEPA — TODO: T-SSL Pretraining

Domain-adaptive self-supervised pretraining of V-JEPA 2 ViT-L on **unlabeled
rtMRI vocal-tract video**. This is the headline track; the downstream lift is
measured in `TODO_eval.md` and results live in `RESULTS.md`. File map: `Master.md`.

**Env / node:** conda `artijepa` (torch 2.6+cu124). `source dev_artiJEPA/scripts/_env.sh`.
Current node = L40S (46 GB, Ada cc 8.9, bf16); was V100 — check `nvidia-smi`.
T-SSL configs use float16 (V100-safe, also fine on L40S); flip `meta.dtype:
bfloat16` on L40S/A100/H100.

**Method:** `tssl_train.py` — EMA target encoder, L1 feature loss, multiblock
masks per grid; loads pretrained V-JEPA2 ViT-L clean (292/292). Logs **label-free**
collapse metrics only (feature_std, effective_rank, mean_abs_cosine). Gradient
accumulation via `optimization.effective_batch` (schedules step per optimizer
update); resume via `--resume <ckpt>` (atomic save, `.prev` backup, scaler+schedule
restore).

**Artifacts** under `/scratch1/hongn/artijepa/` (never `/project2`):
`manifest_split.csv` (75-spk: train 1808 / val 279 / test 284, subject-disjoint),
`manifest_combined.csv` (+longitudinal), `grayscale_stats.json`,
`runs/<name>/{latest.pt,train_log.csv,diagnostics.jsonl,train_stdout.log}`.

---

## ▶ IN PROGRESS — WITH-longitudinal 256px combined run
**Status 2026-06-29 09:33: @ epoch 50/215, RUNNING on V100-32GB (d14-07, job 9742362).**
- Resumed cleanly: `RESUMED latest.pt @ epoch 49 (6125 opt-updates fast-forwarded);
  continuing to epoch 215` (6125 = 49×125 oue ✓). Training epoch 50 @ ~12 s/iter,
  GPU 24.2/32 GB / 100%.
- Diagnostics clean through ep49 — **no collapse** (feature_std ~1.42, eff_rank
  ~40/1024, mean_abs_cosine ~0.44); raw L1 loss on its expected ~0.40–0.41 high-LR
  plateau (LR still ~4.5e-4, cosine decays in the back half — see
  [[combined-tssl-256-run]]).
- The run died once ~02:29 (alloc/SSH loss) mid epoch-50 step 140; `latest.pt` =
  clean end-of-epoch-49 ckpt → re-resumed.
- **~166 epochs left × ~100 min ≈ ~11.5 days** → spans many allocations;
  checkpoints every epoch.

**Config** `configs/tssl_vitl_256_combined.yaml` (folder `runs/tssl_vitl_256_combined`,
**epochs 215** = ~10 passes / ~99.995% coverage), fresh from V-JEPA2 ViT-L
(`pretrained: true`, `resume: null`; loaded 292/292). **bs16 / ipe1000 / accum×8**
→ `effective_batch 128`, **oue=125/epoch** (resume-compatible with original
bs16/ipe1000 ckpts; the LR/EMA schedule + 125×215 = 26,875-update horizon are
byte-identical). OOM fix: micro-batch 16 (was 32) shrinks GPU activation peak
(~24.5→~14–16 GB). Launcher defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**▶ RESUME in a fresh alloc** (use ABSOLUTE ckpt path — no `runs/` symlink under
REPO_ROOT):
```bash
source dev_artiJEPA/scripts/_env.sh
bash dev_artiJEPA/scripts/03_train_tssl.sh \
     dev_artiJEPA/configs/tssl_vitl_256_combined.yaml \
     --resume /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/latest.pt
```

**Next (→ `TODO_eval.md`):** phoneme-eval the combined checkpoint vs the 75-only
`runs/tssl_vitl_256` to measure the longitudinal lift (κ + PER, gold OOD usc_lss).
Results row in `RESULTS.md` is TBA.

---

## DONE
- [x] **Longitudinal corpus added to the pretraining pool (2026-06-24).**
      `/project2/shrikann_35/kevinyhu/data/longitudinal` — 21 disjoint speakers,
      **7,110 `.avi` clips, 104×104 @ ~81.97 fps** (USC `rt_ssfp`, no labels →
      pre-train only). New `build_manifest_longitudinal.py` (parses `nframes`/`tRes`
      from the clip name, auto-decord-probes the 197 short-named residual →
      `manifest_longitudinal.csv`), `merge_manifests.py` (column-reconciling concat +
      subject-collision/leakage guard → `manifest_combined.csv`),
      `scripts/01b_prepare_longitudinal.sh`. Combined train pool: 9,353 videos →
      **343,517 tile chunks @ 256/50fps** (236k longitudinal + 107k 75-speaker,
      ~4× the prior 85.6k); 96 subjects, no overlap. Dataloader needed **zero
      changes** (per-row resolution/fps-agnostic). Validated a longitudinal clip
      end-to-end (`(3,32,256,256)`, finite) + `tests/test_longitudinal_smoke.py` (12/12).
- [x] **128px T-SSL trained to completion (50 ep, 2026-06-07).** No collapse;
      ckpt `runs/tssl_vitl_128/latest.pt`. (Numbers in `RESULTS.md`.)
- [x] **256px T-SSL (75-only) trained to completion (50 ep, 2026-06-16).** No
      collapse; ckpt `runs/tssl_vitl_256/latest.pt`. This is the
      **without-longitudinal baseline**, untouched by the combined run.
- [x] **T-SSL trainer (`tssl_train.py`, B.3).** EMA target, L1 feat loss,
      multiblock masks, gradient accumulation + resume. Both smoke-tested. Loads
      pretrained ViT-L clean (292/292).
- [x] **Phase 0 — data engineering (A.1–A.9).** manifest, subject-disjoint splits,
      grayscale stats, decord linear-interp resample (`crop`/`tile`), safe aug.

## VRAM / tuning notes (256px on 32 GB)
- **Activation checkpointing is MANDATORY** — `use_activation_checkpointing: false`
  OOMs even at bs8 (target encoder sees all 4096 tokens → ~30 GB activations). So
  recompute can't be removed; we stay **compute-bound** (6 vs 2 dataloader workers
  = same speed).
- Only lever is bigger batch under ckpt: bs8≈12 GB, bs16≈14–16 GB, bs32≈24.5 GB,
  bs64 OOMs. Micro-batch must divide effective_batch 128 → {8,16,32,64}.
- P100 ~28 s/step (infeasible); L40S ~40 h (flip `dtype: bfloat16`); V100 ~12 s/step.

## DEFERRED
- T1–T3 fine-tuning ladder; **ViT-g scale-up**; bf16+grad-accum exploration for
  256px (already partly adopted in the combined run).
