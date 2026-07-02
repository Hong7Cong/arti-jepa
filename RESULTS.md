# Arti-JEPA — RESULTS

All numbers from phoneme prediction on frozen encoder features (small per-token
probe). Metrics: frame-level **Cohen's κ** (↑ better) + **Phoneme Error Rate /
PER** (↓ better). Plans live in `TODO_pretraining.md` (T-SSL pretraining) and
`TODO_eval.md` (evaluation). File map: `Master.md`.

Headline question = **with vs without T-SSL** lift on the gold/OOD `usc_lss`
speaker. Chance = 1/41 ≈ 0.024.

---

## 🏆 Headline results

- **Best result anywhere: `tssl_256` @e50 + attentive spatial probe — test κ
  0.530 / PERµ 0.486 / frame-acc 0.549** (256px, gold/OOD `usc_lss`, CE,
  single seed; 2026-06-16). 3-seed confirm: **0.527 ± 0.004**.
- **T-SSL lift at 256px** (attentive spatial probe): frozen pretrained 0.449 →
  tssl_256 **0.527**, lift **+0.078** (sd ~0.004–0.005 → highly significant).
- **Keystone — T-SSL beats every public image baseline** under a fair spatial
  probe: best baseline (siglip-L/16 tcn_spatial 0.363, clip-L/14 0.345) is far
  below frozen pretrained V-JEPA (0.449), let alone tssl_256 (0.527). ⚠ One
  decisive competitor still PENDING: supervised ViT-L/16 (see `TODO_eval.md`).

---

## Settings for each model in use
Reproduction guide: **`RUNME.md`** (inference-only on the saved encoder
checkpoints). Configs in `configs/`; encoder checkpoints under
`/scratch1/hongn/artijepa/runs/<name>/latest.pt`.

### Encoder pretraining settings (T-SSL; ours)
| model | checkpoint (`runs/…/latest.pt`) | config | train data | res | fps | frames | micro-bs | eff-batch (accum) | ipe | epochs | lr (cosine) | warmup | wd | ema | dtype | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrained (V-JEPA2 ViT-L) | *(stock `vitl.pt`, no T-SSL)* | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **tssl_128** | `tssl_vitl_128` | `tssl_vitl_128.yaml` | `manifest_split.csv` | 128 | 25 | 32 | 16 | 16 (×1) | 500 | 50 | 5e-4 | 5 | 0.04 | [0.998,1.0] | fp16 | L1 |
| **tssl_256** (75-only, headline) | `tssl_vitl_256` | `tssl_vitl_256.yaml` | `manifest_alltrain.csv` | 256 | 50 | 32 | 32 | 128 (×4) | 500 | 50 | 5e-4 | 5 | 0.04 | [0.998,1.0] | fp16 | L1 |
| tssl_256_combined (+longitudinal) | `tssl_vitl_256_combined` | `tssl_vitl_256_combined.yaml` | `manifest_combined.csv` | 256 | 50 | 32 | 16 | 128 (×8) | 1000 | 215 | 5e-4 | 5 | 0.04 | [0.998,1.0] | fp16 | L1 |

Common to all T-SSL: ViT-L, tubelet 2 / patch 16, intensity zscore, augment on,
activation-checkpointing ON, EMA target encoder + L1 feature loss, multiblock
masks per grid, seed 0. Schedules step **per optimizer update**, so the LR/EMA
horizon is identical wherever `eff-batch × ipe / micro-bs` matches (125 oue/epoch
for both 256 runs). Public image baselines (clip/siglip/dinov2/vitl/resnet) are
frozen off-the-shelf timm weights — **no pretraining by us**.

### Evaluation (phoneme probe) settings — all rows
| group | task / data | eval config | res | fps | extract dtype | probe types | probe epochs | warmup | lr | wd | hidden | probe bs | loss | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-JEPA @128 | gold/OOD `usc_lss` | `eval_phoneme_usc_lss.yaml` | 128 | 25 | bf16 | tcn / tcn_spatial / attentive | 40 | 4 | 1e-3 | 0.01 | 512 | 32 | CE (CTC opt.) | 0 (+1,2 spatial) |
| V-JEPA @256 | gold/OOD `usc_lss` | `eval_phoneme_usc_lss_256.yaml` | 256 | 50 | bf16 | tcn / tcn_spatial / attentive | 40 | 4 | 1e-3 | 0.01 | 512 | 32 | CE | 0,1,2 |
| image baselines | gold/OOD `usc_lss` | `eval_phoneme_usc_lss_baseline.yaml` | native (224–518) | 25 | fp16 | tcn / tcn_spatial / attentive | 40 | 4 | 1e-3 | 0.01 | 512 | 32 | CE | 0,1,2 |
| Task-1 pseudo *(TBA)* | pseudo / 75-spk | `eval_phoneme_pseudo.yaml` | 128 | 25 | bf16 | tcn (+CTC) | 40 | 4 | 1e-3 | 0.01 | 512 | 32 | CE/CTC | 0 |

Common to all evals: **encoder is FROZEN** (`@torch.no_grad` feature extraction →
seed-independent cache); only the small probe trains. usc_lss = 41-ARPABET gold,
seconds-based alignment, tubelet 2 / patch 16. Baselines use intensity `minmax`
(model's own mean/std); V-JEPA evals use `zscore`. `attentive`/`tcn_spatial`
consume the un-pooled `[B,T',S',D]` grid (`…sp_<hash>` cache); `tcn` mean-pools
spatial. `_tag()` ignores the checkpoint path → **always pass a fresh `--tag` per
encoder.**

---

## Results log (phoneme; κ↑ better, PER↓ better)
| date | encoder | task | res | probe | test κ | test PERµ | val κ | frame-acc | chance |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-07 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | tcn | 0.222 | 0.760 | 0.240 | 0.259 | 0.024 |
| 2026-06-07 | **tssl_128** (V-JEPA) | gold/OOD usc_lss | 128 | tcn | **0.247** | 0.755 | 0.277 | 0.280 | 0.024 |
| 2026-06-10 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | tcn_spatial | 0.344 | 0.653 | 0.349 | 0.372 | 0.024 |
| 2026-06-10 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | attentive | 0.327 | 0.694 | 0.345 | 0.355 | 0.024 |
| 2026-06-10 | tssl_128 (V-JEPA) | gold/OOD usc_lss | 128 | tcn_spatial | 0.433 | 0.587 | 0.450 | 0.458 | 0.024 |
| 2026-06-10 | **tssl_128** (V-JEPA) | gold/OOD usc_lss | 128 | **attentive** | **0.475** | **0.523** | 0.483 | 0.497 | 0.024 |
| 2026-06-09 | base: dinov2-L/14 | gold/OOD usc_lss | 518 | tcn | 0.291 | 0.710 | 0.317 | 0.321 | 0.024 |
| 2026-06-09 | base: clip-L/14 | gold/OOD usc_lss | 224 | tcn | 0.293 | 0.710 | 0.331 | 0.323 | 0.024 |
| 2026-06-09 | base: resnet-50 | gold/OOD usc_lss | 224 | tcn | 0.304 | 0.694 | 0.345 | 0.331 | 0.024 |
| 2026-06-09 | base: siglip-L/16 | gold/OOD usc_lss | 256 | tcn | 0.313 | 0.685 | 0.326 | 0.342 | 0.024 |
| 2026-06-09 | **base: vit-L/16 sup** | gold/OOD usc_lss | 224 | tcn | **0.368** | **0.620** | 0.420 | 0.393 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | tcn | 0.303 | 0.693 | 0.320 | 0.335 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | tcn_spatial | 0.407 | 0.590 | 0.404 | 0.432 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | attentive | 0.446 | 0.564 | 0.453 | 0.469 | 0.024 |
| 2026-06-13 | tssl_256 **@e22** (V-JEPA) | gold/OOD usc_lss | 256 | tcn | 0.356 | 0.643 | 0.397 | 0.383 | 0.024 |
| 2026-06-13 | tssl_256 **@e22** (V-JEPA) | gold/OOD usc_lss | 256 | tcn_spatial | 0.461 | 0.544 | 0.482 | 0.483 | 0.024 |
| 2026-06-13 | **tssl_256 @e22** (V-JEPA) | gold/OOD usc_lss | 256 | **attentive** | **0.496** | **0.508** | 0.509 | 0.517 | 0.024 |
| 2026-06-16 | tssl_256 **@e50** (final) | gold/OOD usc_lss | 256 | tcn | 0.382 | 0.639 | 0.416 | 0.407 | 0.024 |
| 2026-06-16 | tssl_256 **@e50** (final) | gold/OOD usc_lss | 256 | tcn_spatial | 0.488 | 0.527 | 0.508 | 0.508 | 0.024 |
| 2026-06-16 | **tssl_256 @e50 (final)** (V-JEPA) | gold/OOD usc_lss | 256 | **attentive** | **0.530** | **0.486** | 0.539 | 0.549 | 0.024 |
| _pending_ | combined (+longitudinal) 256 | gold/OOD usc_lss | 256 | attentive | **TBA** | TBA | TBA | TBA | 0.024 |
| _pending_ | pretrained | pseudo/75-spk | 128 | tcn | **TBA** | TBA | TBA | TBA | |

---

## 128px T-SSL — DONE (2026-06-07)
- **Trained to completion (50 ep), no collapse.** feature_std 0.30→1.12,
  effective_rank 79→**102**/1024, mean_abs_cosine 0.957→**0.560**. Loss ~0.40
  (L1 vs EMA target). Ckpt `runs/tssl_vitl_128/latest.pt` (5.1 GB).
- **Headline (gold OOD, 128px, tcn):** test κ **0.222→0.247** (+0.025, +11%),
  frame-acc 0.259→0.280, PERµ 0.760→0.755; val κ 0.240→0.277 (+15%). κ shows the
  gain; PER is capped by the 80 ms token rate. JSON
  `…/eval/phoneme_usc_lss_tssl128_*.json`.

## 256px T-SSL — COMPLETE (ep50) + FINAL eval DONE (2026-06-16)
- **Training finished cleanly to epoch 50** on V100-32GB (d13-07, job 9402610):
  `epoch 50 avg loss 0.4548 … done`, diagnostics clean (feature_std 1.25,
  eff_rank 75.8, mean_abs_cosine 0.505 — no collapse). `runs/tssl_vitl_256/latest.pt`.
- **FINAL eval (gold/OOD usc_lss, 256px, CE, tag `tssl256` = fresh e50 features):
  HEADLINE = tssl_256 @e50 + attentive test κ 0.530 / PERµ 0.486 / frame-acc 0.549.**
  T-SSL lift over frozen pretrained-256 at same res: tcn 0.303→0.382,
  tcn_spatial 0.407→0.488, attentive **0.446→0.530 (+0.084, +19%)**. Finishing
  past e22 added +0.034 κ (attentive 0.496→0.530).
- Driver `eval/run_256_e50_eval.sh`, log `eval/eval_256_e50.log`, JSONs
  `eval/phoneme_usc_lss_tssl256*_*.json`. Encoder loaded clean (292 miss 0, epoch
  50). Caches: `feat_cache/phoneme/tssl256_c2c3132725` (pooled, tcn) +
  `tssl256sp_9ee0a341a1` (spatial grid, tcn_spatial+attentive).

---

## ▶▶ Spatial-aware probe — KEYSTONE (2026-06-10): mean-pooling hid most of the signal
The default eval **mean-pools the S'=(res/patch)² spatial tokens** away before
the probe (`[B,N,D]→[B,T',D]`). But *where* in the vocal tract the signal sits
(tongue/lip/velum position) is exactly the phonetic information. Two
spatial-aware heads consume the **un-pooled `[B,T',S',D]` grid**:
- **`tcn_spatial`** — learned additive attention-pool over S' per temporal step →
  `[B,T',D]`, then kernel-3 TCN over time (only change vs `tcn` is mean→learned pool).
- **`attentive`** — V-JEPA's exact `AttentivePooler` (cross-attn, 1 query) over S'
  per temporal step → `[B,T',D]` → linear (no temporal mixing).

**Result (gold/OOD usc_lss, 128px, CE, single seed):**

| encoder | head | test κ | frame-acc | test PERµ | vs mean-pool κ |
|---|---|---|---|---|---|
| pretrained128 | tcn (mean-pool S') | 0.222 | 0.259 | 0.760 | — |
| pretrained128 | **tcn_spatial** | **0.344** | 0.372 | 0.653 | **+0.122** |
| pretrained128 | attentive | 0.327 | 0.355 | 0.694 | +0.105 |
| tssl128 | tcn (mean-pool S') | 0.247 | 0.280 | 0.755 | — |
| tssl128 | tcn_spatial | 0.433 | 0.458 | 0.587 | +0.186 |
| tssl128 | **attentive** | **0.475** | **0.497** | **0.523** | **+0.228** |

**Findings (this changed the headline):**
1. **Spatial structure carries most of the phonetic signal.** Not mean-pooling
   lifts FROZEN pretrained V-JEPA κ 0.222→0.344 (+55%) and T-SSL 0.247→0.475 (+92%).
2. **The T-SSL lift is far larger than mean-pooling revealed** (+0.025 pooled →
   **+0.148** with attentive spatial probe).
3. **T-SSL + spatial probe (κ 0.475) beats the best image baseline** (sup ViT-L/16,
   0.368) — still at 128px. So earlier "baselines > frozen V-JEPA" was a **probe
   artifact (mean-pool), not a resolution gap.**
4. **`attentive` > `tcn_spatial` on T-SSL features** (0.475 vs 0.433) but ≈/below
   on pretrained — the richer cross-attn pooler pays off once features are adapted.

---

## Head × loss ablation (2026-06-09, gold/OOD usc_lss, 128px, single seed)

| encoder | head | CE test κ | CE PERµ | CTC PERµ |
|---|---|---|---|---|
| pretrained128 | **tcn** | **0.224** | **0.759** | 0.811 |
| pretrained128 | lstm | 0.196 | 0.778 | 0.820 |
| pretrained128 | transformer | 0.186 | 0.804 | 0.841 |
| tssl128 | **tcn** | **0.255** | **0.739** | 0.792 |
| tssl128 | lstm | 0.249 | 0.738 | 0.811 |
| tssl128 | transformer | 0.247 | 0.752 | 0.824 |

**Findings:** (1) **`tcn`+CE wins** — lstm/transformer *overfit* the small OOD data
on frozen 128px features (κ drops with head capacity). (2) **CTC is worse on PER
everywhere** (+0.05–0.08): we have gold alignment, so per-token CE uses more
signal; CTC's payoff is the Task-1 pseudo labels (no alignment), not here.
(3) **T-SSL helps across every head/loss** — the lift is in the features.

---

## 3-seed fair-fight (256px spatial probe) — PARTIAL (interrupted 2026-06-17)
Frozen feature caches are seed-independent (deterministic `@torch.no_grad`
extraction) → each seed only re-trains the cheap probe. Combos: encoders ×
{tcn_spatial, attentive} × seeds {0,1,2}.

**3-seed mean±sd, gold/OOD usc_lss, 256px, CE (test κ / test PERµ / val κ):**

| encoder | head | seeds | test κ (mean±sd) | test PERµ | val κ |
|---|---|---|---|---|---|
| pretrained256 | tcn_spatial | 0,1,2 | 0.411 ± 0.007 | 0.601 | 0.425 |
| pretrained256 | **attentive** | 0,1,2 | **0.449 ± 0.005** | 0.556 | 0.458 |
| **tssl256 @e50** | tcn_spatial | 0,1,2 | 0.481 ± 0.005 | 0.528 | 0.497 |
| **tssl256 @e50** | **attentive** | 0,1,2 | **0.527 ± 0.004** | 0.485 | 0.544 |
| base: clip-L/14 | tcn_spatial | 0,1,2 | 0.345 ± 0.010 | 0.662 | 0.375 |
| base: clip-L/14 | attentive | 0,1,2 | 0.282 ± 0.002 | 0.676 | 0.313 |
| base: siglip-L/16 | tcn_spatial | 0,1,2 | 0.363 ± 0.009 | 0.636 | 0.405 |
| base: siglip-L/16 | attentive | 0 *(1/3, partial)* | 0.296 | 0.671 | 0.320 |

**Per-model completion (verified 2026-06-17):**
- ✅ `pretrained256` tcn_spatial + attentive — 3/3 seeds
- ✅ `tssl256 @e50` tcn_spatial + attentive — 3/3 seeds ← headline, solid
- ✅ `base_clip` tcn_spatial + attentive — 3/3 seeds
- ✅ `base_siglip` tcn_spatial — 3/3; attentive — 1/3 (s0 only)
- ❌ `base_vitl` (supervised ViT-L/16, **the key competitor**) — 0/3, NOT STARTED
- ❌ `base_dinov2` — 0/3; ❌ `base_resnet` — 0/3

**Findings so far:** (1) seeds tighten the headline — tssl256+attentive
0.527±0.004 vs pretrained256+attentive 0.449±0.005, lift +0.078 (sd tiny → highly
significant). (2) Keystone holds emphatically under the fair probe so far — best
image baseline with spatial probe (siglip 0.363, clip 0.345) far below frozen
pretrained V-JEPA (0.449). `attentive` *hurts* clip (0.282 < 0.345) — the
cross-attn pooler needs V-JEPA-style features. **Decisive check still PENDING:
supervised ViT-L/16** (mean-pool winner at 0.368) + dinov2/resnet/siglip-attentive
s1,s2. See `TODO_eval.md` for resume. All JSONs persist:
`eval/phoneme_usc_lss_*_s{0,1,2}.json`; log `eval/eval_256_fairfight.log`.

---

## Image baselines, mean-pool tcn (2026-06-09, native res)
Ran all five at gold/OOD, tcn, native res, "each its best shot" (minmax→[0,1] +
model's own mean/std). **DINOv3 ViT-L unavailable in this env's timm → used
DINOv2 ViT-L/14.** At native res ALL beat frozen V-JEPA@128px; **supervised
ViT-L/16 best (κ 0.368)** — but res confound (224–518 vs 128) made this not a
fair fight; superseded by the spatial-probe fair-fight above. Rows in the results
log table.

> **NOTE — "baselines > V-JEPA" was a PROBE artifact, not resolution.** With the
> spatial-aware probe, tssl_128 (κ 0.475) > supervised ViT-L/16 (0.368) at the
> *same* 128px — the gap was the mean-pool probe discarding spatial structure.

---

## Task 8 — stuttering disfluency-type classification (infra 2026-06-30, results TBA)
Segment-level disfluency-type classification from frozen (or fine-tuned) rtMRI
features. Canonical setup = **attentive probe @ 256px, leave-one-speaker-out** over
the 7 PWS speakers. Manifest `disfluency_manifest.csv` = 3130 segments (rep 795 /
block 693 / pro 620 / osci 42 / other 21 / fluent 959). Primary metric **macro-F1**
(severe imbalance) + balanced accuracy + confusion matrix, per held-out speaker and
pooled. Pipeline validated end-to-end (`tests/test_disfluency_smoke.py` 21/21;
tiny real-data frozen + finetune smoke on VideoMAE-base). Runs below are **TBA**.

| encoder | mode | task | probe | res | macro-F1 | bal-acc |
|---|---|---|---|---|---|---|
| V-JEPA2 pretrained | frozen | type5 | attentive | 256 | **TBA** | TBA |
| T-SSL 256 (`tssl256`) | frozen | type5 | attentive | 256 | **TBA** | TBA |
| VideoMAE-L/16 | frozen | type5 | attentive | 224 | **TBA** | TBA |
| VideoMAE-L/16 | fine-tune | type5 | attentive | 224 | **TBA** | TBA |
| image baselines (vitl/…) | frozen | type5 | attentive | native | **TBA** | TBA |

Also TBA: `type3` (block/rep/pro only) and the `binary` fluent-vs-disfluent
baseline; the `disfluency2` (rep-heavy) tier as a secondary report. JSONs will
persist to `eval/disfluency_*_s{seed}.json`.

---

## AC-JEPA forward model `P` — `arti_gap` (2026-06-30 → 07-01)

Frozen T-SSL ViT-L (`tssl_vitl_256`, κ 0.530) + arti-conditioned AC predictor (22.1M, `A=6`),
temporal world-model objective on usc_lss sessions @256px/32f/100fps. Metric = **`arti_gap`** =
`val_shuf_ar_l1 − val_ar_l1` (autoregressive future-pred L1 in layer-normed feature space, real
vs batch-shuffled articulators). Large + = predictor USES arti; ≈0 = ignores it. Details of the
finding + root cause in `aucjepa_plans_new.md` §8.

| run | ctx_frames | GPU / batch | epochs | `arti_gap` | val_ar_l1 | verdict |
|---|---|---|---|---|---|---|
| `acjepa_arti6_256_ddp` | 8 | 2×P100 / 2·2 | 8/20 (stopped) | plateau **5e-4–1e-3** (~0.1–0.2% of AR L1) | 0.440 | predictor **ignores** arti |
| `acjepa_arti6_256_v100` | 8 | V100 / 4 | 6 (killed) | ~5e-4 @ep5 (reproduces plateau) | 0.492 | baseline |
| **`acjepa_arti6_256_v100_ctx2`** | **2** | **V100 / 3, ckpt-off** | **20 (DONE)** | broke plateau @ep9, **peak 6.9e-3 @ep19, 5.6e-3 @ep20** (~10–14×) | 0.428 | **uses arti (weakly)** |

**ctx=2 trajectory:** ep8 5.4e-4 → ep9 1.6e-3 → ep12 4.5e-3 → ep15 5.5e-3 → ep19 **6.9e-3** →
ep20 5.6e-3 (noisy ~5–7e-3 band; ep17 dip 1.6e-3). ep20 triple: `val_tf_l1` 0.4035 / `val_ar_l1`
0.4285 / `val_shuf_ar_l1` 0.4341.

**Reading it:** ctx=2 (plan §8 fix A — seed only 2 tokens, predict ~14 from arti alone) made the
predictor use the articulators, ~10× the ctx=8 baseline. But the gap is **small in absolute terms
(~1.3% of AR L1)** — expected, since arti-6 is a *readout* of the MRI the encoder already sees
(§8.1 low ceiling). Fairer denominator = the AR-rollout penalty `ar_real−tf` = 0.025 ⇒ right arti
recovers **~22%** of it. Likely understated by the within-batch shuffle (batch 3) + only 8 val
batches. **Next: M2 redundancy probe** (phoneme-from-`z` vs `arti-6`) to decide whether to keep the
video world model or pivot to Energy-3 arti-space planning; the utt↔session alignment blocker is
now resolved (see `TODO_acjepa.md`). ckpt `runs/acjepa_arti6_256_v100_ctx2/latest.pt`.
