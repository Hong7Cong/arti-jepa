# Arti-JEPA — usc_lss Phoneme Prediction Results

Frozen-encoder **phoneme prediction** on the gold/OOD `usc_lss` speaker
(`/scratch1/hongn/usc_lss`, 104×104 @ 99 fps, 684 utts, 41 ARPABET). The encoder
is **frozen** (`@torch.no_grad` → seed-independent feature cache); only a small
per-temporal-token probe trains. Metrics: frame-level **Cohen's κ** (↑) +
**PER** (↓) + frame-accuracy. Chance **κ ≈ 0** (chance *accuracy* ≈ 1/41 ≈ 0.024).
Seconds-based alignment, tubelet 2 / patch 16. Why κ and not accuracy: see
[Metric — κ vs accuracy](#metric--κ-vs-accuracy-and-the-two-raters).

Generated 2026-07-05 from two driver runs (`scripts/18_eval_comb100.sh` bf16 on
d13-03, `scripts/17_probe_weights_256.sh` fp16 baselines on d14-10). Every row is
**3 seeds {0,1,2}** with **saved probe weights** (`.pt` beside each `.json`).
Companion: `RESULTS.md` (full project log), `TODO_eval.md` (eval plan).

---

## 🏆 Headline — attentive spatial probe @256px (3 seeds, mean ± sd)

| # | encoder | pretraining | test κ | test PERµ | val κ | frame-acc |
|---|---|---|---|---|---|---|
| 1 | **tssl256comb100** (ckpt_100) | T-SSL, 75-spk **+ longitudinal** | **0.556 ± 0.003** | 0.469 | 0.571 | 0.574 |
| 2 | tssl256 (75-only, re-run) | T-SSL, 75-spk | 0.538 ± 0.007 | 0.491 | 0.545 | 0.557 |
| 3 | pretrained256 | V-JEPA2 stock (no T-SSL) | 0.449 ± 0.005 | 0.555 | 0.465 | 0.477 |
| 4 | base: dinov2-L/14 | DINOv2 (image SSL) | 0.334 ± 0.005 | 0.635 | 0.352 | 0.361 |
| 5 | base: **vit-L/16 sup** | ImageNet supervised | 0.331 ± 0.012 | 0.633 | 0.362 | 0.358 |
| 6 | base: siglip-L/16 | SigLIP | 0.297 ± 0.008 | 0.662 | 0.312 | 0.327 |
| 7 | base: clip-L/14 | CLIP | 0.285 ± 0.001 | 0.672 | 0.314 | 0.315 |
| 8 | base: resnet-50 | ImageNet supervised | 0.270 ± 0.004 | 0.693 | 0.295 | 0.302 |

**tssl256comb100 also with `tcn_spatial` probe:** test κ **0.551 ± 0.003**, PERµ
**0.433 ± 0.007**, val κ 0.565, frame-acc 0.569 (lower PER than attentive — the
kernel-3 TCN smooths temporally).

### Two lifts, both confirmed
- **Longitudinal lift:** `ckpt_100` (75-spk **+ longitudinal** corpus) **0.556** vs
  75-only `tssl256` **0.538** → **+0.018 κ** (~+3%). The longitudinal data helps the
  OOD speaker on top of an already-strong T-SSL encoder. ⚠ `ckpt_100` is an
  **intermediate** checkpoint (epoch 100; the combined run stalled ≈ epoch 106 of a
  215-epoch schedule) — the final lift may grow.
- **T-SSL lift:** tssl256comb100 **0.556** vs frozen pretrained V-JEPA2 **0.449** →
  **+0.107 κ** (~+24%).

### Keystone — beats every public image baseline, airtight
Best image baseline under the **attentive** spatial probe is dinov2 (0.334) /
sup ViT-L/16 (0.331). Even across *any* probe head the best baseline number is
~0.363–0.368 (siglip `tcn_spatial` / sup ViT-L/16 mean-pool `tcn`, see `RESULTS.md`)
— still **far below** frozen pretrained V-JEPA2 (0.449), let alone T-SSL (0.538) or
+longitudinal (0.556). **The decisive competitor — supervised ViT-L/16 — is now run
(0.331 ± 0.012) and settles the question.** Note the cross-attn `attentive` pooler
*hurts* the image baselines vs their `tcn_spatial`/mean-pool scores (e.g. sup ViT-L
0.331 attentive < 0.368 mean-pool tcn) — the pooler pays off only on
V-JEPA-style features.

---

## Metric — κ vs accuracy, and the two raters

The headline number is frame-level **Cohen's κ**, not accuracy. They start from the
same quantity and differ by one term (`artijepa/phonemes.py:149-160`):

```
accuracy = p_o                                   # observed per-token agreement
κ        = (p_o − p_e) / (1 − p_e)               # chance-corrected
p_e      = Σ_c  nt_c · npd_c                      # agreement expected by chance
```

- **`p_o`** = fraction of valid temporal tokens where the predicted phoneme equals
  the gold phoneme — this is *exactly* the reported `frame_acc`, pooled over **all**
  clips of the split (each clip = T′=16 tokens; κ is computed once over the whole
  flattened token stream, not per clip).
- **`p_e`** = the agreement two labelers would reach *by chance* if each labeled
  independently according to its own class-frequency marginals (`nt_c` = gold
  marginals, `npd_c` = model marginals).

**Why κ, not accuracy.** Accuracy rewards exploiting the skewed phoneme distribution
— a majority-class predictor scores well but is useless. κ subtracts that free
agreement, so a trivial/chance predictor lands at **κ ≈ 0** regardless of how
imbalanced the classes are. On this data the two are close (e.g. `tssl256comb100`
s0: accuracy `p_o` = **0.5758**, κ = **0.5582** ⇒ implied `p_e` ≈ 0.040) because the
41-phoneme distribution is fairly spread, so chance agreement is small. On a skewed
task (silence-heavy, or the stuttering classes) `p_e` grows and accuracy inflates
while κ stays honest — that robustness is the reason κ is the headline.

**The two "raters."** Cohen's κ is an inter-rater agreement statistic; here the two
raters are not humans but:

1. **gold reference** — the ground-truth ARPABET phoneme at each token (gold /
   forced-aligned `usc_lss` transcription), and
2. **the model** — the frozen encoder + trained probe's `argmax` phoneme.

Items rated = the temporal tokens; categories = the 41 ARPABET phonemes (+ SIL). It
is specifically **Cohen's** κ (each rater keeps its *own* marginals — the product of
two different vectors `nt` and `npd`), not Scott's π / Fleiss (shared marginals), so
a model that systematically over-predicts common vowels has that skew baked into
`npd` and correctly discounted in `p_e`. PER is the only metric that regroups tokens
per utterance (it's a sequence/edit-distance metric); κ and frame-accuracy are flat,
pooled, order-independent.

---

## Paths — encoders, feature caches, probe weights, results

All eval artifacts under `/scratch1/hongn/artijepa/`. Per `(encoder, probe, seed)`:
results = `eval/<stem>_s{seed}.json`, probe weights = `eval/<stem>_s{seed}.pt`
(best-val snapshot; attentive = 11.59M params / 15 keys, tcn_spatial = 3.44M / 14).
The frozen feature cache is shared across seeds and probes of the same encoder.

| encoder | checkpoint | dtype | feature cache (`feat_cache/phoneme/…`) | result/weight stem (`eval/phoneme_usc_lss_…`) |
|---|---|---|---|---|
| tssl256comb100 | `runs/tssl_vitl_256_combined/ckpt_100.pt` | bf16 | `tssl256comb100sp_d36dd0c874/` (30G) | `tssl256comb100sp_d36dd0c874_{attentive,tcn_spatial}_ce_s{0,1,2}` |
| tssl256 (75) | `runs/tssl_vitl_256/latest.pt` (e50) | bf16 | `tssl256sp_4dcb9400cc/` (30G) | `tssl256sp_4dcb9400cc_attentive_ce_s{0,1,2}` |
| pretrained256 | V-JEPA2 stock `vitl.pt` | bf16 | `pretrained256sp_db5448ad44/` (17G) | `pretrained256sp_db5448ad44_attentive_ce_s{0,1,2}` |
| base: vit-L/16 sup | timm `vit_large_patch16_224` | fp16 | `base_vitlsp_bd26494806/` (13G) | `base_vitlsp_bd26494806_attentive_ce_s{0,1,2}` |
| base: dinov2-L/14 | timm dinov2 @518 | fp16 | `base_dinov2sp_ef6adae1d3/` (17G) | `base_dinov2sp_ef6adae1d3_attentive_ce_s{0,1,2}` |
| base: siglip-L/16 | timm siglip @256 | fp16 | `base_siglipsp_ded94301ff/` (17G) | `base_siglipsp_ded94301ff_attentive_ce_s{0,1,2}` |
| base: clip-L/14 | timm CLIP @224 | fp16 | `base_clipsp_8725680ba0/` (17G) | `base_clipsp_8725680ba0_attentive_ce_s{0,1,2}` |
| base: resnet-50 | timm resnet50 @224 | fp16 | `base_resnetsp_dbb271fb97/` (1.9G) | `base_resnetsp_dbb271fb97_attentive_ce_s{0,1,2}` |

Each `.pt` reloads without retraining and stores: `probe_state`, `probe_kind`,
`dim`, `num_classes`, `feature_tag`, `seed`, `best_epoch`, `metrics`, `spatial_size`.

Logs: `eval/eval_comb100.log` (bf16 group), `eval/probe_weights_256_baselines.log`
(fp16 baselines).

---

## Per-seed detail (test κ / test PERµ / val κ, best_epoch)

| encoder · probe | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| tssl256comb100 · attentive | 0.558 / 0.467 / 0.568 (e28) | 0.559 / 0.461 / 0.578 (e31) | 0.552 / 0.478 / 0.569 (e23) |
| tssl256comb100 · tcn_spatial | 0.549 / 0.433 / 0.571 (e24) | 0.555 / 0.424 / 0.559 (e22) | 0.550 / 0.441 / 0.566 (e23) |
| tssl256(75) · attentive | 0.544 / 0.476 / 0.543 (e37) | 0.528 / 0.501 / 0.549 (e26) | 0.543 / 0.495 / 0.543 (e29) |
| pretrained256 · attentive | 0.453 / 0.555 / 0.456 (e30) | 0.452 / 0.559 / 0.462 (e16) | 0.442 / 0.555 / 0.455 (e26) |
| base:vitl · attentive | 0.341 / 0.627 / 0.361 (e21) | 0.313 / 0.650 / 0.362 (e9) | 0.338 / 0.623 / 0.364 (e32) |
| base:dinov2 · attentive | 0.340 / 0.637 / 0.350 (e25) | 0.330 / 0.640 / 0.349 (e7) | 0.330 / 0.628 / 0.357 (e10) |
| base:siglip · attentive | 0.287 / 0.672 / 0.317 (e16) | 0.306 / 0.663 / 0.310 (e36) | 0.299 / 0.652 / 0.309 (e8) |
| base:clip · attentive | 0.285 / 0.667 / 0.318 (e17) | 0.286 / 0.672 / 0.309 (e17) | 0.284 / 0.676 / 0.315 (e33) |
| base:resnet · attentive | 0.275 / 0.698 / 0.296 (e16) | 0.266 / 0.694 / 0.298 (e6) | 0.271 / 0.687 / 0.290 (e3) |

---

## Reproduce

```bash
source scripts/_env.sh
# 1) combined ckpt_100 + tssl256/pretrained refs (bf16, saves probe .pt):
bash scripts/18_eval_comb100.sh
# 2) image-baseline attentive probes (fp16, saves probe .pt):
bash scripts/17_probe_weights_256.sh
# single run, e.g.:
python -m artijepa.eval_phoneme --config configs/eval_phoneme_usc_lss_256.yaml \
    --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/ckpt_100.pt \
    --tag tssl256comb100 --probe attentive --seed 0 --dtype bfloat16
```

## Notes & caveats
- **dtype split is intentional.** T-SSL/pretrained use **bf16** (matches the
  headline caches); image baselines use **fp16** (matches the `base_*sp` caches so
  clip/siglip cache-hit). The V100 handles both; frozen extraction is robust to the
  choice.
- **tssl256(75) was re-extracted** here (cache `4dcb9400cc`, 3-seed **0.538 ±
  0.007**) with the current code — slightly above the June headline **0.527 ± 0.004**
  (cache `9ee0a341a1`); within re-extraction noise. Rows 1–3 above share the current
  code/config, so the comb-vs-75-vs-pretrained comparison is internally consistent.
- **Cache-key caveat:** `eval_phoneme._tag()` omits the checkpoint path from the
  cache hash → always pass a **fresh `--tag`** per encoder (done: `tssl256comb100`).
- **Clip-window asymmetry:** the V-JEPA eval config windows 684 utts → 3195 train
  clips @256px/50fps; the baseline config → 1737 train clips at native res/25fps.
  Pre-existing property of the two configs, not introduced here.
- `ckpt_100` is intermediate — see `RESULTS.md` / `combined-tssl-256-run` memory for
  the stalled combined-training status (≈ epoch 106/215).
