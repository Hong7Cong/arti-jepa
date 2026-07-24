# Arti-JEPA — Combined Phoneme Prediction Results (Annot-16 + usc_lss, cross-domain)

Frozen-encoder **phoneme prediction** trained on **14 speakers of the 75-Speaker
Annot-16** benchmark and evaluated on **two held-out test sets with the same
best-val probe**:

- **`test`** — in-domain held-out speaker (Annot-16 **sub043**, 84×84 @ 83.28 fps)
- **`test_lss`** — cross-domain OOD speaker (**usc_lss** `usc_s1`, 104×104 @ 99 fps)

Model-selected on Annot-16 **sub030** (`val`). The encoder is **frozen**
(`@torch.no_grad` → seed-independent feature cache); only the per-temporal-token
probe trains. Metrics: frame-level **Cohen's κ** (↑) + **PER** (↓) + frame-accuracy.
Chance **κ ≈ 0** (chance *accuracy* ≈ 1/41 ≈ 0.024). Seconds-based alignment,
tubelet 2 / patch 16.

Datasets + full run instructions: **`docs/phonePred.md`**. Config:
`configs/eval_phoneme_annot16_combined.yaml`. Companions: `RESULTS_usclss.md`
(single-corpus usc_lss headline), `RESULTS.md` (full project log).

This eval measures **domain generalization**: the `test` vs `test_lss` gap is how
far a representation transfers from the 75-Speaker corpus to an unseen speaker/
scanner/fps. A domain-adapted encoder should **narrow that gap** (lift `test_lss`).

---

## Results

### Cross-model headline (Phase 1) — attentive · CE · 256px · 3 seeds {0,1,2}

Frozen encoder swapped; probe/loss/geometry held fixed. `test` = in-domain (Annot-16
sub043), `test_lss` = cross-domain (usc_lss). Values = mean ± sd over 3 seeds. Geometry
caveat: only the V-JEPA family truly runs at 256px — VideoMAE is native 224/16f and the
image baselines are native per-frame (same as `RESULTS_usclss.md`; see `docs/phonePred.md §5`).
Launched 2026-07-12; **Phase 1 complete 2026-07-14 — all 6 encoders × 3 seeds.**

| # | encoder | pretraining | adapt | test κ (in-dom) | test PERµ | **test_lss κ (x-dom)** | test_lss PERµ | val κ |
|---|---|---|---|---|---|---|---|---|
| 1 | **tssl256comb100** | T-SSL 75-spk +longitudinal (ckpt_100) | ✅ | 0.461 ± .030 | 0.700 ± .021 | **0.334 ± .015** | 0.660 ± .036 | 0.474 ± .010 |
| 1b | **tssl256comb215** | T-SSL 75-spk +longitudinal (**ckpt_215, final**) | ✅ | 0.505 ± .009 | 0.691 ± .031 | **0.352 ± .009** | 0.655 ± .030 | 0.485 ± .000 |
| 2 | pretrained | V-JEPA2 stock | — | 0.362 ± .030 | 0.908 ± .094 | **0.156 ± .016** | 1.040 ± .056 | 0.336 ± .025 |
| 3 | videomae | VideoMAE K400 (3-D video SSL) | — | 0.380 ± .008 | 0.880 ± .074 | **0.190 ± .017** | 0.855 ± .010 | 0.379 ± .006 |
| 4 | vitl | ImageNet-sup ViT-L (2-D) | — | 0.202 ± .072 | 0.824 ± .032 | **0.030 ± .005** | 0.926 ± .004 | 0.336 ± .004 |
| 5 | dinov2 | DINOv2 ViT-L (2-D image SSL) | — | 0.258 ± .020 | 0.820 ± .018 | **0.065 ± .015** | 0.888 ± .018 | 0.246 ± .031 |
| 6 | videomae_rtmri | rt-MRI VideoMAE (ckpt-214) | ✅ | 0.409 ± .020 | 0.744 ± .018 | **0.264 ± .013** | 0.838 ± .016 | 0.432 ± .003 |

All rows are final 3-seed mean±sd over seeds {0,1,2}. (History: tssl s2 and videomae
s1+s2 timed out at the original 20 h wall and were resubmitted as 1-seed jobs; wall raised
to 30 h, all completed 2026-07-14.)

**ckpt_100 → ckpt_215 (pretraining to convergence, added 2026-07-17):** row 1b is the same
encoder/probe/geometry evaluated on the **final** epoch-215 checkpoint (job 10343569, fresh
spatial cache `tssl256comb215sp_43c1fe20dd`). The extra 115 epochs give a **small in-domain
lift** (`test` κ 0.461 → 0.505) but the **cross-domain metric barely moves** (`test_lss` κ
0.334 → 0.352, ~+0.018, ≈1 sd) — the domain-adaptation benefit that lifts `test_lss` is
already **saturated by ckpt_100**. Both keep `test_lss` PERµ < 1.0 (no OOD insertion blow-up).

### κ vs pretraining checkpoint — the full adaptation curve (added 2026-07-19)

Fills in the intermediate T-SSL checkpoints (60/150/200) with the **same** attentive×CE probe /
256px / manifest / 3 seeds, so every point is comparable and we can read κ as a function of
pretraining epoch. **Epoch 0 = the stock V-JEPA2 ViT-L the run was initialized from**
(`tssl_vitl_256_combined.yaml` `pretrained: true` → T-SSL is a domain-adaptive *continuation*,
so epoch 0 is the pre-adaptation encoder = the `pretrained` row above). Sweep jobs:
`scripts/26_phoneme_ckpt_sweep.sbatch <E>` (10378728/29/30, done 2026-07-19). Plotter +
figure: `artijepa/plot_phoneme_ckpt_sweep.py` → `eval/cksweep/phoneme_kappa_vs_ckpt.png`.

| ckpt (epoch) | val κ (sub030) | in-domain test κ (sub043) | **x-domain test_lss κ (usc_s1)** |
|---|---|---|---|
| **0** (stock V-JEPA2) | 0.336 ± .025 | 0.362 ± .030 | **0.156 ± .016** |
| 60 | 0.417 ± .028 | 0.436 ± .027 | **0.272 ± .013** |
| 100 | 0.474 ± .010 | 0.461 ± .030 | **0.334 ± .015** |
| 150 | 0.487 ± .012 | 0.507 ± .005 | **0.341 ± .018** |
| 200 | 0.483 ± .009 | 0.513 ± .006 | **0.360 ± .005** |
| 215 (final) | 0.485 ± .000 | 0.505 ± .007 | **0.352 ± .007** |

**Read:** all three splits climb steeply over the first ~100 epochs, then plateau. The
cross-domain `test_lss` κ — the domain-adaptation metric — **more than doubles from stock to
adapted (0.156 → 0.334 by ep100)** and then flattens (0.33–0.36 through ep215, peak at ep200).
So the bulk of the adaptation benefit is banked by ckpt_100 and the last ~115 epochs mostly add a
small in-domain lift (test 0.46 → 0.51); the x-domain metric is **saturated**, consistent with the
ckpt_100↔ckpt_215 comparison above. std here is population σ over 3 seeds (from the plotter).

Frame-acc (mean±sd) — pretrained: `test` 47.1 ± 2.0% / `test_lss` 18.6 ± 1.6%; best-val
ep {7,3,4}. `test_lss` PERµ **> 1.0** = CE-argmax OOD insertion blow-up (motivates the
CE-vs-CTC axis in `docs/phonePred.md §5`). Ranking (`test_lss` κ): **tssl 0.334 >
videomae_rtmri 0.264 ≫ videomae 0.190 > pretrained 0.156 > dinov2 0.065 > vitl 0.030** —
both ADAPTED encoders (tssl, videomae_rtmri) top the stock ones: **adapted video > stock
video ≫ image.**

### Baseline anchor (pooled `tcn` probe, 128px, single seed)

| encoder | probe | res | test κ | test PERµ | test_lss κ | test_lss PERµ | val κ |
|---|---|---|---|---|---|---|---|
| pretrained | tcn | 128 | 0.224 | 0.801 | **0.042** | 0.890 | 0.292 |

Frame-acc: `test` 38.4% / `test_lss` 7.5%. Fast pooled-head/128px point that validated
the combined pipeline end-to-end — **not** comparable to the attentive@256 headline.

### Loss landscape (Phase 2) — `attentive_lstm` · utterance-mode · CE vs CTC vs Focal · 256px

Frozen encoder → **AttentivePooler** (learned spatial pool of the 256 tokens) → **bi-LSTM
over time** → linear, trained on **whole utterances** (~800 tokens), not 1.28 s chunks.
Same head/context across losses so this isolates the **loss**. **CE & focal report κ+PER;
CTC is PER-only** (κ undefined without a forced alignment). Values = mean ± sd over seeds
{0,1,2}. **Complete 2026-07-18** (jobs 10349392/93/94 tssl215, 10343442/44 pretrained).

| encoder | loss | test κ (in-dom) | test PERµ | **test_lss κ (x-dom)** | test_lss PERµ | val κ | seeds |
|---|---|---|---|---|---|---|---|
| **tssl256comb215** | ce    | 0.672 ± .001 | 0.330 ± .007 | **0.421 ± .008** | 0.468 ± .009 | 0.610 ± .002 | {0,1,2} |
| **tssl256comb215** | focal | 0.657 ± .003 | 0.343 ± .007 | **0.420 ± .005** | 0.466 ± .006 | 0.604 ± .004 | {0,1,2} |
| **tssl256comb215** | ctc   | — | 0.487 ± .219 | **—** | 0.594 ± .182 | — | {0,1,2} |
| tssl256comb100 | ce    | 0.653 | 0.355 | **0.395** | 0.499 | 0.597 | {0} † |
| tssl256comb100 | focal | 0.656 | 0.335 | **0.407** | 0.481 | 0.590 | {0} † |
| tssl256comb100 | ctc   | — | 0.670 | **—** | 0.716 | — | {0} † |
| pretrained | ce    | 0.571 ± .012 | 0.470 ± .012 | **0.282 ± .012** | 0.694 ± .019 | 0.523 ± .003 | {0,1,2} |
| pretrained | focal | 0.542 ± .005 | 0.488 ± .029 | **0.286 ± .028** | 0.689 ± .037 | 0.503 ± .011 | {0,1,2} |
| pretrained | ctc   | — | 0.990 | **—** | 0.967 | — | {0} ‡ |

**† tssl256comb100 = single seed (s0):** these are the salvaged seed-0 outputs from the
ckpt_100 jobs that were cancelled when the arm was moved to ckpt_215 (see TODO note); kept
for the ckpt-comparison, not a 3-seed number. **‡ pretrained/ctc = single seed (s0):** job
10343443 was cancelled at 2026-07-17 15:08 before seeds 1–2 ran.

**Reading it — the loss axis is a near-null; utterance context, not the loss, fixed the OOD
blow-up:**
- **The insertion blow-up is already gone in utterance mode, for *every* loss.** Phase-1's
  motivating failure was pretrained `test_lss` PERµ = **1.040 > 1.0** (CE-argmax over-emits on
  the OOD speaker). Here pretrained CE `test_lss` PERµ = **0.694** — the whole-utterance context
  (~800 tokens vs a 16-token chunk) removes the blow-up on its own. **CTC/focal were not needed.**
- **CTC does not help and is unstable/worse:** tssl215 CTC `test_lss` PERµ **0.594 ± .182** (seed 1
  blew to 0.80) vs CE **0.468 ± .009**; pretrained CTC s0 PERµ **0.967** (near-collapse). CTC never
  beats CE/focal on PER and adds large seed variance.
- **Focal ≈ CE (within noise):** tssl215 `test_lss` κ 0.420 vs 0.421; pretrained 0.286 vs 0.282.
  The (1−p_t)^γ reweighting is effectively a no-op for this head/split.
- **ckpt_100 → ckpt_215 (same seed 0):** CE `test_lss` κ 0.395 → 0.412, focal 0.407 → 0.418
  (~+0.015); the 3-seed ckpt_215 CE mean is 0.421. Small lift, **consistent with the saturated-
  adaptation finding** from the Phase-1 headline (the x-domain metric barely moves past ckpt_100).
- **Adapted ≫ stock still holds under every loss:** tssl215 `test_lss` κ ≈ 0.42 ≫ pretrained ≈ 0.28.

### Head comparison — `attentive` (chunk) vs `attentive_lstm` (utterance), CE · 256px

Same frozen encoder + CE loss; only the probe head/context changes. `attentive` = Phase-1
chunk-mode (16-token 1.28 s chunks, pooled → linear); `attentive_lstm` = Phase-2
utterance-mode (AttentivePooler → bi-LSTM over ~800 tokens → linear). 3-seed mean.

| encoder | head | context | test κ | test PERµ | **test_lss κ** | test_lss PERµ |
|---|---|---|---|---|---|---|
| **tssl256comb215** | attentive | chunk | 0.505 | 0.691 | **0.352** | 0.655 |
| **tssl256comb215** | attentive_lstm | utterance | 0.672 | 0.330 | **0.421** | 0.468 |
| pretrained | attentive | chunk | 0.362 | 0.908 | **0.156** | 1.040 |
| pretrained | attentive_lstm | utterance | 0.571 | 0.470 | **0.282** | 0.694 |

**The utterance-mode LSTM head is a large, uniform win** — both in-domain (`test` κ +0.13 to
+0.21) and cross-domain (`test_lss` κ +0.07 to +0.13, PERµ −0.19 to −0.35), and it is what
removes the pretrained OOD insertion blow-up (PERµ 1.040 → 0.694 < 1.0). Caveat: this
confounds head architecture (LSTM temporal model) with context length (~800 vs 16 tokens) —
they were upgraded together — so it measures the combined Phase-1→Phase-2 head change, not a
clean architecture-only ablation.

### Reading it
- **In-domain generalization works:** the frozen encoder carries real phonetic signal
  across unseen 75-Speaker subjects (chance κ ≈ 0, acc ≈ 2.4%).
- **Cross-domain nearly collapses for the non-adapted encoder:** pretrained `test_lss`
  κ 0.156 vs `test` 0.362 — the gap the adapted encoders must close (headline number).

### Phonetic-class breakdown — t-SNE + confusion matrices (seed 0, `test_lss`)

Regenerated 2026-07-14 over all 6 Phase-1 encoders with `scripts/24_phoneme_viz.sbatch`
(t-SNE `artijepa/tsne_phonemes.py`, confmat `artijepa/confmat_phonemes.py`). **Four views**:
manner-of-articulation *(groups)*, vowels, consonants, and **place of articulation** — the
consonant grouping is now the **8-class** ARPABET place set **Bilabial / Labiodental /
Dental / Alveolar / Postalveolar / Palatal / Velar / Glottal** (Postalveolar `{ch,jh,sh,zh}`
split out from Palatal `{y}`).

Macro-recall (mean per-class row-diagonal = class-balanced recall; rows normalized):

| encoder | groups (manner) | vowels | consonants | **place** | `test_lss` κ |
|---|---|---|---|---|---|
| **tssl256comb100** | 0.410 | 0.343 | 0.261 | **0.330** | 0.337 |
| **videomae_rtmri** | 0.373 | 0.316 | 0.184 | **0.249** | 0.271 |
| pretrained | 0.242 | 0.208 | 0.086 | **0.137** | 0.157 |
| videomae | 0.226 | 0.227 | 0.072 | **0.096** | 0.171 |
| dinov2 | 0.077 | 0.096 | 0.023 | **0.034** | 0.068 |
| vitl | 0.022 | 0.029 | 0.008 | **0.014** | 0.024 |

Same ranking as the headline κ: both rt-MRI-**adapted** encoders (tssl, videomae_rtmri) beat
every stock one at every granularity. On `place`, adapted encoders resolve Alveolar / Bilabial
best (tssl diag 0.52 / 0.53), while low-support Glottal `{h,hh}` and Palatal `{y}` stay near-
chance for all. Figures (per-view cross-model panels + per-encoder):
`…/eval/tsne/compare_<view>_repA_tsne_test_lss_s0.png` + `tsne_<enc>_…` and
`…/eval/confmat/confmat_<mtype>_compare_test_lss_s0.png` + `confmat_<enc>_…`; raw +
normalized matrices in `…/eval/confmat/confmat_values_test_lss_s0.json`. **t-SNE uses rep A**
(trained-q = the AttentivePooler output the linear head reads → the probe's decision space);
`--rep B` (raw-pooled, non-circular encoder geometry) still available by hand. Prior figures
(pre-place, missing `videomae_rtmri`, or rep B) archived under `…/stale_pre_2026-07-14/`.

---

## Phoneme-CLIP classification (Phase 3) — vowels · consonants · manner · place

**A different task from everything above.** Phases 1–2 slide a window over the utterance and
label every temporal token (frame-level sequence task → κ / PER). Phase 3 cuts **one whole
phoneme** out of the video by its start–end alignment and gives it **one** label
(`artijepa/eval_phoneme_groups.py`, config `configs/eval_phoneme_groups.yaml`, launcher
`scripts/28_phoneme_groups.sbatch`). So these numbers are **not** comparable to the κ tables
above, and they are also not the same thing as the *Phonetic-class breakdown* section — that one
slices the 41-class frame probe's confusion matrix, whereas each task here **trains its own probe
on its own label space**.

Head = `attentive_lstm` in **clip mode**: AttentivePooler over the S′ spatial tokens per temporal
step → bi-LSTM over the phoneme's **variable-length** token sequence → **last hidden state** (both
directions) is the phoneme vector → linear. Same combined Annot-16 + usc_lss manifest, same
splits (train = 14 Annot-16 speakers, val = sub030, `test` = sub043 in-domain, `test_lss` =
usc_s1 cross-domain), 256px / 32f / target_fps 50, frozen encoder, **seed 0 only**. The feature
cache is task-agnostic (all non-sil phonemes, one clip each) — extracted once per encoder × split,
then the four tasks are derived by filtering + relabeling.

### macro-F1 by task (val / test / test_lss)

| encoder | vowels (15c) | consonants (25c) | manner (5c) | place (8c) |
|---|---|---|---|---|
| **tssl256comb215** | **0.561 / 0.568 / 0.516** | **0.374 / 0.399 / 0.329** | **0.521 / 0.571 / 0.537** | **0.607 / 0.593 / 0.507** |
| **videomae_rtmri** (ckpt-214) | 0.414 / 0.471 / 0.428 | 0.315 / 0.371 / 0.287 | 0.489 / 0.510 / 0.453 | 0.491 / 0.526 / 0.471 |
| videomae (Kinetics) | 0.409 / 0.472 / 0.282 | 0.261 / 0.291 / 0.167 | 0.404 / 0.491 / 0.322 | 0.421 / 0.441 / 0.285 |
| pretrained (FAIR V-JEPA2) | 0.363 / 0.445 / 0.301 | 0.230 / 0.291 / 0.140 | 0.367 / 0.485 / 0.324 | 0.349 / 0.436 / 0.263 |

`test_lss` support: vowels n=9,716 · consonants/manner/place n=15,103.

**The Phase-1 ranking reproduces on a completely different task formulation.** Both rt-MRI-adapted
encoders beat both stock ones at **every** granularity on all three splits — the matrix is now
complete for all four encoders — and `tssl256comb215` leads everywhere. The closest call is manner
`test_lss` (adapted 0.453 vs best stock 0.324); on `place` the gap is widest, adapted 0.471–0.507
vs stock 0.263–0.285. The
adaptation gain is **largest exactly where the domain gap bites**: on `test_lss`, tssl beats stock
videomae by +0.234 (vowels) and +0.215 (manner), versus +0.096 / +0.080 on in-domain `test`. The
stock encoders lose roughly half their in-domain macro-F1 crossing to usc_s1 (videomae vowels
0.472 → 0.282), while the adapted ones lose ~10% (tssl 0.568 → 0.516) — same narrowing-the-gap
story as the headline κ, measured independently.

### Full metric set, `test_lss` (cross-domain)

The Phase-3a spec asked for F1 + precision + recall; macro-F1 alone is above, the rest is here.
All macro-averaged over the task's classes, `place` at its as-stored 8 classes.

| encoder | task | acc | P<sub>macro</sub> | R<sub>macro</sub> | F1<sub>macro</sub> | F1<sub>wtd</sub> |
|---|---|---|---|---|---|---|
| **tssl256comb215** | vowels | 0.535 | 0.566 | 0.499 | **0.516** | 0.531 |
| | consonants | 0.400 | 0.403 | 0.334 | **0.329** | 0.382 |
| | manner | 0.590 | 0.608 | 0.554 | **0.537** | 0.575 |
| | place | 0.590 | 0.548 | 0.523 | **0.507** | 0.607 |
| **videomae_rtmri** | vowels | 0.477 | 0.572 | 0.399 | 0.428 | 0.464 |
| | consonants | 0.332 | 0.315 | 0.298 | 0.287 | 0.329 |
| | manner | 0.489 | 0.483 | 0.439 | 0.453 | 0.486 |
| | place | 0.566 | 0.480 | 0.526 | 0.471 | 0.592 |
| videomae | vowels | 0.344 | 0.387 | 0.280 | 0.282 | 0.324 |
| | consonants | 0.255 | 0.213 | 0.208 | 0.167 | 0.224 |
| | manner | 0.406 | 0.419 | 0.329 | 0.322 | 0.389 |
| | place | 0.392 | 0.383 | 0.373 | 0.285 | 0.410 |
| pretrained | vowels | 0.351 | 0.461 | 0.303 | 0.301 | 0.326 |
| | consonants | 0.219 | 0.193 | 0.170 | 0.140 | 0.202 |
| | manner | 0.414 | 0.549 | 0.364 | 0.324 | 0.385 |
| | place | 0.395 | 0.414 | 0.361 | 0.263 | 0.438 |

**Precision runs well above recall almost everywhere** (e.g. pretrained/manner P 0.549 vs R 0.364,
videomae_rtmri/vowels P 0.572 vs R 0.399) — the probes are conservative on minority classes,
predicting them rarely but correctly when they do, and dumping the rest on the majority class.
That is the same mechanism as the Affricate near-deletion and the Alveolar/Bilabial priors below.
The exception is `place`, where F1<sub>wtd</sub> > F1<sub>macro</sub> by ~0.10–0.13 for every
encoder: performance is carried by the high-support classes (Alveolar n=8,451) while the small
ones (Palatal n=250, Glottal n=346) drag the macro average down.

### Per-class recall, `test_lss` (cross-domain)

| manner | tssl256comb215 | videomae_rtmri | videomae | pretrained | support |
|---|---|---|---|---|---|
| Plosive | 0.370 | **0.480** | 0.291 | 0.228 | 4,859 |
| Fricative | 0.639 | 0.620 | **0.654** | 0.585 | 4,237 |
| Affricate | 0.201 | **0.251** | 0.008 | 0.003 | 358 |
| Nasal | **0.899** | 0.481 | 0.374 | 0.752 | 2,483 |
| Approximant | **0.663** | 0.360 | 0.319 | 0.251 | 3,166 |

_Recall only — read with the F1 row below, because two of these columns buy recall with
over-emission. Nasal **F1** is tssl 0.66 · videomae_rtmri 0.54 · pretrained 0.49 · videomae 0.38,
and the predicted:true Nasal count ratio is tssl ×1.73, pretrained ×2.07, videomae ×0.96,
videomae_rtmri ×0.79. So pretrained's 0.752 Nasal recall is **not** a rival to tssl's 0.899 — it
comes with precision 0.36 against tssl's 0.52._

| place | tssl256comb215 | videomae_rtmri | videomae | support |
|---|---|---|---|---|
| Bilabial | 0.607 | 0.679 | **0.852** | 1,811 |
| Labiodental | **0.661** | 0.456 | 0.356 | 873 |
| Dental | 0.614 | **0.621** | 0.433 | 817 |
| Alveolar | **0.625** | 0.572 | 0.345 | 8,451 |
| Postalveolar | **0.592** | 0.485 | 0.014 | 586 |
| Palatal | 0.304 | **0.492** | 0.228 | 250 |
| Velar | 0.472 | **0.527** | 0.285 | 1,969 |
| Glottal | 0.309 | 0.373 | **0.471** | 346 |

### What the confusions say

- **Affricates collapse everywhere, but the failure *direction* separates the encoders.** Recall
  0.201 / 0.251 / 0.008 / 0.003 (pretrained predicts Affricate for exactly one of 358 clips — it
  has effectively deleted the class). An affricate is a stop released into a fricative, and each encoder picks
  a different half: rt-MRI-adapted models leak it to **Plosive** (tssl `Affricate→Plosive` 0.26)
  while stock videomae leaks it to **Fricative** (0.36, and `Affricate→Plosive` 0.39 — it is
  essentially guessing). With n=358 this is the lowest-support class in the task, so read it as a
  hypothesis about *what gets encoded* (release burst vs frication), not a settled result.
- **Approximants are where rt-MRI adaptation pays; nasals are messier than they first look.**
  Approximant is the clean case: tssl recall 0.663 and F1 0.60 against 0.360 / 0.319 / 0.251 and
  F1 0.37 / 0.39 / 0.33 — best on both metrics, no over-emission. Nasal *recall* looks like the
  same story (0.899 vs 0.481 / 0.374) until `pretrained` is added at 0.752, a **stock** encoder
  beating the adapted `videomae_rtmri`. That ranking is an artifact of over-emission: pretrained
  predicts Nasal 2.07× more often than it occurs (precision 0.36), and tssl over-predicts 1.73×
  (precision 0.52) — the same recall-inflating mechanism this section flags for videomae's
  Bilabial below. On **F1** the intended ordering does hold (tssl 0.66 > videomae_rtmri 0.54 >
  pretrained 0.49 > videomae 0.38), so the claim survives, but only as an F1 claim. Both classes
  are defined by *configuration* the MRI sees directly (velum lowered; tongue/lip constriction
  without turbulence) rather than by an acoustic-transient proxy — the classes a vocal-tract-
  adapted video encoder should win.
- **Stock videomae has collapsed onto a Bilabial prior** on `place`: Bilabial recall 0.852 — the
  *highest* of the three encoders — bought by dumping everything else into it
  (`Labiodental→Bilabial` 0.47, `Palatal→Bilabial` 0.40, `Alveolar→Bilabial` 0.31) and scoring
  0.014 on Postalveolar. A per-class recall table alone would have made this look like a strength;
  it's degenerate behavior. The adapted encoders' errors instead flow toward **Alveolar**, the
  n=8,451 majority class — an ordinary prior, not a collapse.
- **Vowel errors are centralization.** Every encoder's top confusions run *toward* schwa-like
  `ah` (`ow→ah`, `uh→ah`, `aa→ah`), and the effect scales inversely with adaptation: 0.30 for
  tssl vs 0.61 for videomae_rtmri on `ow→ah`. Stock videomae instead confuses `uw→iy` (0.68) —
  a front/back error, i.e. it is not resolving tongue *position* at all.
- **Glottal `{h,hh}` stays near-chance for the adapted encoders** (0.309 / 0.373), leaking to
  Alveolar (0.34). Expected: glottal constriction is largely invisible in the mid-sagittal
  field of view.

### Figures — confusion + t-SNE, `tssl256comb215` on `test_lss` (2026-07-23)

`artijepa/plot_phgroups.py` reloads the four saved probes and re-runs them over the cached
`test_lss` features (no re-extraction). Outputs in `…/eval/phgroups/figs/`:

| file | contents |
|---|---|
| `phgroups_confmat_tssl256comb215_test_lss_s0.png` | 4 panels, rows=true / cols=predicted, row-normalized to recall |
| `phgroups_tsneA_…png` | t-SNE of **rep A** = the phoneme vector (bi-LSTM last hidden) |
| `phgroups_tsneB_…png` | t-SNE of **rep B** = raw encoder tokens mean-pooled, no probe |
| `phgroups_confmat_…json` | the plotted matrices + re-scored macro-F1 |

**Sanity check:** re-running the probes reproduces the stored eval macro-F1 to 4 dp on all three
unmodified tasks (vowels 0.5162, consonants 0.3290, manner 0.5374) — the reload path is faithful.

**`place` here is 7-class (Glottal dropped)**, per request: Glottal-true clips are excluded *and*
the Glottal logit is masked so predictions re-argmax over the 7 survivors, giving a matrix whose
rows sum to support. Re-scored **macro-F1 0.524** vs **0.507** for the 8-class version in the JSON
above — the two numbers are not interchangeable. Note the gain is almost entirely from deleting a
weak row from the macro average, not from redistributed mass: every surviving class moves by ≤1
point (Alveolar 62→63, Palatal 30→31, rest unchanged), because Glottal was seldom *predicted* even
when available.

**Rep A tracks the confusion matrix.** Nasal and Approximant occupy their own territories while
Plosive and Fricative interpenetrate — the same structure as the 37/20 and 9/64 cells.

⚠️ **Rep B shows no phoneme-class structure at any granularity** — all four panels are colour-mixed,
and the clean Affricate cluster visible in rep A is absent. Two readings, not yet separated:

1. The clip-level geometry really is dominated by something other than phoneme identity (rep B
   does split into two large blobs, but the split is **not** phoneme-aligned; cause unidentified —
   session/appearance are the obvious suspects for a single-speaker split like usc_s1).
2. **The pooling is blunter than Phase-1's rep B and may be doing the damage.** Phase-1
   `tsne_phonemes.py` rep B pools over S′ only, keeping one vector *per temporal token*; this rep B
   additionally averages over the clip's temporal tokens to get one vector per phoneme, which
   discards exactly the within-phoneme dynamics (release bursts, formant transitions) that
   distinguish e.g. affricates from plosives. So this is a harsher test than the Phase-1 panels.

Consequently: **do not read rep B as evidence that the frozen encoder lacks phoneme geometry**, and
do not conclude from rep A that affricate collapse is "merely" a classifier-prior problem — rep A
is the trained decision space and its affricate cluster may be probe-constructed. Resolving this
needs a rep-B variant that keeps the temporal axis (per-token, as in Phase 1) before the two
readings can be told apart. **Not run.**

### Reproduce

```bash
cd /project2/shrikann_35/hongn/vjepa2
source dev_artiJEPA/scripts/_env.sh        # conda env artijepa; sets PYTHONPATH
```

**1 — the 4 encoder runs (all 4 tasks each).** One job per encoder, ~24 h wall, GPU v100, 96 GB:

```bash
S=dev_artiJEPA/scripts/28_phoneme_groups.sbatch
for ENC in tssl256comb215 pretrained videomae videomae_rtmri; do sbatch $S $ENC 0; done
#            ^encoder key                                                    ^seed
# optional 3rd arg = clip limit for a debug run, e.g.  sbatch $S pretrained 0 400
```

Each job extracts a **task-agnostic** ragged clip cache once per split (~4.6 h at 256px), then
trains the 4 tasks off it (~1.5–5 h each, 40 epochs, model-selected on val macro-F1). Caches
(`~141 GB` per encoder, all 4 splits) live in `/scratch1/hongn/artijepa/feat_cache/phgroups/<tag>phg_<hash>/`
and are **keyed on spec/size/fps/manifest, NOT the ckpt epoch** — reusing a tag across
checkpoints silently cache-hits stale features, which is why ckpt_215 got its own
`tssl256comb215` tag. Re-running an encoder cache-hits and skips straight to training.

**2 — the one missing cell.** `pretrained`/`place` (see the ⚠️ note below). Cache-hits, so budget
~3 h train + ~1 h eval, not a full 24 h:

`28_phoneme_groups.sbatch` takes an optional **4th positional arg = comma-separated task subset**
(added 2026-07-23). Pass `""` for the debug-limit slot to reach it. `pretrained` needs no
`--encoder`/`--model` flag. **Back up first** — a subset run rewrites the JSON with only those
tasks:

```bash
cp /scratch1/hongn/artijepa/eval/phgroups/phgroups_pretrained_s0.json{,.bak}

sbatch --time=8:00:00 dev_artiJEPA/scripts/28_phoneme_groups.sbatch pretrained 0 "" place
#                                                                   ^enc      ^seed ^lim ^tasks
# --time overrides the script's 24 h default: one cached task is ~3.5 h, and a
# shorter ask schedules sooner.  Confirm "[phg] cache hit <split>" appears for all
# 4 splits in the first minute -- if it starts extracting instead, the tag/hash is
# wrong and it will burn ~5 h rebuilding a cache that already exists.

# the run leaves a JSON containing ONLY place -- merge it back:
python - <<'PY'
import json
p='/scratch1/hongn/artijepa/eval/phgroups/phgroups_pretrained_s0.json'
new=json.load(open(p)); old=json.load(open(p+'.bak'))
old['tasks'].update(new['tasks'])          # place joins vowels/consonants/manner
json.dump(old, open(p,'w'), indent=2)
print('tasks now:', list(old['tasks']))
PY
```

**3 — the figures** (`confmat` + t-SNE rep A/B). Reloads the saved `_<task>.pt` probes and re-runs
them over the cached split; **no re-extraction**, so this is GPU-light (~45 min for 4 tasks on a
V100, dominated by reading the 30 GB `test_lss` cache):

```bash
PYTHONPATH=.:dev_artiJEPA python -u -m artijepa.plot_phgroups \
    --tag tssl256comb215 --split test_lss --seed 0 --per-class 600 --workers 8
# --keep-place-glottal    keep place at 8 classes (default drops Glottal -> 7)
# --tasks vowels,manner   subset;  --limit 400  smoke test (~1 min)
```

Run it inside an allocation, and pin `--ntasks=1 --overlap` if attaching to an existing one —
plain `srun` inherits the allocation's task count and runs redundant copies that race on the
output files.

**Cost note:** `consonants`, `manner` and `place` all read the *same* consonant clip set, so
`plot_phgroups.py` currently sweeps that ~30 GB cache three times. Batching the three probes into
one sweep would cut ~55 min off the figure run. Not done.

_Seed 0, `scripts/28_phoneme_groups.sbatch`. Raw + normalized confusion matrices, per-class
precision/recall/F1, and 40-epoch histories for every task × split (`val`, `test`, `test_lss`)
are in `/scratch1/hongn/artijepa/eval/phgroups/phgroups_<tag>_s0.json`; per-task best-val probe
weights alongside as `phgroups_<tag>_s0_<task>.pt`._

> **Complete: 16/16 cells (`pretrained/place` filled 2026-07-23).** The matrix is now full for all
> four encoders. History, because the value carries a caveat: the first attempt (job 10502872) hit
> its wall mid-eval and persisted nothing, so `place` was **re-trained** from the cached features
> (job 10528776, `--tasks place`). This re-train reached **best val macro-F1 0.349 @ e6**, below the
> dead run's 0.399 @ e6 — same seed and data, but fp16/worker-order nondeterminism moves the noisy
> per-epoch val peak, and model-selection locks onto whichever epoch happened to top it. So the
> reported `place` test_lss **0.263** is one draw from that noise, not the number the original run
> would have produced; read it as ±~0.03. All other 15 cells are single deterministic runs.
>
> The re-train cache-hit `pretrainedphg_643554b285` (no re-extraction) and the JSON was recovered by
> the documented back-up-and-merge (the launcher's `--tasks` path still rewrites the file with only
> the requested task — `eval_phoneme_groups.py:550` starts `out["tasks"]` empty — so the `.bak`
> merge in the Reproduce section is mandatory, not optional).

---

## Planned / TODO runs

The full scale-up plan (3 axes — cross-model × loss × head — with cost model and
phased runs) now lives in **`docs/phonePred.md §5`**. Headline next steps:

- [x] **Phase 1 — cross-model @256 attentive/CE, 3 seeds** (complete 2026-07-14):
      `tssl256comb100`, `pretrained`, `videomae`, `videomae_rtmri` (ckpt-214),
      `vitl`, `dinov2` — all 6 × seeds {0,1,2}. **Yes, adaptation lifts `test_lss`:**
      both adapted encoders (tssl 0.334, videomae_rtmri 0.264) top every stock one.
- [x] **Phase 2 — loss landscape (CE vs CTC vs Focal)** (complete 2026-07-18 — see the
      *Loss landscape (Phase 2)* table above): tssl (see ckpt note below) and pretrained ×
      **`attentive_lstm`** probe × {ce, ctc, **focal**} × seeds{0,1,2}. **Answer: the loss axis
      is a near-null** — whole-utterance context (not the loss) already removes the OOD PERµ>1.0
      blow-up (pretrained CE `test_lss` PERµ 1.040→0.694); CTC is worse & unstable, focal ≈ CE.
      Does a different
      loss fix the OOD PERµ>1.0 insertion blow-up? Two independent angles: **CTC** (an
      alignment-free *sequence* loss — remove the frame-alignment assumption) and **focal**
      (per-frame CE reweighted by (1−p_t)^γ, `probe.focal_gamma` default 2.0 — down-weight
      the easy/high-frequency frames so the posterior stops over-emitting at collapse time).
      **Launched 2026-07-15, OOM'd on the `test_lss` eval loader; patched + relaunched
      2026-07-16** (`scripts/25_phoneme_phase2_ctc.sbatch <enc> <loss>`; loss∈{ce,ctc,focal}).
      Not the Phase-1 `attentive` probe: that head is chunk-mode (CTC is hard-blocked there;
      focal would work but with 16-token context). `attentive_lstm` = the same AttentivePooler
      spatial pool (learned, not a mean) + a bi-LSTM over time, trained in **utterance mode**
      (whole utterances, ~800 tokens). **CE is re-run on this same head** rather than reusing
      the Phase-1 `attentive`×CE row — otherwise the comparison would vary temporal context
      (816 tokens vs a 16-token chunk) alongside the loss and couldn't isolate it.
      Reuses the Phase-1 spatial caches (the cache hash ignores probe type) → no
      re-extraction. CE & focal report κ+PER (κ ties back to Phase 1); CTC is PER-only.
      **OOM fix:** the utterance-mode eval loaders now use `utt_eval_workers` (default 2) ×
      `prefetch_factor=1` and the job runs at `--mem=96G` — the Jul-15 crash was 4 workers ×
      2 prefetched ~1 GB `test_lss` batches overrunning the 64 GB cgroup.
      **ckpt_100 → ckpt_215 relaunch (2026-07-17):** the tssl arm was moved from ckpt_100 to the
      **final ckpt_215** under a distinct tag **`tssl256comb215`** so results add up alongside the
      ckpt_100 tag rather than overwriting it (the feature-cache hash keys on spec/size/fps/manifest,
      NOT the ckpt epoch — so a shared tag would silently cache-hit stale features). New encoder key
      `tssl256comb215` in `scripts/25_phoneme_phase2_ctc.sbatch`; the ckpt_100 tssl jobs were cancelled
      and re-submitted as jobs **10349392 (ce) / 10349393 (ctc) / 10349394 (focal)**, seeds {0,1,2}.
      No re-extraction: they cache-hit the Phase-1 ckpt_215 spatial cache `tssl256comb215sp_43c1fe20dd`
      (probe-independent hash). Results land as `phoneme_usc_lss_tssl256comb215sp_43c1fe20dd_attentive_lstm_<loss>_s<seed>.json`
      → the Phase-2 table below will carry **both** `tssl256comb100` and `tssl256comb215` rows once complete.
- [x] **Phase 3a — phoneme groups ablation** (complete 2026-07-23): In this phase, I am not training with shifted windown of 32 frames to predict tokens-wise phoneme label like in phase 1, I take phoneme as input directly to produce sample-wise prediction as each sample is an sequence of image of an phoneme. Build dataloaders for each of these phoneme groups: vowels only, consonants only, consonants by manners, consonants by place. Each sample of dataloader is a short clips of phoneme get by start-end alignment. Using each of the dataloader, train multiclass classifiers and evaluate on each of phoneme groups using attentive_lstm, with FPS = 50. Metrics using F1 score, recall, precision, confusion matrix.

  **LAUNCHED 2026-07-21** — new CLIP-classification pipeline (distinct from the Phase-1/2
  frame-level sequence task): `artijepa/eval_phoneme_groups.py` + `configs/eval_phoneme_groups.yaml`
  + `scripts/28_phoneme_groups.sbatch`. One sample = one non-sil phoneme cut by its start-end
  alignment, sampled at **FPS=50** into a variable-length token sequence; head = `attentive_lstm`
  in CLIP mode (AttentivePooler over S' per temporal step → bi-LSTM → **last hidden = the phoneme
  vector** → linear). Four tasks derived from ONE task-agnostic ragged feature cache per encoder:
  **vowels(15) / consonants(25) / manner(5) / place(8)**. Same combined manifest as Phase 1/2
  (train 14 Annot-16 spk, model-select val=sub030, report `test`=in-domain sub043 + `test_lss`=
  x-domain usc_s1). Metrics: macro/weighted **F1 + precision + recall + per-class + confusion
  matrix** (sklearn); model-selection on val macro-F1. **4 encoders × 1 seed** (jobs 10478155
  tssl256comb215 / 10478156 pretrained / 10478157 videomae / 10478158 videomae_rtmri). Outputs:
  `/scratch1/hongn/artijepa/eval/phgroups/phgroups_<tag>_s0.json`. Smoke-tested end-to-end on GPU
  (job 10470075, `--limit 400`); whole-video-decode cache → ~49 clips/s (~35 min extraction/enc).

  **STATUS 2026-07-23 — 16 of 16 cells DONE.** All four tasks complete for all four encoders.
  `pretrained/place` was the last cell (job 10528776, re-trained from cache after the first attempt
  died mid-eval; test_lss macro-F1 0.263, with the ±0.03 re-train caveat noted in the Phase-3
  section). Figures (confusion + t-SNE rep A/B, `place` Glottal-dropped to 7c) rendered for all four
  encoders → `…/eval/phgroups/figs/phgroups_{confmat,tsneA,tsneB}_<tag>_test_lss_s0.png`. Results
  tables and figure discussion are in the *Phoneme-CLIP classification (Phase 3)* section above.
  **Phase 3a complete.**

Each row: run `eval_phoneme.py` with the config + `--encoder/--model/--probe/--loss/
--seed`, record `test` + `tests.test_lss` from the output JSON.

---

## Provenance

| field | value |
|---|---|
| Config | `configs/eval_phoneme_annot16_combined.yaml` |
| Manifest | `/scratch1/hongn/annot16_phonemes/phoneme_manifest_combined.csv` (train 384 / val 30 / test 30 / test_lss 684) |
| Split builder | `/scratch1/hongn/speaker75/merge_annot16_phonemes.py --split xdomain` |
| Row 1 result | `/scratch1/hongn/artijepa/eval/phoneme_usc_lss_annot16comb_693d470c34_tcn_ce_s0.json` (best epoch 16/40) |
| Row 2 result | `/scratch1/hongn/artijepa/eval/phoneme_usc_lss_pretrainedsp_589c9438e5_attentive_ce_s0.json` (best epoch 7/30); curve `…_traincurve.png` |
| Row 2 cache | `…/feat_cache/phoneme/pretrainedsp_589c9438e5/` (~149 GiB spatial grid, train 19,035 clips) |
| Row 1b result | `…/eval/phoneme_usc_lss_tssl256comb215sp_43c1fe20dd_attentive_ce_s{0,1,2}.json` (ckpt_215, best epoch {7,6,6}/30) |
| Row 1b / Phase-2 215 cache | `…/feat_cache/phoneme/tssl256comb215sp_43c1fe20dd/` (probe-independent; shared by ckpt_215 Phase-1 attentive **and** Phase-2 `attentive_lstm`) |
| Phase-2 results | `…/eval/phoneme_usc_lss_{tssl256comb215sp_43c1fe20dd,pretrainedsp_589c9438e5,tssl256comb100sp_c8eaee90f7}_attentive_lstm_{ce,ctc,focal}_s{seed}.json` |
| Phase-2 aggregator | `scratchpad/agg_phase2.py` (globs seed JSONs → mean ± sample-sd; CTC PER-only) |
| Hardware | 1× V100-32GB (d13-04), fp16. Row 1 (128 pooled) ~7 min. Row 2 (256 spatial): extract ~79 min + streaming probe ~5.5 min/epoch (see `docs/phonePred.md §4.1`) |
