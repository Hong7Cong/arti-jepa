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
Launched 2026-07-12; **filling in as jobs complete.**

| # | encoder | pretraining | adapt | test κ (in-dom) | test PERµ | **test_lss κ (x-dom)** | test_lss PERµ | val κ |
|---|---|---|---|---|---|---|---|---|
| 1 | **tssl256comb100** | T-SSL 75-spk +longitudinal (ckpt_100) | ✅ | ~0.44 ⏳2/3 | ~0.71 | **~0.33 ⏳2/3** | ~0.68 | ~0.48 |
| 2 | pretrained | V-JEPA2 stock | — | 0.362 ± .030 | 0.908 ± .094 | **0.156 ± .016** | 1.040 ± .056 | 0.336 ± .025 |
| 3 | videomae | VideoMAE K400 (3-D video SSL) | — | 0.390 ⏳1/3 | 0.933 | **0.171 ⏳1/3** | 0.867 | 0.373 |
| 4 | vitl | ImageNet-sup ViT-L (2-D) | — | 0.202 ± .072 | 0.824 ± .032 | **0.030 ± .005** | 0.926 ± .004 | 0.336 ± .004 |
| 5 | dinov2 | DINOv2 ViT-L (2-D image SSL) | — | 0.258 ± .020 | 0.820 ± .018 | **0.065 ± .015** | 0.888 ± .018 | 0.246 ± .031 |
| 6 | videomae_rtmri | rt-MRI VideoMAE (ckpt-214) | ✅ | 0.409 ± .020 | 0.744 ± .018 | **0.264 ± .013** | 0.838 ± .016 | 0.432 ± .003 |

**⏳ N/3 = provisional** (partial-seed mean; the missing seeds timed out at the 20 h wall
and were resubmitted as 1-seed jobs — tssl s2, videomae s1+s2 — wall now 30 h). tssl from
s0/s1 (`test_lss` κ 0.337/0.315); videomae from s0 only. Final 3-seed mean±sd replaces
these on completion.

Frame-acc (mean±sd) — pretrained: `test` 47.1 ± 2.0% / `test_lss` 18.6 ± 1.6%; best-val
ep {7,3,4}. `test_lss` PERµ **> 1.0** = CE-argmax OOD insertion blow-up (motivates the
CE-vs-CTC axis in `docs/phonePred.md §5`). Ranking so far (`test_lss` κ): **tssl ≈0.33 >
videomae_rtmri 0.264 ≫ videomae ~0.17 ≈ pretrained 0.156 > dinov2 0.065 > vitl 0.030** —
both ADAPTED encoders (tssl, videomae_rtmri) top the stock ones: **adapted video > stock
video ≫ image.**

### Baseline anchor (pooled `tcn` probe, 128px, single seed)

| encoder | probe | res | test κ | test PERµ | test_lss κ | test_lss PERµ | val κ |
|---|---|---|---|---|---|---|---|
| pretrained | tcn | 128 | 0.224 | 0.801 | **0.042** | 0.890 | 0.292 |

Frame-acc: `test` 38.4% / `test_lss` 7.5%. Fast pooled-head/128px point that validated
the combined pipeline end-to-end — **not** comparable to the attentive@256 headline.

### Reading it
- **In-domain generalization works:** the frozen encoder carries real phonetic signal
  across unseen 75-Speaker subjects (chance κ ≈ 0, acc ≈ 2.4%).
- **Cross-domain nearly collapses for the non-adapted encoder:** pretrained `test_lss`
  κ 0.156 vs `test` 0.362 — the gap the adapted encoders must close (headline number).

---

## Planned / TODO runs

The full scale-up plan (3 axes — cross-model × loss × head — with cost model and
phased runs) now lives in **`docs/phonePred.md §5`**. Headline next steps:

- [ ] **Phase 1 — cross-model @256 attentive/CE, 3 seeds:** `tssl_vitl_256_combined`,
      `pretrained` (done, row 2 = s0), `videomae`, rt-MRI VideoMAE (ckpt-214),
      `vitl`, `dinov2`. Does adaptation lift **test_lss**?
- [ ] **Phase 2 — CE vs CTC:** top encoders × pooled `tcn`/`lstm` × {ce, ctc} — does
      CTC fix the OOD PERµ>1.0 insertion blow-up?
- [ ] **Phase 3 — head ablation:** best encoder × all 7 heads (pooled + spatial).

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
| Hardware | 1× V100-32GB (d13-04), fp16. Row 1 (128 pooled) ~7 min. Row 2 (256 spatial): extract ~79 min + streaming probe ~5.5 min/epoch (see `docs/phonePred.md §4.1`) |
