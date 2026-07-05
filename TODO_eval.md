# Arti-JEPA — TODO: Evaluation

Evaluate frozen encoder features (our T-SSL checkpoints + public image models) on
**phoneme prediction** from the silent rtMRI video. Metrics: frame-level
**Cohen's κ** + **PER**. Completed numbers live in `RESULTS.md`; pretraining plans
in `TODO_pretraining.md`; file map in `Master.md`.

**Eval task pivot (2026-06-07):** the weak stimulus-group classification was
**removed** (clip-type says nothing about articulation). Label source:
- **Task 2 — gold** phonemes w/ timestamps for one **OOD** speaker
  (`/scratch1/hongn/usc_lss`, 104×104 @ 99 fps, 684 utts, 41 ARPABET).

**Task-1 pseudo phonemes removed (2026-06-30):** the audio-model (wav2vec2/WavLM
CTC) pseudo-label route on the 75-speaker corpus's paired audio was dropped —
superseded by the new downstream tasks below (stuttering/gloss classification,
articulatory-condition JEPA, AAI). `audio_phoneme.py`,
`configs/eval_phoneme_pseudo.yaml` are now dead for eval.

**Alignment** is in **seconds** (frame-rate agnostic): clips resampled to
target_fps (50); 1 JEPA token = tubelet 2 → 80 ms. The 99-fps OOD speaker needs
no special-casing.

**Eval infra (DONE):** `eval_phoneme.py` (freeze encoder → per-temporal-token
features → per-token probe → κ + PER), `phonemes.py` (41-ARPABET, seconds-based
alignment, CTC-collapse, PER, κ), `usc_lss.py` (OOD manifest+dataset),
`baselines.py` (public image models).
Probe heads: `linear`/`mlp`/`tcn`/`lstm`/`transformer` (CE/CTC) +
`tcn_spatial`/`attentive` (spatial-aware, the winners).

---

## ▶ HIGH PRIORITY

### 1. Eval the combined (+longitudinal) 256px checkpoint — **TBA (blocked on training)**
The longitudinal-lift headline. Once `runs/tssl_vitl_256_combined/latest.pt`
finishes (see `TODO_pretraining.md`), run the 256px spatial-probe eval vs the
75-only `runs/tssl_vitl_256` checkpoint.
```bash
source dev_artiJEPA/scripts/_env.sh
# combined encoder, attentive spatial probe (use a FRESH tag to avoid cache collision):
bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml \
     --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/latest.pt --tag tssl256comb
```
- Compare: `tssl256comb` vs `tssl256` (75-only @e50, κ 0.527±0.004) vs
  `pretrained256` (0.449±0.005). **Result rows in `RESULTS.md` = TBA.**
- ⚠ **Cache-key caveat:** `eval_phoneme._tag()` omits the checkpoint path from the
  cache hash → ALWAYS use a fresh `--tag` per encoder or `rm` the stale
  `feat_cache/phoneme/<tag>*` first, else it silently re-reads another encoder's
  features.

### 2. Finish the fair-fight image baselines (spatial probe) — **PARTIAL, RESUME**
The decisive keystone check. Status (2026-06-17): the run died mid `base_siglip |
attentive | seed 1` on GPU de-allocation. **Missing: supervised ViT-L/16 (the key
competitor, 0/3), dinov2 (0/3), resnet (0/3), siglip-attentive s1/s2.** Done so
far is in `RESULTS.md`.
- **Decisive question:** does sup ViT-L/16 + spatial probe stay below tssl_256 +
  spatial probe (0.527)? It was the one competitive baseline under the old
  mean-pool probe (0.368). Until it runs, the keystone is not airtight. → **TBA.**
```bash
source dev_artiJEPA/scripts/_env.sh
# To finish only what's missing, edit run_baselines_spatial.sh `for m in …` → `vitl dinov2 resnet siglip`:
setsid env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash /scratch1/hongn/artijepa/eval/run_baselines_spatial.sh < /dev/null \
  >> /scratch1/hongn/artijepa/eval/eval_256_fairfight.log 2>&1 &
```
- clip+siglip feature caches (`base_clipsp`,`base_siglipsp`, 17 GB ea) exist →
  cache-hit; siglip just (re)trains its missing attentive seeds. ⚠ Do NOT co-launch
  on a node already running another job — the 16 GB SLURM host-RAM cap will OOM both.
- Then aggregate all `_s{0,1,2}.json` (`eval/aggregate_spatial.py`) → fill the
  3-seed table in `RESULTS.md`.

---

## ▶ NEW DOWNSTREAM TASKS (added 2026-06-30)

These extend eval beyond OOD phoneme decoding to clinical articulation corpora and
to the condition-aware / acoustic directions. Each needs a TextGrid parser + a
manifest builder pairing the annotation with the paired rtMRI video; reuse the
seconds-based alignment + frozen-encoder→per-token-feature path from
`eval_phoneme.py`.

### 8. Stuttering disfluency-type classification — **PIPELINE BUILT, RESULTS TBA**
Segment-level classification of **disfluency type** from frozen (or fine-tuned)
rtMRI features. Infra landed 2026-06-30; the runs themselves are TBA.
- **Data:** `/data1/span_data/stuttering/PWS{3,4,5,6,7,8,10}/textgrid/*.TextGrid`
  (498 files, 7 PWS speakers) + paired `avi/` video (104×104 @ 99 fps, **same
  geometry as usc_lss**). Two label tiers:
  - `disfluency` (primary): ~2100 labeled events. Canonical types
    **block / rep / pro** + rare **osci / revert / filler / abandon**.
  - `disfluency2` (secondary/overlapping): 126 labeled events, ~99% **rep**.
- **Manifest (DONE):** `build_manifest()` → `disfluency_manifest.csv` = **3130
  segments** (rep 795 / block 693 / pro 620 / osci 42 / other 21 / fluent 959;
  62 dropped for dur, 1 uncanon). Per-speaker: PWS3 735, PWS6 729, PWS5 433,
  PWS4 378, PWS8 364, PWS10 296, PWS7 195.
- **Labels (DONE):** `stutter.canonicalize()` — type = substring after first `_`;
  typo repair (`blcok/blok`→block, `repo/red/ep/wordrep`→rep, `fille`→filler),
  strip `?`; bucket5 {block, rep, pro, osci, other}. Compound `a+b` → primary
  component (single-label) + full component set (`multi` column). Tasks: `type5`
  / `type3` (block/rep/pro) / `binary` (fluent-vs-disfluent, needs fluent negs).
- **Probe (DONE):** frames sampled uniformly across each padded interval → frozen
  encoder → **attentive pool over all clip tokens → linear** (`SegmentProbe`;
  mean/mlp variants too). Inverse-freq class weighting for imbalance.
- **Encoders wired:** frozen V-JEPA2 / T-SSL (`--encoder`), image baselines
  (`--model {clip|siglip|dinov2|vitl|resnet}`), and **VideoMAE-L frozen +
  fine-tune** (`--model videomae [--mode finetune]`). **Canonical = attentive @
  256px, LOSO** over the 7 PWS (frozen path only, one shared feature cache).
- **Metric:** severe imbalance → **macro-F1 (primary) + balanced accuracy +
  confusion matrix**, per held-out speaker and pooled over folds.
- **Built:** `artijepa/stutter.py`, `artijepa/eval_disfluency.py`,
  `artijepa/videomae_baseline.py`, `configs/eval_disfluency.yaml`,
  `scripts/15_build_stutter_manifest.sh`, `scripts/16_eval_disfluency.sh`,
  `tests/test_disfluency_smoke.py` (21/21). **Result rows in `RESULTS.md` = TBA.**
```bash
source scripts/_env.sh
bash scripts/15_build_stutter_manifest.sh --fluent-per-file 3      # once
bash scripts/16_eval_disfluency.sh                                  # frozen V-JEPA2 baseline
bash scripts/16_eval_disfluency.sh --encoder $ARTI_OUT/runs/tssl_vitl_256/latest.pt --tag tssl256
bash scripts/16_eval_disfluency.sh --model videomae --tag vmae_frozen
bash scripts/16_eval_disfluency.sh --mode finetune --model videomae --tag vmae_ft
```

### 9. Pre/post-glossectomy phoneme classification (gloss) — **NOT STARTED (TBA)**
Phoneme decoding evaluated **separately for pre- vs post-glossectomy** to quantify
articulatory change and compensation.
- **Data:** `/data1/span_data/gloss/spk{1,2,3}/{pre,post}/textgrids/*.TextGrid`
  (spk2 has `post1`/`post2`). Tiers: `words` + `phones` (ARPABET + stress, MFA
  forced-aligned). Paired video in `gloss/resampled_video`; ROIs in
  `gloss/roi_boxes`,`gloss/roi_time_series`; SAM seg in `gloss/spkN/sam_seg`.
- **Task:** per-token phoneme classification (same head/metrics as Task 2) run on
  pre and on post; report the **pre→post Δ** in κ/PER/accuracy per speaker
  (degradation under altered anatomy).
- **Bonus:** binary **pre-vs-post condition classification** from features — a
  proxy for whether the encoder captures compensatory articulation.
- **Metric:** per-phoneme accuracy, **κ, PER**; pre vs post delta; per-speaker.
- **Need:** `gloss.py` (parser + manifest, video pairing), `configs/eval_gloss.yaml`.

### 10. Articulatory-condition JEPA — planning + eval metric — **PLANNING (TBA)**
A **condition-aware JEPA**: encoder/predictor conditioned on articulatory condition
{healthy-75spk, PWS-fluent, PWS-disfluent, gloss-pre, gloss-post}. Extends the
AUC-JEPA (audio-conditioned) scaffolding from commit `d8f5efb`.
- **Planning tasks:**
  1. Define a condition embedding + injection point (FiLM / token-prepend in the
     predictor; mirror AUC-JEPA's audio-conditioning path).
  2. Build a unified multi-corpus manifest (75-spk + stuttering + gloss) with a
     `condition` column; document in `Master.md`.
  3. Train condition-conditioned predictor; ablate condition-on vs condition-off
     (and shuffled-condition control) — pretraining plan in `TODO_pretraining.md`.
- **Eval metrics (new):**
  - **(i) Conditioning lift:** condition-held-out phoneme κ vs the unconditioned
    encoder (does conditioning help downstream decode?).
  - **(ii) Condition separability:** linear-probe condition accuracy + silhouette
    of pooled features by condition.
  - **(iii) Counterfactual error:** masked-target reconstruction error when the
    condition token is swapped (pre↔post) — should rise if condition is used.
  - **(iv) Few-shot transfer:** pre→post adaptation with k labeled clips.

### 11. Acoustic-to-articulator inversion (AAI) — **NOT STARTED (TBA)**
Predict the **articulatory representation from acoustics** (reverse of the
silent-video phoneme task) — tests whether the JEPA latent lies on a shared
acoustic–articulatory manifold.
- **Input:** acoustic features (wav2vec2/WavLM or mel) from paired audio.
- **Target (pick one):** frozen rtMRI **JEPA tokens**, or `gloss/roi_time_series`
  / tongue contours / SAM landmarks, or usc_lss kinematics.
- **Data:** corpora with paired audio — 75-spk corpus, gloss (`denoise_audio.py`
  pipeline), usc_lss (ships contours/kinematics).
- **Metric:** standard AAI **Pearson correlation** (per-articulator, mean) +
  **RMSE**; for JEPA-token targets also cosine/MSE. Correlation is the headline.
- **Need:** an audio feature extractor (decoupled, transformers-compatible env)
  + a regression head (`aai.py`, `configs/eval_aai.yaml`). **`RESULTS.md` = TBA.**

---

## ▶ MEDIUM PRIORITY

### 4. ≥3 seeds for the winning T-SSL spatial heads — mostly DONE
- `pretrained256` + `tssl256` × {tcn_spatial, attentive} × seeds {0,1,2} = **DONE**
  (in `RESULTS.md`, 3-seed table). `eval_phoneme.py --seed` overrides `meta.seed`
  (probe init/shuffle only); seed in the JSON name (`…_s{seed}.json`) so seeds
  don't clobber. Driver `eval/run_seeds_spatial.sh`.
- Remaining: error bars on the 128px spatial heads if needed; the combined-256
  checkpoint (#1) should also get 3 seeds once it lands.

### 5. Probe ablations (cheap — features cached)
- Already DONE (single seed): `linear`/`mlp`/`tcn`/`lstm`/`transformer` × CE/CTC on
  `pretrained128`+`tssl128` (`scripts/06_probe_sweep.sh`); `tcn_spatial`+`attentive`
  (`scripts/07_probe_spatial.sh`). Numbers in `RESULTS.md`.
- Optional: beam decode for CTC; a **25 fps frame-level head** (×2 upsample) to
  beat the 80 ms token-rate ceiling on fast phonemes.

---

## ▶ LOW PRIORITY / BONUS

### 6. usc_lss dense heads — **NOT STARTED (TBA)**
usc_lss ships tongue contours / SAM segmentation / kinematics → enables
segmentation / landmark probes on the OOD speaker (Plan B.4). Future work.

### 7. Re-run image baselines at additional res / DINOv3
DINOv3 ViT-L was unavailable in this env's timm (only ViT-7B/ConvNeXt) → DINOv2
ViT-L/14 used as substitute. Revisit if a newer timm lands.

---

## Public models evaluated (image baselines)
All via `artijepa/baselines.py` (timm, per-frame → tubelet-pool to V-JEPA's token
grid → same probe+labels). Each at native res, "its best shot" (minmax→[0,1] +
model's own mean/std). For the spatial probe, `baselines.py` emits the per-frame
patch-token grid when `pool_spatial=False` (ViT → `forward_features` minus
CLS/reg; ResNet → conv map; grid side capped at 16 via adaptive-pool so dinov2@518
37×37→16×16 stays tractable). Configs `eval_phoneme_usc_lss_baseline.yaml`;
drivers `scripts/05_eval_baselines.sh`, `eval/run_baselines_spatial.sh`.

| model | res | mean-pool tcn (done) | spatial-probe fair-fight |
|---|---|---|---|
| supervised ViT-L/16 | 224 | ✅ κ 0.368 (best baseline) | ❌ **NOT STARTED — key competitor** |
| siglip-L/16 | 256 | ✅ κ 0.313 | ⚠ tcn_spatial 3/3; attentive 1/3 |
| clip-L/14 | 224 | ✅ κ 0.293 | ✅ 3/3 (tcn_spatial 0.345) |
| dinov2-L/14 (DINOv3 unavailable) | 518 | ✅ κ 0.291 | ❌ REMOVE, NOT RUN |
| resnet-50 | 224 | ✅ κ 0.304 | ❌ REMOVE, NOT RUN |

## Video model baseline — VideoMAE (added 2026-06-30)
`artijepa/videomae_baseline.py` (transformers `VideoMAEModel`, **video** 3-D tubelet
encoder, not per-frame). The generic video-SSL competitor to Arti-JEPA's video-JEPA.
Default `MCG-NJU/videomae-large` (ViT-L, D=1024, 16f, tubelet 2, patch 16 →
8×14×14 tokens). Same input contract as the image baselines (minmax→[0,1] +
ImageNet mean/std, resize 224). Native geometry (224px/16f) overrides the 256px
config — that's the baseline's best shot. Two modes on the disfluency task:

| model | mode | phoneme (usc_lss) | disfluency (Task 8) |
|---|---|---|---|
| VideoMAE-L/16 | **frozen** (attentive probe) | TBA (optional) | ❌ TBA |
| VideoMAE-L/16 | **fine-tune** (encoder + head) | — | ❌ TBA |

Frozen exposes `.backbone(clip)` (drop-in for the extractor); fine-tune trains the
encoder end-to-end (`--mode finetune`, `encoder_lr` ≪ head `lr`). Run via
`scripts/16_eval_disfluency.sh --model videomae [--mode finetune]`.

---

## How to run
```bash
source dev_artiJEPA/scripts/_env.sh
# Task 2 (gold OOD) phoneme eval, pretrained baseline (128px):
bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml --tag pretrained128
# with-T-SSL phoneme eval:
bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml \
     --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt --tag tssl128
# 256px (needs eval_phoneme_usc_lss_256.yaml; --probe {tcn_spatial,attentive} auto-selects the spatial cache)
```
