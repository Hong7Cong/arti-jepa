# Fine-tuning V-JEPA 2 on rtMRI for Speech-Production Analysis
### Preprocessing & Fine-tuning Experiment Plan

---

## 0. Scope and guiding principle

**Goal.** Adapt the pretrained V-JEPA 2 video encoder (action-free, self-supervised) to real-time MRI (rtMRI) midsagittal vocal-tract video, and evaluate its learned representations on speech-production downstream tasks.

**Domain constraints (given).**
- Native frame rate: **83.28 fps** → target **50 fps**.
- Native spatial resolution: **84 × 84**, grayscale, single midsagittal slice — far from the 256–384 px RGB natural video V-JEPA 2 saw in pretraining.

**Strategy: cheap first, escalate only if needed.** The encoder is the asset and rtMRI labels are scarce, so we move up a ladder of cost/risk:

1. Frozen encoder + attentive probe (V-JEPA 2's own eval protocol) — establishes out-of-the-box transfer.
2. (Optional) Continued self-supervised pretraining on *unlabeled* rtMRI (domain-adaptive) — closes the modality gap in the spirit of JEPA.
3. Partial fine-tuning (unfreeze top blocks / LoRA).
4. Full fine-tuning — highest capacity, highest overfitting/forgetting risk.

Do not jump to (4) before (1) tells you what the frozen features already deliver.

**Relevant V-JEPA 2 facts this plan relies on (from the paper):**
- Encoder is a ViT (ViT-L ≈ 300M, dim 1024; ViT-g ≈ 1B, dim 1408). Predictor ≈ ViT-small.
- Input is patchified into **tubelets of 2 × 16 × 16 (T × H × W)** → temporal patch size 2, spatial patch size 16.
- Position info via **3D-RoPE** (relative), *not* absolute embeddings — this is important: it generalizes across grid sizes, so we can fine-tune at a different resolution / clip length than pretraining without re-learning absolute position embeddings.
- Pretraining resolution 256 → cooldown 384; clip length 16 → 64 frames.
- Pretraining objective = mask-denoising in representation space, L1 feature loss, EMA target encoder + stop-gradient (anti-collapse).
- Official code: `github.com/facebookresearch/vjepa2`.

---

## Part A — Data Preprocessing Pipeline

The big three adaptations are **temporal resampling (83.28→50 fps)**, **spatial resolution (84×84 must become a multiple of 16)**, and **grayscale→3-channel**. Process in this order:

### A.1 Intensity normalization (do this before resizing)
rtMRI has scanner/subject-dependent intensity inhomogeneity. Per-video:
1. Percentile clip to suppress outliers (e.g., clip to [1st, 99th] percentile).
2. Z-score or min–max to a common range.
3. Compute a **single global grayscale mean/std over the training split** and store it; apply at load time. (Do *not* reuse ImageNet/natural-video stats — the intensity distribution is different.)

Optionally evaluate per-frame vs per-video normalization as an ablation (per-video preserves relative brightness dynamics; per-frame removes slow drift but can flatten real intensity cues).

### A.2 Temporal resampling 83.28 → 50 fps
The ratio is **83.28 / 50 = 1.6656** — *non-integer*, so simple frame-dropping (e.g., every other frame → 41.6 fps) is wrong.

- **Recommended:** resample onto a uniform 50 Hz time grid by **linear temporal interpolation** of pixel values (equivalently, realize it in the dataloader by index-based interpolation rather than re-encoding the video — avoids codec artifacts and keeps the raw frames available).
- **Ablation:** video frame interpolation (flow-based, e.g., RIFE) for cleaner fast-motion handling, vs nearest-frame snapping (cheapest).
- **Caveat to test:** speech transients (stops, taps, flaps) can be ~20–40 ms; at 50 fps (20 ms/frame) the fastest events sit near the sampling limit. Keep an **83 fps variant** and a **25 fps variant** as temporal-rate ablations to confirm 50 fps doesn't alias fast articulation.

With temporal tubelet T = 2, each token-pair spans 2 frames = **40 ms** at 50 fps.

### A.3 Spatial resolution (84×84 → multiple of 16)
84 is **not divisible by 16**, so it cannot be patchified directly. Three principled options — note that 84×84 carries little genuine high-frequency content, so upsampling adds compatibility, not information:

| Variant | Transform | Patch grid | Tokens/frame-pair | Use as |
|---|---|---|---|---|
| **Full-transfer (primary)** | bicubic resize 84→**256** | 16×16 | 256 | Default baseline — matches pretraining scale, max reuse of pretrained spatial structure |
| Middle | bicubic resize 84→**128** (or 224→14×14) | 8×8 | 64 | Efficiency ablation |
| Native-pixel | reflect-pad 84→**96** (+6 px/side) | 6×6 | 36 | Cheapest; preserves raw pixels, no interpolation |

**Experimental logic:** establish the strongest transfer baseline at 256 first (isolating the resolution question), then ablate *downward* (128, 96) to find the compute sweet spot. Because 3D-RoPE is relative, these grid changes are well-supported.

### A.4 Grayscale → 3 channels
The pretrained patch-embedding conv expects 3 (RGB) channels. **Replicate the single grayscale channel ×3** so the pretrained patch-embed weights are reused directly. (Alternative ablation: average the RGB patch-embed kernels into a 1-channel conv. Channel replication is the safe default.)

### A.5 ROI / cropping
84×84 is already small/cropped, so keep the full frame by default. Optional ablation: tight vocal-tract ROI crop (then resize) to concentrate capacity on articulators.

### A.6 Clip sampling & length
- Temporal patch T=2 means clips should be **even-length**.
- Recommended clip length: **32 frames (640 ms) default**, with **64 frames (1280 ms)** as a longer-context variant for coarticulation. (16 frames = 320 ms is likely too short for syllable-scale dynamics.)
- Token budget sanity check (256-res): 32 frames → 16 temporal × 256 spatial = **4,096 tokens**; 64 frames → **8,192 tokens** (ViT-g handled 64-frame/384 in pretraining, so feasible). At 128-res these drop to 1,024 / 2,048 — the efficiency argument for lower resolution.

### A.7 Augmentation (be anatomically careful)
**Allowed:** small spatial translation/jitter, mild intensity jitter, light additive Gaussian noise (Rician-noise proxy), random *temporal* crop, mild speed perturbation (treat as articulation-rate augmentation — log it).
**Forbidden / risky:** **horizontal flip** (mirrors anterior↔posterior, destroys phonetic meaning), large rotations, color jitter (grayscale). Aggressive RandomResizedCrop can crop out articulators — use a conservative scale range if used at all.

### A.8 Splits
**Split by speaker (subject-disjoint)** to prevent anatomy leakage; also ensure no utterance overlap across splits. Report subject-independent results as the headline; subject-dependent only as an upper-bound reference. (Candidate corpora to confirm against your data: USC-TIMIT, USC 75-Speaker Speech MRI DB, and similar — verify exact specs for your dataset.)

### A.9 Preprocessing config (template)
```yaml
preprocess:
  intensity: {clip_percentiles: [1, 99], norm: zscore, scope: per_video}
  grayscale_norm: {mean: <compute_from_train>, std: <compute_from_train>}  # applied to 3 replicated channels
  temporal: {target_fps: 50, method: linear_interp, realize_in: dataloader}
  spatial: {primary: {resize: 256, interp: bicubic},
            ablations: [{resize: 128}, {pad_to: 96, mode: reflect}]}
  channels: replicate_to_3
  clip: {length_frames: 32, even_only: true, stride: random_temporal_crop}
  augment:
    enabled: [translate_small, intensity_jitter, gaussian_noise, speed_perturb]
    disabled: [hflip, large_rotation, color_jitter]
  split: {by: speaker, disjoint: true}
```

---

## Part B — Fine-tuning Experiment Plan

### B.1 Backbone choice
Start with **ViT-L (300M)** for fast iteration and lower overfitting risk on small rtMRI data. Escalate to **ViT-g (1B)** only if ViT-L shows headroom and compute allows. Initialize from the released V-JEPA 2 checkpoints.

### B.2 The fine-tuning ladder

| Tier | What's trainable | Cost | When to use |
|---|---|---|---|
| **T0 — Frozen + attentive probe** | 4-layer attentive probe only (the paper's protocol) | Lowest | Always first. Pure test of representation transfer. |
| **T1 — Frozen + task decoder** | Task-specific head (e.g., segmentation/landmark/inversion decoder) | Low | Establish best frozen-backbone task performance. |
| **T2 — Partial fine-tune** | Top *N* blocks (or LoRA on attention) + head; backbone otherwise frozen | Medium | If T1 plateaus below target. |
| **T3 — Full fine-tune** | Entire encoder + head | High | Last resort; strong regularization required. |
| **T-SSL — Domain-adaptive pretraining** | Re-run V-JEPA objective on unlabeled rtMRI, *then* T0–T2 | High | Likely the biggest single win given the modality gap (see B.3). |

### B.3 Optional but recommended: domain-adaptive self-supervised pretraining (T-SSL)
Before any labeled fine-tuning, continue the **V-JEPA mask-denoising objective** on the (plentiful) *unlabeled* rtMRI, initialized from V-JEPA 2 weights:
- Keep the core recipe: EMA target encoder + stop-gradient + **L1 feature loss**, multiblock masking.
- **Adjust masking for the smaller token grid** — at 128/96-res there are far fewer tokens, so block sizes and mask ratio must be re-tuned (a 256-res mask ratio applied to a 6×6 grid is degenerate).
- Watch for **representation collapse** (the JEPA failure mode): monitor **label-free** feature variance / effective-rank / mean-abs-cosine and EMA momentum during SSL (`collapse.py`). (The old weak stimulus-group probe was dropped as not meaningful; downstream usefulness is measured separately by the phoneme eval in B.4 / Part C.)
- This adapts the encoder to grayscale, low-res, articulatory motion statistics while preserving the pretrained prior. Treat "with vs without T-SSL" as a headline comparison.

### B.4 Downstream task = PHONEME PREDICTION (pinned)

The primary downstream task is **per-frame/per-token phoneme prediction** from the
silent rtMRI video — the direct articulatory test of whether the JEPA
representation encodes *speech content*, not just appearance. (The earlier weak
stimulus-group classification was dropped as not meaningful — knowing a clip is
"a picture-description vs a passage" says nothing about articulation.)

A small probe sits on the **frozen** encoder's per-temporal-token features
(`[B, T', D]`, obtained by mean-pooling the spatial tokens at each time step,
since ViT tokens are temporal-major).

**Probe-head ablation** (`eval_phoneme.py`), ordered by temporal-context
capacity: `linear` (per-token, no context) → `mlp` (per-token nonlinear) →
`tcn` (local 1-D conv, ±2 tokens) → `lstm` (bi-LSTM, full-sequence recurrence)
→ `transformer` (self-attention encoder, global context). The recurrent/attention
heads test whether *coarticulation*-scale temporal dependency the conv head misses
is recoverable from the frozen features.

**Loss ablation — per-token CE vs CTC.** Default is per-token cross-entropy on the
seconds-aligned labels (ignore padded tokens; κ-native — every 80 ms token has a
gold phoneme). Alternatively train with **CTC** (`nn.CTCLoss`, blank-augmented
vocab): drop the forced alignment, feed **whole utterances** (variable length,
padded + `input_lengths`) with the *collapsed* phoneme sequence as target
(`target_lengths`), marginalizing over alignments. CTC is the alignment-free
option — it (i) is the natural fit for Task-1 *pseudo* labels where exact alignment
is untrusted, and (ii) optimizes the sequence objective directly, so it can cut
PER insertion/deletion errors. **Caveats:** CTC emits ≤1 label per token, so it does
**not** beat the 80 ms token-rate ceiling on sub-80 ms phonemes (same as CE); and
κ is undefined for CTC outputs without a forced-alignment pass (report PER as the
primary CTC metric, κ via CTC-forced-alignment optionally). CTC pairs most
naturally with the lstm/transformer heads (they model the whole sequence).

Read off frame-level **Cohen's κ** and sequence **PER**. Two label sources:

- **Task 1 — pseudo phonemes (75-speaker corpus).** The corpus ships no phoneme
  labels but every clip has paired audio (aac @ 22.05 kHz). Run an audio
  phoneme-recognition model (wav2vec2 / WavLM CTC, e.g.
  `facebook/wav2vec2-lv-60-espeak-cv-ft`) on the audio to get a ~50 Hz phoneme
  stream; ask whether the *video* features can predict it. Self-consistent
  transfer probe in the audio model's phone vocabulary. (`audio_phoneme.py`.)
- **Task 2 — gold phonemes (OOD speaker `usc_lss`).** Hand-annotated ARPABET
  phonemes with timestamps for one held-out speaker (`/scratch1/hongn/usc_lss`,
  684 utterances, **104×104 video @ 99 fps**, 16 kHz audio, 41 phones incl.
  `sil`). OOD on three axes — speaker, resolution (vs 84×84), and frame rate (vs
  83.28) — so it is the clean generalization test. (`usc_lss.py`.)

**Temporal alignment (audio/labels ↔ video tokens) — done in SECONDS, hence
frame-rate agnostic.** Clips are resampled onto the `target_fps` grid (25 fps)
that feeds the encoder, so output-frame *f* sits at time `f / target_fps`. Audio
phoneme models emit ~50 Hz (20 ms) frames, so **2 audio units ≈ 1 video frame**
(40 ms) and **4 ≈ 1 JEPA token** (tubelet 2 → 80 ms). We label each temporal
token by the phoneme covering its time-centre (`phonemes.py`). Because labels
live in seconds, the 99-fps OOD speaker needs no special-casing — only the decord
resample sees the native fps. *Caveat:* the 80 ms token rate (12.5 Hz) cannot
resolve sub-80 ms phonemes — a real ceiling on κ/PER; a 25 fps frame-level head
(×2 feature upsample) or smaller tubelet is the refinement if needed.

> Deferred heads (need dense labels we don't have for the 75-speaker corpus):
> air–tissue/vocal-tract **segmentation** (Dice/IoU), **landmark** regression
> (RMSE/PCK), **acoustic-to-articulatory inversion** (per-articulator RMSE/ρ).
> `usc_lss` *does* ship tongue contours / SAM segmentations / kinematics, so
> those become possible there later.

### B.5 Hyperparameters (starting points)

| Setting | T0/T1 (frozen) | T2 (partial) | T3 (full) | T-SSL |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Backbone LR | — (frozen) | 5e-5 | 2e-5–5e-5 | 1e-4 (cosine) |
| Head/probe LR | 5e-4–1e-3 | 5e-4 | 5e-4 | n/a |
| LR schedule | cosine, 5–10% warmup | cosine | cosine | warmup-constant-decay |
| Layer-wise LR decay | n/a | 0.75 | 0.75 | n/a |
| Weight decay | 0.01–0.05 | 0.05 | 0.05 | 0.04 |
| Stochastic depth | 0 | 0.1 | 0.1–0.2 | per recipe |
| Epochs | 20–50 (early stop) | 20–50 | 20–50 | data-bounded |
| Precision | bf16 | bf16 | bf16 | bf16 |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 |
| Weight EMA for eval | — | yes | yes | (teacher EMA) |
| Batch (per-GPU) | larger | small + grad-accum | small + grad-accum | small + grad-accum |

Use **activation checkpointing** and **gradient accumulation** (effective batch ~64–256) at 256-res; much larger batches are feasible at 128/96-res.

---

## Part C — Evaluation Protocol

**Task = phoneme prediction (B.4).** Metrics:
- **Cohen's κ** — chance-corrected per-token agreement (frame-level), the headline
  number (robust to the heavy phoneme class imbalance; raw frame-accuracy reported
  alongside).
- **PER** — Phoneme Error Rate = edit-distance / reference-length, after
  CTC-collapsing the per-token predictions (merge runs, drop silence/blank).
  Report micro (corpus-level) and macro (per-utterance mean).

- **Headline comparison:** **with vs without T-SSL** — run `eval_phoneme.py` on the
  *pretrained* encoder vs a `tssl_*` adapted checkpoint, same probe/data; report
  Δκ / ΔPER. Run on both Task 1 (pseudo, in-domain speakers) and Task 2 (gold, OOD
  speaker); the OOD lift is the strongest claim.
- **Other baselines:** from-scratch (no pretrained init) quantifies transfer value;
  a natural-image/video ViT of similar size under the same probe quantifies the
  JEPA-specific benefit; chance/majority for context.
- **Statistics:** ≥3 seeds for the probe (cheap once features are cached), report
  mean ± std; significance test vs the strongest baseline. Subject-independent is
  the headline (Task 2 is a *single* OOD speaker — report it as generalization,
  Task 1 as the multi-speaker number).
- **Diagnostics beyond the metric:** confusion by phoneme class (which articulatory
  contrasts survive), κ/PER vs speech-rate, error on fast transients (stops/taps —
  where the 80 ms token rate bites), per-utterance failure cases.

*Measured (2026-06-07, Task 2 gold OOD speaker, 128 px, `tcn` probe):*

| encoder | test κ | test PERµ | val κ | frame-acc |
|---|---|---|---|---|
| pretrained (no T-SSL) | 0.222 | 0.760 | 0.240 | 0.259 |
| **+ T-SSL (128px, 50 ep)** | **0.247** | 0.755 | 0.277 | 0.280 |
| Δ | **+0.025 (+11%)** | −0.005 | +0.037 | +0.021 |

So **domain-adaptive T-SSL measurably improves the representation's phonetic
content and the gain transfers to an unseen speaker.** The T-SSL run itself was
collapse-free (effective_rank 79→102/1024, mean_abs_cosine 0.957→0.560 over
training). κ (chance-corrected) shows the lift cleanly; PER barely moves —
phoneme-sequence recovery is capped by the 80 ms token rate, not the encoder.
256px (resolution ablation) pending a fresh allocation (the 16 GB host-RAM cap
OOM'd the first 256px attempt; config fixed to num_workers 2 / pin_mem off).

---

## Part D — Ablation Matrix

| Axis | Conditions | Question answered |
|---|---|---|
| Spatial resolution | 256 / 128 / 96 (pad) | Does upsampling buy accuracy, or is low-res enough? (compute sweet spot) |
| Frame rate | 50 / 83 / 25 fps | Does 50 fps alias fast articulation? |
| Clip length | 16 / 32 / 64 frames | Context needed for coarticulation |
| Fine-tune depth | T0 / T1 / T2 / T3 | Capacity vs overfitting trade-off |
| Domain-adaptive SSL | with / without T-SSL | Value of closing the modality gap |
| Channel handling | replicate×3 / 1-ch conv | Patch-embed adaptation |
| Normalization | per-video / per-frame | Intensity-handling choice |
| Backbone size | ViT-L / ViT-g | Scaling on small data |
| **Probe head** | linear / mlp / tcn / **lstm** / **transformer** | Temporal-modeling capacity the read-out needs to surface phonetic content |
| **Probe loss** | per-token CE / **CTC** | Aligned per-token (κ-native) vs alignment-free sequence (PER-native) training |
| Encoder family (baseline) | V-JEPA / CLIP / SigLIP / DINOv2 / ViT-L / ResNet | Is the transfer JEPA-specific or generic to any pretrained vision encoder? |

Run ablations on the cheapest viable config (e.g., 128-res, frozen probe) before committing compute to the full grid.

---

## Part E — Phasing & Milestones

1. **Phase 0 — Data engineering.** Build the A.1–A.9 pipeline; verify clip tensors, fps grid, normalization stats; sanity-check a few clips visually; finalize subject-disjoint splits.
2. **Phase 1 — Transfer baseline (T0).** Frozen ViT-L + attentive probe at 256-res. This is the go/no-go signal.
3. **Phase 2 — Task head (T1) + key ablations.** Add the real task decoder; run resolution and fps ablations.
4. **Phase 3 — Escalate (T2, then T-SSL).** Partial fine-tune; then domain-adaptive SSL → re-run T0–T2 to measure its lift.
5. **Phase 4 — Full fine-tune (T3) + scale to ViT-g** only if justified by Phases 2–3.
6. **Phase 5 — Final eval, seeds, baselines, write-up.**

---

## Part F — Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Small rtMRI data → overfitting in T3 | Prefer frozen/partial; strong WD, stochastic depth, weight EMA, early stopping |
| Modality gap (grayscale, low-res, MRI texture) | Domain-adaptive SSL (T-SSL); rtMRI-specific normalization |
| **Representation collapse** during T-SSL | Monitor feature variance/rank + held-out probe; tune EMA momentum & mask ratio for the small grid |
| Temporal aliasing of fast events at 50 fps | 83/25 fps ablations; flow-based interpolation; per-articulator error analysis |
| Resolution info ceiling (84×84 has little detail) | Don't assume 256 > 128; let ablations decide; consider native-pixel pad-to-96 |
| Speaker leakage inflating scores | Strict subject-disjoint splits; report subject-independent as headline |
| Catastrophic forgetting of useful prior | Layer-wise LR decay; freeze lower blocks; short schedules |
| Latent features ≠ interpretable articulatory variables | Decoders/probes to recover constriction/landmark measures; validate the JEPA "ignore unpredictable detail" bias isn't discarding fine constriction-degree cues |

---

## Part G — Resources
- V-JEPA 2 paper: *Self-Supervised Video Models Enable Understanding, Prediction and Planning* (arXiv:2506.09985).
- Code & checkpoints: `github.com/facebookresearch/vjepa2`.
- Confirm input/patch/RoPE conventions against the repo's data loaders before locking the pipeline.

> **Primary downstream task is pinned: phoneme prediction** (B.4 / Part C),
> evaluated by frozen-feature probe with κ + PER on pseudo (75-speaker) and gold
> (OOD `usc_lss`) labels. Heads/metrics/configs for that are implemented
> (`eval_phoneme.py`, `phonemes.py`, `usc_lss.py`, `audio_phoneme.py`). Other
> heads (segmentation/landmarks/inversion) remain deferred until labels exist.