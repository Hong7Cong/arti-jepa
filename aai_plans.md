# AAI-JEPA — Acoustic → rolled-out rtMRI video embeddings (WavLM-conditioned, self-supervised)

**Goal.** Train an **acoustic-conditioned predictor** that, given `ctx=2` seed frames,
**rolls out the remaining 14** of `T'=16` temporal tokens in a frozen ViT-L rtMRI feature
space, conditioned **only on WavLM audio embeddings**. Over that ~0.56 s audio-only horizon
the only signal that can drive the evolving vocal-tract representation is the audio, so the
predictor learns **acoustics → future MRI video embeddings**. The rtMRI latent *is* the
articulatory state (it depicts the vocal tract), so this is acoustic-to-articulator
inversion **in the encoder's video-embedding space**.

**Fully self-supervised — no extra labels.** Data = the rtMRI-75 + longitudinal video
corpus (the same pool T-SSL pretrains on). Audio is the *only* extra input, and it comes
embedded in every video. The target is the frozen encoder's **own future token embeddings**.

> **This is a revival of the (trashed) acoustic AUC-JEPA** (audio→video-latent), now with:
> (1) the `ctx_frames:2` hard-conditioning fix learned from the arti-6 line, and
> (2) run purely self-supervised on the **combined** corpus. **arti-6 is not used anywhere**
> — no arti conditioning, no arti target, no arti readout, no phoneme labels.

---

## 0. TL;DR / status

| | |
|---|---|
| **Corpus** | `manifest_combined.csv` = rtMRI-75 (`speaker75`, 2,371 vids, aac@22.05 kHz) + longitudinal (7,110 `.avi`, mp3@16 kHz). **9,353 train / 128 val, 96 subjects, multi-speaker.** Audio embedded in every video. |
| **Reused ~verbatim (from `trash/`)** | `build_audio_features.py` (WavLM from the video `path`), `audio_cond.py` (pooling + `AudioConditionedPredictor`), `aucjepa_dataset.RTMRIAudioDataset` (`{clip,audio,valid}`), `aucjepa_train.py` (train loop) |
| **New / edits** | add `ctx_frames:2` wiring to the trainer (port from `acjepa_train.py`), `audio_gap` diag, `aai_eval.py` (latent metrics), config `aai_wavlm_256_combined_ctx2.yaml`, revive scripts `08/09` |
| **Target** | the frozen encoder's own future token embeddings `h` (self-supervised JEPA) |
| **Encoder** | frozen domain-adapted T-SSL ViT-L @256px/50fps — `tssl_vitl_256_combined` (matches this pool) or `tssl_vitl_256` (75-only) if combined isn't converged |
| **Needs NO** | arti-6, phoneme labels, tract variables, any manual annotation |
| **Compute** | offline: WavLM over ~9.5k videos (`his-extract` env, GPU). train: V100-32GB, encoder-forward-bound, same envelope as the acjepa ctx=2 run |
| **Status** | PLAN only — nothing built yet |

---

## 1. Why this works

The acjepa post-mortem ([[acjepa-v100-256-run]]) found arti-6 conditioning was weak because
arti-6 is a geometric *readout of the MRI the encoder already sees*. Audio is different:

* **Audio is not in the visual frame.** It cannot be copied from the seed frames, so it
  carries complementary, causally-necessary information about *how the tract will move next*.
* **The MRI latent encodes the articulators.** T-SSL ViT-L features of an rtMRI frame
  describe the vocal-tract configuration; predicting the *future* MRI latent from audio is
  inversion directly in video-embedding space — no scalar readout needed.
* **`ctx=2` makes it an inversion, not an autocomplete.** With only 2 seed frames, frames
  2…15 (~0.56 s) have no visual momentum to coast on — the predictor must decode the
  acoustics to place the future tract representation.
* **Multi-speaker, self-supervised, ~9.5k videos** → far broader than the single-speaker
  usc_lss `.mat`; the audio→video map can actually generalize across speakers.

Primary training signal: `audio_gap = L1(shuffled-audio AR) − L1(real-audio AR)` on the
autoregressive branch. Clearly positive ⇒ the rollout uses the audio.

---

## 2. Data reality (verified)

* **Every video carries an audio stream**, confirmed with `ffprobe`:
  * speaker75 `*.mp4`: **aac @ 22.05 kHz**.
  * longitudinal `*.avi`: **mp3 @ 16 kHz**.
  `audio_phoneme.extract_audio(path)` already decodes either to **16 kHz mono float32** via
  ffmpeg (raw `f32le` pipe) — the exact input WavLM wants. No `.mat`, no sibling wav files.
* **Video is decoded on the fly** by `RTMRIVideoDataset` (decord, tile mode) — no image cache
  needed. Chunk `c` of a video → token windows via `clip_start_frame = c * frames_per_clip`.
* **Manifest** `manifest_combined.csv` columns: `path, subject, …, n_frames, fps, duration_s,
  split`. src fps ≈ 83.28 (speaker75) / per-row (longitudinal); resampled to `target_fps=50`.
* **Alignment is by seconds.** WavLM (~49.95 Hz) pools onto the `T'=16` tokens (25 Hz at
  50 fps / tubelet 2) with the same window math as the phoneme eval — any fps works.
* **Splits:** the manifest is 9,353 train / 128 val (mostly train). For an honest rollout
  eval, carve a **speaker-disjoint** val/test (see §5.0) — cheap, no labels needed.

WavLM: `microsoft/wavlm-base-plus`, 768-D, ~49.95 Hz. Middle layers carry the most
phonetic/articulatory content — default **layer 9**, sweep {6, 9, 12} in M4. Learnable
weighted layer-sum (SUPERB-standard) is the v2 upgrade.

---

## 3. Architecture

```
   video (mp4/avi) ──ffmpeg 16k mono──> wav ──WavLM(frozen,layer L)──> a[T_audio,768]  (offline cache)
                                                                          │ pool→tokens, z-score
                                                                          ▼
   clip[3,32,H,W] ──ViT-L(FROZEN, T-SSL)──> h[B,16·HW,1024]   e[B,16,768]
        (seed=2 real frames)                    │              │ state=e[t], action=e[t]-e[t-1]
                                                ▼              ▼
                          AudioConditionedPredictor(action_embed_dim=768)   ← REUSED verbatim
                                                │  ctx_frames=2: seed 2 real latent frames,
                                                ▼  autoregress frames 2..15 from AUDIO only
                                    ẑ[B,14·HW,1024] ──L1(layer-norm)──> h[frames 2..15]
                                        (predicted future MRI video embeddings = the deliverable)
```

* **Frozen encoder:** domain-adapted T-SSL ViT-L; supplies BOTH the seed latent and the
  **self-supervised target** `h` (its own future tokens). @256px/50fps to match its pretraining grid.
* **Predictor:** `audio_cond.AudioConditionedPredictor(action_embed_dim=768, …)` (identical to
  the dimension-generic `arti_cond` predictor). stock `state_encoder`/`action_encoder` become
  `Linear(768,384)` — no predictor edit. `state=e[t]`, `action=e[t+1]−e[t]`. frame-causal + RoPE free.
* **Rollout & loss:** `forward_predictions(h, state, action, ctx_frames=2)` → teacher-forced
  (frames 1..15) + AR (frames 2..15 = the **14** audio-only frames). Loss = `rollout_l1(z_tf) +
  rollout_l1(z_ar)` in layer-normed feature space, masked by `valid`.
* **Trainable:** ~22 M predictor + `Linear(768,384)` heads (~0.6 M). Encoder + WavLM frozen;
  WavLM never runs at train time (features cached).

---

## 4. Reuse map — mostly revival from `trash/`

| Component | Action | Source |
|---|---|---|
| WavLM feature cache (audio from video `path`) | **REVIVE ~verbatim** → `artijepa/build_audio_features.py` | `trash/artijepa/build_audio_features.py` (already reads `extract_audio(row["path"])`) |
| `pool_audio_to_tokens`, `normalize_audio`, `to_state_action`, `AudioConditionedPredictor` | **REVIVE verbatim** → `artijepa/audio_cond.py` | `trash/artijepa/audio_cond.py` |
| Dataset `{clip, audio[T',768], valid}`, tile | **REVIVE verbatim** → `artijepa/aucjepa_dataset.py` | `trash/artijepa/aucjepa_dataset.py` (`RTMRIAudioDataset`) |
| Train loop (frozen enc → cond rollout → L1) | **REVIVE + EDIT** → `artijepa/aai_train.py` | `trash/artijepa/aucjepa_train.py`; **add `ctx_frames:2`** (port the 3-line path from `acjepa_train.py`) + `audio_gap` diag |
| `extract_audio` | **IMPORT AS-IS** | `artijepa/audio_phoneme.py` (live) |
| Latent-space eval (metrics + controls) | **WRITE** `artijepa/aai_eval.py` | new (§5) |
| Config | **WRITE** `configs/aai_wavlm_256_combined_ctx2.yaml` | fork `trash/configs/aucjepa_vitl_256.yaml` + `acjepa_arti6_256_v100_ctx2.yaml` |
| Scripts | **REVIVE** `scripts/08_build_audio_feats.sh`, `09_train_aucjepa.sh`→`16_train_aai.sh`, `17_aai_eval.sh` | `trash/scripts/08,09` |
| Smoke test | **REVIVE + EDIT** `tests/test_aai_smoke.py` | `trash/tests/test_aucjepa_smoke.py` (add ctx=2 assertions) |

The only genuinely *new* logic is the `ctx_frames:2` addition to the trainer (already proven
in `acjepa_train.py`) and `aai_eval.py`. Everything else is a `git mv trash/… →` revival.

---

## 5. Evaluation — all in the video-embedding space (`aai_eval.py`)

No arti-6, no pixel labels — measured on rolled-out MRI embeddings vs the encoder's true future.

### 5.0 Held-out eval set (do first)
Carve a **speaker-disjoint** eval split from the manifest (e.g. hold out K subjects entirely,
+ use the existing 128 val). Self-supervised ⇒ no labels; just reassign `split`. This gives an
honest audio→video rollout test on unseen speakers.

### 5.1 Core metrics (frames 2..15, the audio-only horizon)
* **AR L1** in layer-normed feature space (the objective, on the held-out set).
* **Per-frame cosine** `cos(ẑ[t], h[t])` — scale-free fidelity vs rollout depth (report the curve).
* **Nearest-neighbour frame-retrieval accuracy**: for each `ẑ[t]`, retrieve the nearest true
  frame embedding (within the video / within the eval pool); report top-1/top-5 hit-rate +
  mean frame-offset. "Did the audio place the tract at the right moment?" — no scalar target.

### 5.2 Controls & baselines
1. **Shuffled-audio** (permute audio across batch): AR L1 up / cosine + retrieval collapse →
   proves the audio, not a per-video prior, drives the rollout (= `audio_gap` on the eval set).
2. **No-audio (unconditional)**: same predictor with audio state/action zeroed → how much does
   audio add over pure visual extrapolation from the 2 seed frames? That gap is the acoustics' value.
3. **Teacher-forced upper bound**: TF-branch metrics (past still visible) upper-bound the AR numbers.
4. **Layer sweep**: rerun on caches from WavLM {6, 9, 12}; pick best by AR L1 / cosine.
5. **Speaker split**: report seen-speaker vs held-out-speaker rollout to show generalization.

### 5.3 Optional qualitative decode (nice-to-have)
Map each predicted latent to its nearest-neighbour **true MRI frame** and render the retrieved
sequence — a decoder-free way to *see* the acoustic→articulator rollout. (A learned latent→pixel
decoder is a later option for crisper reconstructions.)

### 5.4 Success criteria (first-pass)
* `audio_gap` clearly positive on the held-out set and above the shuffled control.
* Real-audio AR L1 / cosine beat the no-audio baseline by a clear margin.
* Retrieval top-5 well above chance; held-out-speaker rollout degrades gracefully vs seen.

---

## 6. Config & compute

Fork to `configs/aai_wavlm_256_combined_ctx2.yaml`:

```yaml
meta:  { folder: /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_ctx2, seed: 0,
         dtype: float16, eval_freq: 1, save_freq: 1, probe_max_batches: 8 }
data:
  manifest: /scratch1/hongn/artijepa/manifest_combined.csv   # + a speaker-disjoint eval split (§5.0)
  spatial_size: 256
  spatial_mode: resize
  frames_per_clip: 32
  target_fps: 50.0            # matches the T-SSL encoder's grid; 0.64 s / 32-frame chunk
  sampling: tile
  tubelet_size: 2             # -> T'=16 tokens
  patch_size: 16
  intensity_norm: zscore
  grayscale_stats: /scratch1/hongn/artijepa/grayscale_stats_combined.json
  augment: true
  batch_size: 3               # ctx=2 ckpt-off ~22.7 GB on V100-32GB (bench from acjepa)
  num_workers: 4              # --mem=64G cgroup OOM fix; video decode is the loader cost now
  pin_mem: false
  persistent_workers: false
audio:
  dim: 768
  cache_dir: /scratch1/hongn/artijepa/audio_feats/wavlm_base_plus_L9
  layer: 9
  normalize: zscore
predictor: { kind: ac_audio, pred_depth: 12, pred_embed_dim: 384, pred_num_heads: 12,
             use_rope: true, frame_causal: true }
objective: temporal
ctx_frames: 2                 # seed 2, predict 14  <-- the requested rollout
model:
  model_name: vit_large
  checkpoint: /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/latest.pt   # or tssl_vitl_256 if not converged
  checkpoint_key: target_encoder
  use_activation_checkpointing: false
optimization: { epochs: 20, ipe: 1000, effective_batch: 128, lr: 5.0e-4, start_lr: 1.0e-4,
                final_lr: 1.0e-5, warmup: 2, weight_decay: 0.04, final_weight_decay: 0.04 }
```

* **Encoder choice.** Prefer `tssl_vitl_256_combined` (trained on this exact pool). Per
  [[combined-tssl-256-run]] that run is long (215 ep, was ~ep50 on 2026-06-29) — use its
  furthest-along `latest.pt`; if under-converged, fall back to the finished 75-only
  `tssl_vitl_256`. Pass the **absolute** checkpoint path (the launcher `cd`s away).
* **Train compute:** encoder-forward-bound at 256px (act-ckpt off fastest; bs3 ~22.7 GB on a
  V100-32GB). Now the loader also **decodes video** (not mmap `.npy`), so keep `num_workers`
  modest and watch the `--mem=64G` cgroup ([[acjepa-v100-256-run]]).
* **Offline WavLM cache:** ~9.5k videos in the **`his-extract`** env (transformers + torch 2.6,
  GPU). Rough size ≈ 768-D fp16 @ ~50 Hz over ~35 s ≈ ~2.7 MB/video ≈ **~25 GB** total on
  scratch. Cache is env-decoupled; training reads only `.npy` + `meta.json`.

---

## 7. Milestones

**M0 — revive + build the WavLM cache.**
`git mv` the four `trash/` modules back (`build_audio_features.py`, `audio_cond.py`,
`aucjepa_dataset.py`, `aucjepa_train.py`→`aai_train.py`), then:
```bash
conda activate /scratch1/hongn/conda/envs/his-extract
PYTHONPATH=.:dev_artiJEPA python -m artijepa.build_audio_features \
    --manifest /scratch1/hongn/artijepa/manifest_combined.csv \
    --out /scratch1/hongn/artijepa/audio_feats/wavlm_base_plus_L9 \
    --model microsoft/wavlm-base-plus --layer 9 --limit 20    # smoke, then drop --limit
```
Verify a token-pooled audio vector lands in the same time window as its MRI token (reuse the
phoneme-eval alignment check). Deliverable: ~9.5k `.npy` + `meta.json` (dim, layer,
`audio_rate_hz`, train-split z-score stats).

**M1 — smoke (CPU).** `tests/test_aai_smoke.py`: `RTMRIAudioDataset` returns
`{clip, audio[16,768], valid}`; predictor `action_embed_dim=768` +
`forward_predictions(ctx_frames=2)` = 14 predicted frames; `rollout_l1` finite; one train step
lowers loss on a 2-clip overfit. Confirm the `ctx_frames` path is wired in `aai_train.py`.

**M2 — train the audio-conditioned world model.**
```bash
source dev_artiJEPA/scripts/_env.sh
bash dev_artiJEPA/scripts/16_train_aai.sh $PWD/configs/aai_wavlm_256_combined_ctx2.yaml
# resume per-epoch: ... --resume <abs folder>/latest.pt
```
Watch `diagnostics.jsonl`: `audio_gap` positive and rising; val AR-L1 falling.

**M3 — latent-space eval.** `aai_eval.py` on the M2 ckpt over the held-out split: AR L1,
cosine-vs-horizon, NN retrieval, plus shuffled-audio / no-audio / teacher-forced controls, and
seen-vs-held-out-speaker (§5). Optional NN-frame decode video. Save `aai_results.json` + plots.

**M4 — layer sweep.** Rebuild caches for WavLM {6, 12}, rerun M3; pick the best layer.

**M5 — write up.** Table (AR L1 / cosine / retrieval: real vs shuffled vs no-audio vs TF, seen
vs held-out) into `RESULTS.md`; note `wavlm-large` / weighted-layer follow-ups.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Longitudinal `.avi` audio quality / desync** | verified audio streams exist (mp3@16k); spot-check a few for A/V sync; drop bad rows (dataset already resamples on load failure). |
| **T-SSL combined encoder not converged** | use furthest-along `tssl_vitl_256_combined/latest.pt`, else the finished 75-only `tssl_vitl_256`. |
| **Video-decode loader bottleneck** (no image cache) | tile mode + decord is the T-SSL norm; compute-bound on the encoder anyway; if loader-bound, pre-tile or cache frames later. |
| **WavLM last layer weak for articulation** | default layer 9; sweep {6,9,12} (M4); v2 = learnable weighted sum. |
| **`audio_gap` tiny (audio ignored)** | ctx=2 is the strongest lever; if flat, drop to ctx=1, add FiLM, or lengthen the audio-only horizon. |
| **Rollout barely beats no-audio** | try `wavlm-large` / mid layers before concluding acoustics add little at this encoder/rate. |
| **`--mem=64G` cgroup OOM** | `num_workers:4, pin_mem:false, persistent_workers:false`. |
| **Multi-speaker leakage in eval** | carve a speaker-disjoint held-out split (§5.0) before reporting. |

---

## 9. Open questions

1. **WavLM layer** — start at 9; learnable weighted-sum now or v2?
2. **Eval split** — how many speakers to hold out; retrieval within-video vs across the pool?
3. **Encoder** — combined (data-matched, mid-run) vs 75-only (finished) as the frozen backbone?
4. **Resolution** — 256px (matches encoder) vs 128px (`aucjepa_vitl_128`, 4× fewer tokens, faster) for a quick first pass?
5. **`wavlm-large`** (1024-D) later for a stronger acoustic front-end? (more cache/compute).

---

### Relationship to existing work
This **revives the acoustic AUC-JEPA** ([[aucjepa-ddp-256-run]], now in `trash/`) — audio→video
latent — on the combined rtMRI-75 + longitudinal pool ([[combined-tssl-256-run]]), and grafts
on the `ctx_frames:2` hard-conditioning fix proven in the arti line ([[acjepa-redesign]] /
[[acjepa-v100-256-run]]). Unlike both prior lines it is **label-free**: no arti-6, no phonemes
— the target is the encoder's own future MRI video embeddings, and inversion is measured
entirely in that latent space.
