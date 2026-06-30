# AC-JEPA-audio (AUC-JEPA) — Progress & TODO

Implementation of `plans_aucjepa.md`: freeze the (T-SSL domain-adapted) V-JEPA 2
ViT-L encoder, train **only an audio-conditioned predictor** that rolls the latent
state forward given synchronized WavLM audio embeddings (audio plays the role of
robot *actions* in V-JEPA 2-AC). Only edit `dev_artiJEPA/`; parent `src/`/`app/`
are read-only reference (reuse `src/models/ac_predictor.py` verbatim).

**Env / node (2026-06-22):** training/eval uses conda `artijepa` (torch 2.6+cu124).
GPU currently free = **Tesla P100-PCIE-16GB** (small → smoke at 128px). Audio
feature extraction needs transformers → use **`his-extract`** env (torch 2.6+cu124
+ transformers 4.56.2): it drives the P100 *and* loads WavLM (the `artijepa` env's
transformers import hangs — the documented torch2.6/transformers5.x conflict).

Key artifacts (all under `/scratch1/hongn/artijepa/`, never `/project2`):
- frozen encoder init: `runs/tssl_vitl_256/latest.pt` key `target_encoder` (ep50, κ 0.530)
- manifests: `manifest_alltrain.csv` (pretrain), `manifest_split.csv` (train/val/test)
- audio cache (NEW): `audio_feats/wavlm_base_plus/<stem>.npy` + `meta.json`

---

## Build order (plan §9) — status

- [x] **1. Audio cache (offline).** `build_audio_features.py` done; FULL corpus
      cached on the V100 (2371/2371 clips, 0 failures; corpus z-score stats in
      `meta.json` over 3,485,801 frames / 2243 train clips; ~50.18 Hz, dim 768).
- [x] **2. Alignment unit test.** `pool_audio_to_tokens` window pooling + validity mask
      verified (token0=mean(0,1), token4=mean(8,9), tail invalid, clip_start offset).
- [x] **3. Predictor adapter.** `AudioConditionedPredictor` forward + rollout shapes
      verified on CPU (ViT-tiny).
- [x] **4. Freeze + optimizer.** `aucjepa_train.py` wired; smoke asserts **ALL encoder
      grads None** + state/action-encoder grads finite after `loss.backward()`. ✅ 18/18.
- [x] **5. Smoke run (P100, 128px).** ✅ End-to-end on the Tesla P100-16GB.
      Encoder loaded **clean (292/0/0)** from `tssl_vitl_256` even at 128px (RoPE →
      resolution-flexible). 23.0M trainable predictor vs 303.9M frozen encoder.
      fp16, no NaN. **Overfit loss 1.63→1.00** over 8 epochs (auto_steps=2) and
      1.63→1.23 over 5 epochs (ctx_frames=8). Both jloss(TF) and sloss(AR) drop.
      Step time ~1.6s (auto_steps=2) / ~4.2s (ctx_frames=8, 8 sequential AR steps).
- [~] **6./7. Full 256px run** (primary config) RUNNING on the **Tesla V100-32GB**
      (2026-06-23). 256px + the 8-step ctx-prefix AR rollout retains a huge backward
      graph → OOMs at batch ≥6 on 32GB. **Throughput benchmark (2026-06-23)** settled
      the VRAM question: the run is **compute-bound, not memory-bound** — batch 8→16
      (+activation-ckpt) gave *identical* throughput (0.924→0.945 clips/s), so a bigger
      batch buys nothing. Fastest viable = **batch 4 + `use_activation_checkpointing:
      false`** (1.070 clips/s, **1.16×** vs batch-8+ckpt) — removing the ~30% recompute
      directly cuts compute. ~25 GB, deterministic shapes (tile sampling → every step
      identical), survives cudnn autotune. eff_batch 128 via `ipe 2000` (accum 32, 62
      oue/epoch → T_max 1240 unchanged). **20 epochs** ("first result"; cosine anneal
      by ep20, warmup 2 = 10% shape), **~3.7 s/step, ~41 h** total (epoch 1 ran at
      batch-8+ckpt, then resumed at batch-4 no-ckpt from `latest.pt`). Resumable
      per-epoch → extend to 50 ep later if loss + audio_gap still improving.
      NB: if raising batch ≥6, MUST set `use_activation_checkpointing: true` (else OOM).

## Smoke results (2026-06-22, P100, 128px, 20 train + 8 val clips cached)
| objective | epochs | loss start→end | jloss(TF) | sloss(AR) |
|---|---|---|---|---|
| droid AR (auto_steps=2) | 8 | 1.60 → 1.00 | 0.80→0.51 | 0.80→0.49 |
| ctx-prefix (ctx_frames=8) | 5 | 1.63 → 1.23 | 0.81→0.61 | 0.82→0.62 |

**Audio real-vs-shuffled gap** (val, plan §10 — does the predictor *use* audio?):
measured on the AR branch (where audio is necessary). `auto_steps=2`: gap ≈ 4e-5
(≈0 — audio barely used; next-frame video is guessable from past frames, plan §11
risk #1). `ctx_frames=8`: gap grows 2.1e-4 (e2) → 6.4e-4 (e4) — **positive and
rising** as training proceeds, i.e. the model starts to rely on audio. Magnitudes
are tiny because: only ~4 epochs, lr still in warmup, val is ONE speaker (sub012)
so within-speaker shuffle is "plausible" audio (smaller penalty), frozen features.

### Making audio *necessary* — knobs for the full run (plan §11 risk #1)
- **`ctx_frames`** (added): context-prefix rollout predicts T'−K frames from audio
  only. Lower K = harder = audio more necessary. Configs default `ctx_frames: 8`.
- More epochs past warmup; **cross-speaker** shuffle in the diagnostic; action-only
  ablation; FiLM into every spatial token; check the gap trend over training.

## Files (plan §8)
| File | Status |
|---|---|
| `artijepa/build_audio_features.py` | ✅ |
| `artijepa/audio_cond.py` | ✅ |
| `artijepa/aucjepa_dataset.py` (RTMRIAudioDataset) | ✅ |
| `artijepa/aucjepa_train.py` | ✅ |
| `configs/aucjepa_vitl_256.yaml` / `aucjepa_vitl_128.yaml` | ✅ |
| `scripts/08_build_audio_feats.sh` / `09_train_aucjepa.sh` | ✅ |
| `tests/test_aucjepa_smoke.py` | ✅ (18/18 checks pass) |
| `artijepa/aucjepa_train_ddp.py` (DDP trainer, both GPUs) | ✅ |
| `configs/aucjepa_vitl_256_ddp.yaml` / `aucjepa_vitl_128_ddp.yaml` | ✅ |
| `scripts/10_train_aucjepa_ddp.sh` (torchrun launcher) | ✅ |

## Design decisions locked (plan §0 defaults)
- Trainable = AC predictor + its state/action Linear projections only; encoder + target **frozen**.
- Target = the frozen encoder itself (no EMA, no collapse risk).
- Objective = **temporal** world-model: encode whole clip (frozen) → predict next-frame
  tokens, teacher-forced + `auto_steps` AR (droid recipe, audio as actions/states).
- Audio = `wavlm-base-plus` last hidden (768-D, ~50 Hz), cached fp16, per-dim z-score (corpus stats).
- Injection = state `e[t]` + action `e[t+1]−e[t]`, distinct Linear(A,D), AC predictor verbatim (`add_tokens=2`).
- Encoder init = T-SSL `tssl_vitl_256/latest.pt` `target_encoder`.

## How to run
```bash
# 0. one-time: cache WavLM audio features for the FULL corpus (his-extract env;
#    ~2.4k clips, ~1-2 h on the P100, a few GB fp16). Smoke used --limit 20.
bash dev_artiJEPA/scripts/08_build_audio_feats.sh            # full manifest_alltrain
#    smoke subset only:  ... 08_build_audio_feats.sh --limit 20

# 1. train (artijepa env). 128px ablation (fits P100) or 256px primary:
source dev_artiJEPA/scripts/_env.sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash dev_artiJEPA/scripts/09_train_aucjepa.sh dev_artiJEPA/configs/aucjepa_vitl_128.yaml
#    quick check:  ... aucjepa_vitl_128.yaml --max-steps 50
#    resume:       ... --resume /scratch1/hongn/artijepa/runs/aucjepa_vitl_128/latest.pt
#    256px primary (needs a faster GPU than the P100): aucjepa_vitl_256.yaml

# 1b. MULTI-GPU (DDP) -- BOTH GPUs via torchrun (one process/GPU, NCCL). Independent
#     of the single-GPU trainer; checkpoints are interchangeable (unwrapped predictor).
bash dev_artiJEPA/scripts/10_train_aucjepa_ddp.sh dev_artiJEPA/configs/aucjepa_vitl_256_ddp.yaml
#    auto-detects GPUs from CUDA_VISIBLE_DEVICES; 128px DDP: aucjepa_vitl_128_ddp.yaml
#    resume:  ... aucjepa_vitl_256_ddp.yaml --resume /scratch1/hongn/artijepa/runs/aucjepa_vitl_256_ddp/latest.pt

# smoke test (CPU, no GPU/transformers needed): 19/19 checks
PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_aucjepa_smoke.py
```
Outputs per run under `meta.folder`: `train_log.csv` (loss/jloss/sloss/lr/wd),
`diagnostics.jsonl` (val TF/AR L1 + `audio_gap`), `latest.pt` (predictor/opt/
scaler/epoch — resumable; encoder NOT stored, reloaded frozen from `model.checkpoint`).

## Remaining work
- [x] **Full-corpus audio cache** — DONE 2026-06-22: all 2371 clips cached on the
      V100 (2323 new, 0 fail), corpus stats recomputed over 2243 train clips.
- [~] **Full training run** — LAUNCHED 2026-06-22 on the V100 at 256px, 20 epochs
      (see step 6/7 above). ~48 h, resumable. Monitor `train_log.csv` +
      `diagnostics.jsonl` (`audio_gap`). To extend: `09_train_aucjepa.sh
      aucjepa_vitl_256.yaml --resume .../runs/aucjepa_vitl_256/latest.pt` after
      raising `epochs`.
- [ ] **Downstream eval** (plan §10): phoneme probe on predictor rolled-out features
      vs encoder-only T-SSL (reuse `eval_phoneme.py`).
- [ ] **Objective: masked** branch (plan §4 ablation) — currently asserts temporal.
- [ ] **Speaker conditioning** (`audio.speaker`, extrinsics slot) — adapter supports
      it (`use_extrinsics`/`spk_dim`); not yet wired into the dataset/train loop.

## Notes / decisions log
- (2026-06-25) **Multi-GPU (DDP) 256px run RUNNING on 2x Tesla P100-16GB** ->
  `runs/aucjepa_vitl_256_ddp` (SEPARATE from the V100 `runs/aucjepa_vitl_256`).
  New independent path: `aucjepa_train_ddp.py` (torchrun/NCCL, frozen encoder
  replicated per rank, only the AC predictor DDP-wrapped via a `forward`->
  `forward_predictions` shim so grad-sync arms; `DistributedSampler`; rank-0-only
  log/diag/ckpt; checkpoints store the *unwrapped* predictor -> 1:1 interchangeable
  with `aucjepa_train.py`). `aucjepa_vitl_256_ddp.yaml`: **batch 2/gpu x world 2 =
  global 4** (== the V100 batch-4 trajectory), eff_batch 128 (accum auto = 32), oue
  62/epoch.
  **THE non-obvious DDP gotcha — `static_graph=True` is REQUIRED:** `VisionTransformer
  PredictorAC` ALWAYS constructs an `extrinsics_encoder` that is UNUSED when
  `use_extrinsics=False`, AND the predictor backbone is invoked 9x/step (1 TF + 8 AR
  rollout). With a plain reducer DDP raises "Expected to have finished reduction... params
  not used in producing loss" *at the first all-reduce* (it surfaced at the accum
  boundary because `no_sync()` had suppressed reduction earlier). `static_graph=True`
  records the (deterministic tile-sampled) graph once -> tolerates unused params +
  repeated module calls + non-reentrant ckpt, and keeps `extrinsics_encoder` trainable
  so optimizer param-groups still match the single-GPU trainer. NB static_graph does
  NOT compose with `no_sync()`, so the trainer all-reduces every micro-step (negligible;
  compute-bound). `init_process_group(device_id=...)` binds rank->GPU.
  **256px on Pascal is COMPUTE-bound (both GPUs ~100%), so VRAM/batch does NOT speed it
  up** (V100 bench: batch 8->16 = same throughput). Measured peaks: act-ckpt **ON ~3.5 GB**
  (safe) vs **OFF ~14.8 GB** (near-max on 16 GB; AdamW state allocates ~+0.2 GB at the 1st
  optimizer boundary -> watch it there). ckpt-OFF skips the recompute -> **~14.25 s/step
  (~11% faster than ckpt-ON's 16 s)** -- the ONLY VRAM-for-speed lever; the 8-step AR
  rollout is what makes it slow, not memory. **Launched as a PROBE: ckpt OFF + 6 epochs
  (~7.9 h/epoch -> ~2 days), loss dropping, fits 16 GB.** Resumable per-epoch (spans
  allocations): `bash scripts/10_train_aucjepa_ddp.sh dev_artiJEPA/configs/aucjepa_vitl_256_ddp.yaml
  --resume runs/aucjepa_vitl_256_ddp/latest.pt`. Faster turnaround = 128px DDP
  (`aucjepa_vitl_128_ddp.yaml`, ~1/4 the tokens) or flip ckpt ON + 20 ep for the safe long run.
- (2026-06-24) **cgroup OOM at epoch 5** (NOT GPU): the salloc was `--mem=32G`; a
  DataLoader worker's anon/shmem crept over epochs (+ page cache from decoding 2371
  videos) and SIGKILLed at the epoch-5 `torch.save` spike (`Killed`, no traceback;
  confirmed in dmesg `oom_memcg=.../job_9548533`). Fix: `num_workers 4→2`,
  `pin_mem false`, and `persistent_workers=False` (respawn per epoch releases the
  creep). Resumed clean from epoch-4 `latest.pt`; host RAM now ~16 GB/32. If it
  recurs, drop to `num_workers: 1` or request a larger `--mem`. Monitor now watches
  the cgroup usage and warns at 28 GB.
- (2026-06-24) Allocation note: salloc `--time=47:00:00`; the 20-epoch run (~41 h
  net) will NOT fit one allocation — it checkpoints per epoch, so finish it across
  allocations by re-running `09_train_aucjepa.sh aucjepa_vitl_256.yaml --resume
  .../runs/aucjepa_vitl_256/latest.pt` in a fresh salloc (ideally `--mem=64G`).
- (2026-06-22) Naming = `aucjepa` (matches `plans_aucjepa.md` / this file); plan body
  text occasionally says `accjepa` — same thing.
- Token order from `encoder.backbone(clip)` is **temporal-major** (`[B,T'·S',D]`,
  view→`[B,T',S',D]`), matching `ac_predictor.view(B,T,H*W,D)`. Verified vs `eval_phoneme.py:182`.
- `forward_predictions(...)` returns `(z_tf, z_ar, ar_start)`. Two AR modes: droid
  (`auto_steps` from frame 0) and the plan's §4 **context-prefix** (`ctx_frames` real
  frames → predict the rest from audio only). Configs default to `ctx_frames: 8`.
- Audio extraction env = **`his-extract`** (torch 2.6+cu124 + transformers 4.56),
  driven on the P100. `artijepa` env's transformers (5.x) import hangs.
- WavLM-base-plus output: 768-D, ~50.1 Hz (verified T_audio ≈ rate·duration per clip).
- Embedding cache format: one `<stem>.npy` per clip = `[T_audio,768]` fp16 (RAW WavLM
  last-hidden, un-pooled, un-normalized) + shared `meta.json` (per-dim z-score corpus
  stats). Pooling→tokens and z-scoring are deferred to the dataloader (no re-extract
  when fps/resolution/normalization change).
