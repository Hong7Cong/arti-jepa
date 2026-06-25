# Acoustic-Conditioned JEPA (AC-JEPA-audio) — design & build plan

**One-line goal.** Freeze the V-JEPA 2 (or T-SSL-adapted) ViT context encoder and train
**only a predictor that is conditioned on synchronized audio embeddings** (WavLM /
Wav2Vec2), so the predictor learns the *acoustics → vocal-tract dynamics* mapping in
frozen feature space. Audio plays the exact role that robot **actions** play in
V-JEPA 2-AC.

> **Why this is the right framing.** V-JEPA 2-**AC** (the "action-conditioned world
> model" stage) does *exactly* what you're asking — it **freezes the V-JEPA 2 encoder**
> and trains *only* an action-conditioned predictor that rolls the latent state forward
> given robot actions. We are reusing that recipe verbatim and swapping the control
> signal: **per-frame audio embedding instead of the 7-D robot action/state**. For
> rtMRI this is well-motivated — the audio at time *t* is essentially a noisy readout of
> the articulator configuration at time *t*, so it is a *causally necessary* conditioning
> signal for predicting how the vocal tract moves.

This plan is written so each module and each design choice is explained and easy to
swap. Everything new lives under `dev_artiJEPA/` and **reuses the parent repo's
encoder / predictor / RoPE machinery without modifying it** (same rule the README states
for T-SSL).

---

## 0. TL;DR of the design decisions (so you can change them fast)

| Decision | Default (recommended) | Cheaper / alternative | Where to flip it |
|---|---|---|---|
| **What is trainable** | predictor + audio-projection MLP only; encoder + target **frozen** | also unfreeze last N encoder blocks (LoRA-style) | optimizer param-groups (§6) |
| **Target encoder** | the **same frozen encoder** (no EMA, no collapse risk) | EMA copy (only if you ever unfreeze) | §3.2 |
| **Prediction objective** | **temporal / world-model**: encode context frames, predict *future* frame tokens | masked-block 3D (reuse `MaskCollator`) | `objective:` in config (§4) |
| **Audio model** | `microsoft/wavlm-base-plus` (768-D, ~50 Hz) | `facebook/wav2vec2-base` (768-D) | `audio.model_name` (§5) |
| **Audio features** | **pre-computed + cached to `.npy`** (env-decoupled) | on-the-fly frozen WavLM in-proc | `audio.cache_dir` / `audio.online` (§5) |
| **Audio injection** | **state + action dual-token** per frame (`state = e[t]`, `action = e[t+1]−e[t]`), **distinct** projections — AC predictor reused **verbatim** | single audio token (`add_tokens=1`) or FiLM/additive | `predictor.kind` (§3.4–3.5) |
| **Audio normalization** | **per-dim z-score** of `e` (corpus stats) before forming state/action | raw embeddings | `audio.normalize` (§3.4) |
| **Speaker conditioning** | **off** (audio-only, clean story) | speaker emb → `extrinsics` slot (`add_tokens=3`) | `audio.speaker` (§3.5) |
| **Encoder init** | the **T-SSL** domain-adapted checkpoint (`runs/tssl_vitl_256/latest.pt → target_encoder`) | raw V-JEPA 2 `vitl.pt` | `model.checkpoint` (§7) |
| **Rollout** | teacher-forced + 1–2 autoregressive steps | teacher-forced only | `optimization.auto_steps` (§4) |

---

## 1. How it maps onto the existing repo

We mirror the structure of `tssl_train.py` (single-process, one V100, grad-accum,
resumable, label-free diagnostics) but change three things: the **predictor** (audio-
conditioned), the **dataset** (also returns aligned audio embeddings), and the
**objective** (temporal future-prediction instead of/in addition to spatial masking).

| Component | T-SSL today | AC-JEPA-audio | Reuse? |
|---|---|---|---|
| Context encoder | `vit_large`, **trained** w/ EMA | `vit_large`, **frozen** (`eval()`, `no_grad`) | reuse `model.build_models` encoder half |
| Target | EMA momentum copy | **= the frozen encoder** (a `deepcopy`, never updated) | simplify |
| Predictor | `vit_predictor` (mask tokens) | `vit_ac_predictor` reused **verbatim** (state+action tokens) | reuse `src/models/ac_predictor.py` |
| Conditioning | none | `state = e[t]`, `action = e[t+1]−e[t]`, **distinct** projections | **new** (`audio_cond.py`) |
| Data item | `([clip], label, [idx])` | `([clip], audio_emb, [idx])` | extend `rtmri_dataset.py` |
| Loss | L1(pred, LN(target)) on masked tokens | L1(pred, LN(target)) on **future** tokens (+AR) | reuse pattern |
| Optimizer | encoder+predictor | **predictor + audio proj only** | new param-group helper |
| Diagnostics | feature_std / eff_rank / cosine | same + **audio-pred MSE / teacher-forcing gap** | reuse `collapse.py` |

Reference points already in the codebase:
- Standard predictor: `src/models/predictor.py::VisionTransformerPredictor` (factory `vit_predictor`).
- **Action-conditioned predictor we will mimic:** `src/models/ac_predictor.py::VisionTransformerPredictorAC`
  — `forward(x, actions, states, extrinsics=None)`; the action MLP is
  `self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)`, and tokens
  are injected by **per-frame concat**: `x = torch.cat([a, s, x], dim=2).flatten(1, 2)`
  (see `ac_predictor.py:136-192`). Frame-causal attention mask + RoPE are already built in.
- AC factory/wiring lives in `app/vjepa_droid/utils.py::init_video_model` (vs. the masked
  one used by T-SSL in `app/vjepa/utils.py::init_video_model`). The droid training engine
  `app/vjepa_droid/train.py` (`forward_target`, `forward_predictions`, `loss_fn`,
  `auto_steps`) is the exact loop we adapt.
- Audio decode already exists: `artijepa/audio_phoneme.py::extract_audio(path, sr=16000)`
  (ffmpeg → mono float32). Token↔seconds alignment: `artijepa/phonemes.py::token_center_times`.

---

## 2. Intuition / the objective in words

A 32-frame, 50 fps clip (0.64 s) becomes **`T' = frames_per_clip / tubelet_size = 16`
temporal tokens** × `H*W = (256/16)² = 256` spatial tokens after the encoder. We:

1. Run the **frozen** encoder on the **whole clip** → target tokens
   `h ∈ [B, T'·H·W, D]` (layer-normed). This is the regression target (fixed, so no
   representation collapse is possible — the target net never moves).
2. Take a **context prefix** of `k` frames (e.g. the first `T'_ctx` temporal tokens)
   and roll the **predictor** forward frame-by-frame, **conditioned on the audio
   embedding of each frame**, to predict the remaining (future) target tokens.
3. Loss = `L1(predicted future tokens, target future tokens)`, teacher-forced + a short
   autoregressive rollout (exactly `jloss + sloss` in `app/vjepa_droid/train.py:439`).

Because the encoder is frozen, **the predictor + tiny audio-projection MLP are the only
trainable parameters** (~the predictor's 12×384 transformer). The audio stream is what
makes the future predictable, so the model is forced to *use* the acoustics.

> **Alternative objective (toggle, not headline):** keep the V-JEPA spatial
> mask-denoising (reuse `MaskCollator`) and just hand the predictor the audio tokens as
> extra context. Smaller diff, but audio is less *necessary* (a lot of masked tokens are
> guessable from visible spatial neighbours), so it is a weaker test of acoustic
> conditioning. Good as an ablation. Select with `objective: masked`.

---

## 3. Modules (what each is, and the choice behind it)

### 3.1 Frozen context encoder — `vit_large`
- **What:** the V-JEPA 2 ViT-L wrapped in `MultiSeqWrapper`, built by
  `artijepa/model.py::build_models` (the encoder half). We **do not** touch its weights.
- **Choice:** *frozen* because (a) you asked for it, and (b) it matches V-JEPA 2-AC,
  where the world model is learned on top of fixed perceptual features. Practical wins:
  run the context pass under `torch.no_grad()` + `encoder.eval()` → big VRAM/throughput
  savings, and **activation checkpointing on the encoder becomes unnecessary** (no
  backward through it). You can likely raise `batch_size` vs. T-SSL.
- **Init choice:** prefer the **T-SSL domain-adapted** weights
  (`runs/tssl_vitl_256/latest.pt`, key `target_encoder`) — they already moved toward
  rtMRI and gave the headline κ gains. Fall back to raw `vitl.pt` for an apples-to-apples
  ablation. Controlled by `model.checkpoint` / `model.checkpoint_key`.

### 3.2 Target encoder — the **same** frozen encoder
- **What:** `make_target_encoder(encoder)` already returns a frozen `deepcopy`
  (`model.py:87`). Since the context encoder never updates, **target = context**; we
  simply never run the EMA update.
- **Choice:** drop the entire EMA/`momentum_scheduler` block from the loop. This removes
  the only source of collapse and a chunk of bookkeeping. (Keep an `ema:` config field
  unused, so that if you later *unfreeze* the encoder you can re-enable EMA without a
  schema change.)
- **Note:** in V-JEPA 2-AC `forward_target` for *robot* data does a per-frame
  `repeat(1,1,2,1,1)` hack (`vjepa_droid/train.py:408`) because DROID frames are single
  images. **We do not need that** — rtMRI clips are real video, so we encode the clip
  natively (the T-SSL `forward_target` path) and reshape `[B, T'·H·W, D]`.

### 3.3 Audio feature extractor (WavLM / Wav2Vec2) — **cached, env-decoupled**
- **What:** for each clip, decode its audio (`extract_audio`, already in the repo →
  16 kHz mono), run a frozen HF audio encoder, and save per-frame hidden states
  `[T_audio, A]` (A=768 for `*-base`) to `.npy`. ~50 Hz output rate conveniently sits
  near the 50 fps video grid.
- **Choice — why cache offline:** the README/`audio_phoneme.py` already document a hard
  env conflict — the `artijepa` env pins **torch 2.6** but `transformers 5.x` needs
  **torch ≥ 2.7**. So **do not import `transformers` inside training.** Mirror the
  `build_pseudo_labels` pattern: a *decoupled batch step* (`build_audio_features.py`) run
  in a transformers-compatible env writes `.npy` + a small `meta.json` (model name,
  frame rate, dim); training reads only NumPy. This also makes training fast and
  deterministic.
- **Choice — which model:** `wavlm-base-plus` (robust, speech-pretrained, 768-D) as
  default; `wav2vec2-base` as a drop-in. Use the **last hidden state** (not CTC logits)
  — we want a *representation*, not phoneme posteriors. Make `layer` a config knob (an
  intermediate layer often encodes articulation better than the last).
- **Optional `audio.online: true`:** if you build a single env with torch ≥ 2.7, you can
  load a frozen WavLM in-process and skip the cache. Keep it behind the flag; default off.

### 3.4 Audio → state/action conditioning + temporal alignment — `audio_cond.py` (new)

**Semantic parallel to V-JEPA 2-AC.** The robot predictor takes *two* conditioning
signals with the **same dimensionality but different meaning**, each with its **own
learned projection** (`ac_predictor.py:53-54`): `state` = *absolute* pose, `action` =
*frame-to-frame delta* (`droid.py:222` → `poses_to_diffs`, `droid.py:137-147`). The
delta is fully derivable from the state, yet both are fed — a deliberate **inductive
bias** that hands the predictor "what changed" without making it difference internally.
We mirror this **exactly**, so the AC predictor is reused **verbatim** (no trimming):

| V-JEPA 2-AC (robot) | Acoustic analog | Shape | Meaning |
|---|---|---|---|
| `state` = absolute pose `[T,7]` | **`e[t]`** — audio embedding at frame *t* | `[T', A]` | absolute point in acoustic space → which tract shape (a phone ⇒ a configuration) |
| `action` = pose delta `[T−1,7]` | **`e[t+1] − e[t]`** — acoustic delta | `[T'−1, A]` | how the acoustics change ⇒ how the tract moves |
| `extrinsics` = camera pose (context) | **speaker embedding** (optional) | `[T', A']` | per-clip context — the 75-speaker variability |

> **Manifold note:** `poses_to_diffs` uses a *proper SO(3) relative rotation*
> (`R[t+1] @ R[t]ᵀ`) because euler angles aren't Euclidean. Audio embeddings live in
> (approximately) Euclidean ℝ^A, so the delta is **plain subtraction** `e[t+1] − e[t]`
> — `poses_to_diffs` collapses to a `np.diff`. No manifold correction.

`audio_cond.py` has three responsibilities:

1. **Temporal alignment / pooling.** Encoder temporal tokens live at the **tubelet
   rate**: token *j* covers output frames `[j·tubelet, (j+1)·tubelet)`, centred at
   `token_center_times(...)` (`phonemes.py:55`). The cached audio stream is ~50 Hz. For
   each of the `T'` temporal tokens, **average-pool the audio frames whose timestamps fall
   in that token's window** → one vector per temporal token → `e ∈ [B, T', A]`. Reuse
   `token_center_times` so alignment is identical to the phoneme eval (any video fps works).
2. **Normalization (`audio.normalize`, default `zscore`).** Robot poses are in bounded
   physical units (m, rad); audio embeddings are not, so a few high-variance dims can
   dominate the raw delta. **Per-dim z-score `e` with corpus stats** (computed once in the
   offline cache step, stored in `meta.json`) *before* forming state/action, so both
   projection heads see comparable scales. Set `audio.normalize: none` to disable.
3. **State / action construction.** From the pooled, normalized `e`:
   ```python
   state  = e                 # [B, T',   A]   absolute  -> state_encoder  = Linear(A, D)
   action = e[:, 1:] - e[:, :-1]   # [B, T'-1, A]   delta     -> action_encoder = Linear(A, D)
   # pairing mirrors train.py:430 — predicting frame t+1 uses state[t] + action[t]
   ```
   Both projections are **trainable** and, with the predictor, are the *entire* trainable
   set. Keep them as submodules of the predictor adapter so they serialize with it.

### 3.5 Audio-conditioned predictor — `vit_ac_predictor` reused verbatim
- **Primary (`predictor.kind: ac_audio`).** Wrap `VisionTransformerPredictorAC` in a thin
  adapter (`AudioConditionedPredictor`) so we don't edit parent-repo files:
  - Build via `app/vjepa_droid/utils.py::init_video_model` with `action_embed_dim = A`
    (audio dim). State and action share `A`, so the stock `state_encoder` /
    `action_encoder` (`ac_predictor.py:53-54`) drop in unchanged — `add_tokens=2`.
  - At call time, feed `actions = e[:, 1:] − e[:, :-1]` and `states = e` exactly where the
    robot loop feeds `actions, states[:, :-1]` (`train.py:430`). The per-frame injection
    we inherit is `ac_predictor.py:147-153`:
    ```python
    s = self.state_encoder(states).unsqueeze(2)      # [B, T', 1, D]   absolute audio
    a = self.action_encoder(actions).unsqueeze(2)    # [B, T', 1, D]   audio delta
    x = torch.cat([a, s, x], dim=2).flatten(1, 2)    # [B, T'*(H*W+2), D]
    ```
  - You get **frame-causal attention** (`build_action_block_causal_attention_mask`) and
    **3-axis RoPE** for free — exactly what makes the temporal rollout work.
- **Speaker conditioning (`audio.speaker`, default off).** The clean home for speaker
  identity is the **`extrinsics`** slot (the robot's "context/viewpoint" axis): set
  `use_extrinsics=True` (`add_tokens=3`) and feed a per-clip speaker embedding broadcast
  over `T'`. One caveat — the stock `extrinsics_encoder` is hardwired to
  `action_embed_dim − 1` (`ac_predictor.py:55`); since we own the adapter, give it its own
  `Linear(spk_dim, D)` instead. Good ablation for the 75-speaker variability; off by
  default for a clean audio-only story.
- **Alternatives.** *Single audio token* (`add_tokens=1`, state-only or action-only) if
  you want to test whether the dual-token inductive bias matters. *FiLM* (`predictor.kind:
  film`): keep the standard `vit_predictor` and add `audio_proj(e)` broadcast over each
  frame's spatial tokens before the blocks — smallest diff, no sequence-length change, but
  loses frame-causality/rollout, so pair it with the `masked` objective.

### 3.6 Loss
- `L1` in layer-normed feature space, identical to T-SSL (`tssl_train.py:247`) and droid
  (`vjepa_droid/train.py:439`): `mean(|pred − LN(target)|^loss_exp)/loss_exp`,
  `loss_exp=1.0`.
- World-model objective: `loss = jloss + sloss` (teacher-forced + autoregressive). Set
  `auto_steps` (1 = teacher-forced only). Start with `auto_steps: 2`.

---

## 4. The two objectives (one flag: `objective`)

### `objective: temporal` (recommended, the true AC analog)
- **Context** = first `ctx_frames` temporal tokens (config; e.g. 8 of 16). **Target** =
  the remaining `T' − ctx_frames` future tokens.
- Predictor is conditioned on the audio of the frames being predicted (teacher-forced),
  then rolled out autoregressively for `auto_steps`.
- No `MaskCollator`; the dataloader just returns `(clip, audio_emb)`. This is the cleanest
  and makes audio causally necessary.

### `objective: masked` (ablation / smaller diff)
- Keep T-SSL's multiblock-3D `MaskCollator` (`masking.mask_config_for`) untouched.
- Encode masked context, predict masked targets, **but also feed the predictor the
  per-frame audio tokens** (FiLM or concat). Compares "masked-only" vs "masked+audio" to
  isolate the audio contribution.

> Recommendation: build `temporal` first (it's the point of the request), wire `masked`
> as a second branch behind the same config once the data/predictor plumbing works.

---

## 5. Data pipeline

### New offline step — `dev_artiJEPA/artijepa/build_audio_features.py`
Mirror `build_pseudo_labels` (`audio_phoneme.py:85`):
```
build_audio_features(manifest, out_dir, model_name="microsoft/wavlm-base-plus",
                     layer=-1, target_rate=50.0, device=...):
  for each clip:
    wav = extract_audio(path, sr=16000)            # reuse existing ffmpeg helper
    feats = wavlm(wav).hidden_states[layer]        # [T_audio, A], frozen, no_grad
    np.save(out_dir/<stem>.npy, feats.astype(float16))   # fp16 to save disk
  # accumulate per-dim mean/std over the TRAIN split for audio.normalize=zscore
  write meta.json: {model, layer, A, audio_rate_hz, sr, mean[A], std[A]}
```
- Run **once** in a transformers-compatible env (same caveat as pseudo-labels). ~2.4k
  clips; cache is a few GB at fp16.
- Store the **native ~50 Hz** features (don't pre-pool) so you can retime to any
  `target_fps`/`tubelet` later in the dataloader.

### Dataset — extend `rtmri_dataset.py` (new `RTMRIAudioDataset` or a flag)
- Subclass/extend so `__getitem__` returns `(clips, audio_emb, clip_indices)` where
  `audio_emb` is `[T', A]` already pooled to temporal tokens for that clip/chunk.
- Reuse `RTMRIVideoDataset._load_clip` for the video. For audio: load the cached `.npy`,
  compute `token_center_times(T', tubelet, target_fps, clip_start_frame=chunk*F)`
  (`phonemes.py:55`), window-average the audio frames per token (the alignment logic
  already exists in `audio_phoneme.PseudoPhonemeDataset._token_labels` — generalize it
  from "nearest id" to "mean-pool over the token window").
- **Mask out-of-range tokens** (clip tail with no audio) by emitting a per-token validity
  mask so the loss can ignore them (analogous to `IGNORE_INDEX`).
- A small **collate_fn** stacks clips + audio (and forwards `MaskCollator` only in the
  `masked` objective).

---

## 6. Training loop — `dev_artiJEPA/artijepa/accjepa_train.py` (new, forked from `tssl_train.py`)

Start from `tssl_train.py` and change:

1. **Build models** (`build_models`): keep the encoder; build the **audio predictor** via
   the droid `init_video_model` (or our adapter), passing `action_embed_dim = A`,
   `pred_depth`, `pred_embed_dim`, `pred_num_heads`, `use_rope=True`,
   `pred_is_frame_causal=True`.
2. **Freeze encoder.** `for p in encoder.parameters(): p.requires_grad_(False)` and
   `encoder.eval()`. Run the context/target encode under `torch.no_grad()`.
3. **Optimizer = predictor + audio_proj only.** `init_opt` builds param-groups from
   `encoder.named_parameters()` **and** `predictor.named_parameters()`
   (`app/vjepa/utils.py:228-241`). Frozen encoder params have `requires_grad=False`, so
   AdamW won't update them, **but** they still bloat optimizer state. Cleanest fix: add a
   tiny `init_opt_predictor_only(...)` (copy of `init_opt` that only groups
   `predictor` + `audio_proj`, filtered by `requires_grad`). Pass `audio_proj` either as
   part of the predictor module (preferred — make it a submodule of the predictor adapter
   so it serializes with the predictor) or as a second module in the param-groups.
4. **Drop EMA.** Remove `momentum_scheduler` and the `_foreach_mul_/_add_` update; target
   is the frozen encoder.
5. **forward_target / forward_context / loss_fn:** replace with the temporal-rollout
   versions adapted from `vjepa_droid/train.py:408-449` (teacher-forced + `auto_steps`
   AR), feeding `audio_emb` where the droid code feeds `actions`. For `objective: masked`,
   keep T-SSL's `forward_context`/`loss_fn` but pass audio into the predictor.
6. **Keep** grad-accum, fp16+GradScaler, resume/atomic-checkpoint, CSV log,
   `diagnostics.jsonl` — all already correct in `tssl_train.py`. Checkpoint now stores
   `{"predictor", "audio_proj", "opt", "scaler", "epoch"}` (no `encoder`/`target` needed —
   they're frozen and reproducible from `model.checkpoint`; optionally store an encoder
   hash to guard against mismatched init on resume).

**Pseudilike sketch of the core step (temporal objective):**
```python
encoder.eval()
with torch.no_grad():                                  # frozen perceptual features
    h = encoder(clip)                                  # [B, T'*H*W, D]
    h = F.layer_norm(h, (h.size(-1),))
ctx = h[:, :ctx_frames * HW]                           # context prefix tokens
state  = e                                             # [B, T', A]   absolute audio
action = e[:, 1:] - e[:, :-1]                          # [B, T'-1, A] audio delta
# predictor's state_encoder/action_encoder project + interleave (ac_predictor.py:147);
# pairing mirrors train.py:430 — frame t+1 sees state[t] + action[t]
z_tf = predictor(ctx, actions=action, states=state[:, :-1])         # teacher-forced
z_ar = rollout(predictor, ctx, action, state, auto_steps)           # mirrors vjepa_droid forward_predictions
loss = l1(z_tf, h[:, ctx_frames*HW:]) + l1(z_ar, h[:, ctx_frames*HW:])
```
(`predictor` is the adapter wrapping `VisionTransformerPredictorAC`; the trainable
`state_encoder`/`action_encoder` live inside it. `e` is the pooled, normalized audio
from §3.4.)

---

## 7. Config — `dev_artiJEPA/configs/accjepa_vitl_256.yaml`

Fork `tssl_vitl_256.yaml`; annotated new/changed fields:
```yaml
meta:
  folder: /scratch1/hongn/artijepa/runs/accjepa_vitl_256
  seed: 0
  dtype: float16            # V100 → fp16 + GradScaler (same as T-SSL)
  eval_freq: 1
  save_freq: 1

data:
  manifest: /scratch1/hongn/artijepa/manifest_split.csv
  spatial_size: 256
  frames_per_clip: 32
  target_fps: 50.0
  tubelet_size: 2
  patch_size: 16
  sampling: tile            # full temporal coverage; one chunk == one (clip, audio) item
  batch_size: 32            # can likely raise — encoder is frozen (no backward through it)
  num_workers: 2

audio:                      # NEW
  model_name: microsoft/wavlm-base-plus
  layer: -1                 # hidden layer to read (try mid-layers for articulation)
  dim: 768                  # A; must match cache meta.json
  cache_dir: /scratch1/hongn/artijepa/audio_feats/wavlm_base_plus
  pool: mean                # per-token temporal pooling over the audio window
  normalize: zscore         # per-dim z-score (corpus stats in meta.json) | none
  speaker: false            # true → add speaker emb in the extrinsics slot (add_tokens=3)
  online: false             # true → run frozen WavLM in-proc (needs torch>=2.7 env)

objective: temporal         # temporal | masked
ctx_frames: 8               # context temporal tokens (of T'=16); rest are predicted
predictor:
  kind: ac_audio            # ac_audio (state+action dual-token) | film
  # state = e[t], action = e[t+1]-e[t]; distinct projections, AC predictor verbatim
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
  use_rope: true
  frame_causal: true

model:
  model_name: vit_large
  checkpoint: /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt   # domain-adapted
  checkpoint_key: target_encoder                                       # frozen encoder init
  use_activation_checkpointing: false   # not needed: no backward through the encoder

loss:
  loss_exp: 1.0

optimization:
  epochs: 50
  ipe: 500
  effective_batch: 128      # grad-accum like T-SSL
  auto_steps: 2             # teacher-forced + 1 AR step
  lr: 5.0e-4
  start_lr: 1.0e-4
  final_lr: 1.0e-5
  warmup: 5
  weight_decay: 0.04
  final_weight_decay: 0.04
  betas: [0.9, 0.999]
  eps: 1.0e-8
  # ema: unused while the encoder is frozen (kept for future unfreeze)
```

---

## 8. New files (responsibilities + signatures)

| File | Responsibility | Key API |
|---|---|---|
| `artijepa/build_audio_features.py` | offline WavLM/Wav2Vec2 → cached `.npy` + `meta.json` | `build_audio_features(manifest, out_dir, model_name, layer, device)` |
| `artijepa/audio_cond.py` | align + normalize audio, build state/action, predictor adapter | `pool_audio_to_tokens(feats, audio_rate, T', tubelet, target_fps, chunk)`, `to_state_action(e)` (`e[1:]−e[:-1]`), `AudioConditionedPredictor(nn.Module)` (owns `state_encoder`/`action_encoder`) |
| `artijepa/accjepa_train.py` | training loop (fork of `tssl_train.py`) | `train(cfg)`, `forward_predictions`, `rollout` |
| `configs/accjepa_vitl_256.yaml` | primary config | — |
| `configs/accjepa_vitl_128.yaml` | 128px ablation (cheaper smoke) | — |
| `scripts/08_build_audio_feats.sh` | run the offline cache step | wraps `build_audio_features` |
| `scripts/09_train_accjepa.sh` | launch training (mirror `03_train_tssl.sh`) | — |
| `tests/test_accjepa_smoke.py` | shape/grad smoke (ViT-tiny, 2 steps, fake audio) | asserts encoder grads are None, predictor grads non-None |

Reused unchanged: `model.build_models` (encoder), `checkpoint.resolve_checkpoint/
load_pretrained`, `rtmri_dataset` helpers, `collapse.py`, `phonemes.token_center_times`,
`audio_phoneme.extract_audio`.

---

## 9. Build order (each step independently verifiable)

1. **Audio cache (offline).** Implement `build_audio_features.py`; run on ~20 clips
   (`--limit 20`) in a transformers env; eyeball `.npy` shapes `[T_audio, 768]` and
   `meta.json` rate ≈ 49–50 Hz. *Verify:* `np.load` a file, check `audio_rate*duration ≈ T_audio`.
2. **Alignment unit test.** `pool_audio_to_tokens` returns `[T', 768]` with the right
   validity mask for a known clip; assert token 0's window matches `token_center_times`.
3. **Predictor adapter.** `AudioConditionedPredictor`: build trimmed `vit_ac_predictor`,
   verify a forward `(ctx[B,N,D], audio[B,T',A]) → [B, N_pred, D]` runs on CPU with
   ViT-tiny dims; assert the frame-causal mask shape matches `T'*(H*W+1)`.
4. **Freeze + optimizer.** Wire `accjepa_train.py`; assert after one `loss.backward()`
   that **all encoder grads are None** and predictor+audio_proj grads are finite
   (this is the core "encoder frozen" guarantee — put it in the smoke test).
5. **Smoke run.** `scripts/09_train_accjepa.sh configs/accjepa_vitl_128.yaml --max-steps 5`
   end-to-end; loss must decrease over a few hundred steps on a single clip (overfit test).
6. **Full 128px run**, watch `diagnostics.jsonl` (eff_rank shouldn't be needed for
   collapse since target is frozen, but track the **teacher-forced vs AR loss gap** and a
   held-out **audio-conditioned prediction MSE**).
7. **256px run** with the domain-adapted encoder init.

---

## 10. Evaluation

- **Primary downstream:** the existing **phoneme probe** (`eval_phoneme.py`, spatial
  `attentive` probe — the README headline). Probe the **predictor's rolled-out features**
  (or `encoder+predictor`) vs. the encoder-only T-SSL features; the hypothesis is that
  audio-conditioned dynamics sharpen articulation-relevant features → higher κ / lower PER.
- **Intrinsic sanity:** audio-conditioned future-prediction MSE on held-out clips, and an
  **ablation: real audio vs. shuffled/zeroed audio** — if predicting the future barely
  needs the audio token, conditioning isn't working. This is the key diagnostic that the
  module actually *uses* the acoustics.
- **Reuse** the label-free `collapse.py` metrics on encoder features (unchanged baseline).

---

## 11. Risks / knobs to tune (where to look first if it underperforms)

- **Audio not used** (predictor ignores the state/action tokens). Fix: smaller `ctx_frames`
  (harder task), more `auto_steps`, check `audio.normalize` is on (un-normalized deltas can
  be near-zero and get ignored), try action-only (the delta is the *causal* signal), or
  FiLM into *every* spatial token. Check the real-vs-shuffled-audio gap.
- **Rate mismatch / misalignment.** WavLM is ~49.95 Hz, video grid is 50 Hz, tubelet=2 →
  token rate 25 Hz. Get the seconds-based pooling right (reuse `token_center_times`);
  a half-token offset will quietly cap performance.
- **fp16 instability** in the predictor rollout (AR amplifies). Mirror T-SSL's GradScaler;
  if NaNs appear, lower `lr`, drop to `auto_steps:1`, or do the rollout in fp32.
- **Encoder choice.** Domain-adapted (T-SSL) vs raw V-JEPA 2 is the cleanest ablation —
  run both; report κ for each.
- **Predictor capacity.** It's the only learner now; try `pred_depth` 12→16 and
  `pred_embed_dim` 384→512 if it underfits (VRAM is freed by the frozen encoder).
- **Layer of WavLM.** Last layer is ASR-biased; mid-layers often carry more articulatory
  info. `audio.layer` is a cheap sweep (re-cache per layer, or cache all layers once).

---

## 12. Open decisions for you

1. **Objective:** ship `temporal` (world-model, recommended) first, or start with the
   smaller-diff `masked` ablation?
2. **Encoder init:** domain-adapted T-SSL checkpoint (recommended) vs raw V-JEPA 2 — or
   both as the headline ablation?
3. **Audio model & layer:** `wavlm-base-plus` default; do you want a Wav2Vec2 / HuBERT
   comparison and a layer sweep budgeted in?
4. **Speaker conditioning:** `state = e[t]` / `action = e[t+1]−e[t]` is fixed (the faithful
   AC mapping); the open call is whether to *also* add a per-clip **speaker embedding** in
   the `extrinsics` slot (`audio.speaker: true`, helps 75-speaker variability) or keep it
   audio-only for a clean story.
5. **Trainable surface:** strictly predictor-only (recommended), or also unfreeze the last
   1–2 encoder blocks for a small extra gain?

---

*Built on: `tssl_train.py` (loop), `src/models/ac_predictor.py` (conditioning),
`app/vjepa_droid/{utils,train}.py` (AC wiring + rollout loss), `audio_phoneme.py`
(audio decode + cache pattern), `phonemes.py` (token↔seconds alignment). Nothing in the
parent `src/` is modified.*
