# AUC-vJEPA *planning*: phoneme-as-goal, articulators-as-actions — design & build plan

**One-line goal.** Reuse the frozen V-JEPA 2 / T-SSL encoder + an **articulator-conditioned
forward model** (`P: arti-action + video-state → next video-state`), then **plan a sequence
of articulator movements** that drives the rolled-out video latent toward a **target phoneme**
(1 of the fixed inventory), scored by an **energy function** on the predicted latent. This is
V-JEPA 2-AC's planning recipe (CEM / receding-horizon MPC), with the robot's end-effector
replaced by the 6-D articulator vector and the goal image replaced by a phoneme.

> Companion to [`plans_aucjepa.md`](plans_aucjepa.md), which builds the *forward-prediction*
> AUC-JEPA (audio→video). **This plan is a different regime — planning, not prediction —
> and it swaps the roles of the conditioning signal.** Read §0 first; the role-swap is the
> single most important thing to get right.

---

## 0. The role-swap (read this first)

| | `plans_aucjepa.md` (forward prediction) | **this plan (planning)** |
|---|---|---|
| Conditioning / **action** `a` | **audio** embedding (768-D, WavLM) | **articulators** `arti-6` (Δ between frames) |
| **State** `s` | audio absolute `e[t]` | articulators absolute `arti[t]` (6-D) |
| Predicted **z** | video latent | video latent (**same**) |
| **Goal** | none (just predict) | **phoneme `p*`** (1 of 39) |
| Inference | one forward pass | **CEM search over `a^{1:T}`** (many forward passes) |
| What you optimize | model weights (train) | **the action sequence** (at test time; weights frozen) |

So audio is **no longer** the conditioner here — the **articulators** are. The phoneme is the
**goal**, and goals exist *only* because we are planning (searching over actions). The forward
model itself is trained exactly like the existing AUC-JEPA, just with `A = 6` (articulators)
instead of `A = 768` (audio).

### Mapping to V-JEPA 2-AC

| V-JEPA 2-AC | AUC-vJEPA planning |
|---|---|
| end-effector pose `s` (7-D) | articulator state `s = arti[t]` (6-D) |
| action `a` = Δ end-effector (7-D) | action `a` = `arti[t+1] − arti[t]` (6-D) |
| visual latent `z` | rtMRI latent `z` (frozen encoder output) |
| goal = encoded image `z_g` | **goal = phoneme `p*`** → energy via a classifier/prototype bridge |
| energy `‖P(a^{1:T}) − z_g‖₁` | `−log C(p* | P(a^{1:T}))` (recommended) — see §3 |
| CEM + receding horizon | **identical** |

Articulator inventory note: the codebase ships a 41-symbol ARPABET set
(`artijepa/phonemes.py:30`, incl. `sil`). "39 phonemes" = the standard CMU/TIMIT reduced set;
pick **one** canonical inventory and freeze its index order — the planner, the classifier head,
and the goal table must all agree. Below uses `K = NUM_PHONEMES` (41 today; set to 39 if you
collapse).

---
## 0.5. Data inventory and Data pipeline

`arti-6` lives at `/scratch1/hongn/usc_lss/articulators/*.mat` and exists **only** for USC LSS. I have not inspected these files; everything below marked **[VERIFY]** is an assumption to confirm in Step 0. Get this right and the rest is mechanical; get it wrong (especially time alignment) and every downstream metric is silently corrupt.

### 1.1 Time alignment (the make-or-break step)
 
The forward model interleaves `(a_t, s_t, z_t)` per frame, so **arti-6 and the encoder's frame stream must be sample-synchronous**. Two pitfalls:
 
1. **fps mismatch.** V-JEPA 2-AC sampled robot video at **4 fps** because arm motion is slow. **Speech articulation is fast** — stop closures/releases occur on ~10–50 ms timescales. Sampling at 4 fps would temporally alias every consonant. **Plan: ingest frames at (or near) the native rtMRI reconstruction rate**, and resample arti-6 to exactly match that frame index. Confirm the native rate in Step 0; do not inherit `4 fps` from the robot recipe.
2. **Index origin / trimming.** If video and arti-6 were trimmed differently (leading silence, reconstruction warmup), a constant frame offset will quietly wreck the action signal. Validate alignment on a few utterances by checking that large `‖a_t‖` spikes coincide with visible articulatory movement in the frames.
Action definition (matches the planning doc and the robot Δ-convention):
```
s_t  = arti6[t]                  # absolute articulator state, 6-D
a_t  = arti6[t+1] - arti6[t]     # action = articulator delta, 6-D
```
 
### 1.2 Encoder usage (frozen)
 
- Encode **each rtMRI frame independently as an image** with the frozen rtMRI encoder `E` (V-JEPA 2 / T-SSL continued-pretrained), → `z_t ∈ R^{H×W×D}`. This mirrors how V-JEPA 2-AC used its encoder as a per-frame image encoder. **[VERIFY]** the encoder's output token grid `H×W` and dim `D` for your resolution.
- *Alternative to consider:* use the encoder's native 2-frame tubelet instead of frame-independent encoding. Frame-independent is the safer default (clean 1:1 frame↔state correspondence); note the choice and don't mix them.
- Encoder is **frozen throughout** — no gradients, no updates. Only the predictor trains.
### 1.3 Splits (two generalization axes)
 
Build **speaker-disjoint** splits — a world model meant to plan must generalize across vocal tracts, so seen-speaker metrics alone are misleading. Also retain a **seen-speaker / unseen-utterance** condition to separate "new sentence" difficulty from "new anatomy" difficulty.
 
```
USC LSS speakers  →  TRAIN / VAL / TEST_speaker   (speaker-disjoint)
                                                    e.g. 70% / 15% / 15% of speakers
Within TRAIN+VAL speakers:
   hold out a set of utterances → TEST_utt        (seen speaker, unseen utterance)
```
- **[VERIFY]** speaker count to fix the exact ratios; if few speakers, use k-fold over speakers instead of a single test split.
- Report all §4 metrics **separately** on `TEST_speaker` (hard, cross-anatomy) and `TEST_utt` (easier). The gap is itself a result.
- Freeze the split (write `splits/{train,val,test_speaker,test_utt}.json`) so every experiment is comparable.
### 1.4 Normalization
 
- Fit per-dimension mean/std of `arti6` (and of the deltas `a`) on **TRAIN only**; apply to all splits. Store stats in the split dir.
- Keep raw (un-normalized, in mm) deltas too — `a_clip` and the CEM init in the planner must be set in **interpretable physical units** (§5).
- Mask dropped-tracking frames; never let a NaN propagate into a delta.
---

## 1. The key constraint: discrete goal vs. continuous prediction

The predictor outputs a **continuous** latent `z ∈ [B, n·hw, D]` (`hw = tokens_per_frame`,
`audio_cond.py:122`). The goal `p*` is a **discrete symbol**. You **cannot** write
`‖z − p*‖`. You must insert a **bridge** that maps the continuous prediction into phoneme space.
Three valid energies (pick one; default = #1):

### Energy 1 — classifier NLL  *(recommended, most general)*
Train a phoneme head `C: latent → softmax over K`. For a single target phoneme `p*` reached at
the end of a `T`-step rollout:
```
ẑ_T = P(a^{1:T}; z_k, s_k)                # rolled-out latent of the final frame, [B, hw, D]
ℰ(a^{1:T}) = − log softmax( C( pool(ẑ_T) ) )[p*]
```
`pool` = mean over the `hw` spatial tokens of the final frame (or use the per-token head and
average the log-probs). This is cross-entropy to one-hot(`p*`). `C` *is* the discrete↔continuous
bridge; it tolerates many articulatory realizations of the same phoneme (coarticulation).

### Energy 2 — phoneme prototype  *(keeps the L1 form of V-JEPA 2-AC)*
Precompute `μ_p` = mean **layer-normed** latent for each phoneme `p` over the training set
(one `[hw, D]` prototype per phoneme). Then:
```
ℰ(a^{1:T}) = ‖ ẑ_T − μ_{p*} ‖₁
```
Drop-in identical shape to `rollout_l1` (`audio_cond.py:186`); replace the future-frame target
with the phoneme prototype. Simple, but assumes one canonical latent per phoneme.

### Energy 3 — articulatory target  *(most interpretable, no learned head)*
Each phoneme has a canonical 6-D articulator config (table `s*_p`, estimated as the mean
`arti` over frames labeled `p`). Plan directly in articulator space:
```
ℰ(a^{1:T}) = ‖ ŝ_T − s*_{p*} ‖     where ŝ_T = s_k + Σ_t a_t
```
Purest Task-Dynamics / Articulatory-Phonology view (phoneme = gestural target). Needs **no**
forward video model at all if you only care about reaching the articulatory target — but then
`z` plays no role, so use this only as a cheap baseline / regularizer.

> **Recommendation:** **Energy 1** as the headline (uses `z`, handles coarticulation, gives a
> probabilistic score). Keep **Energy 3** as a cheap sanity baseline and an optional additive
> regularizer `ℰ_total = ℰ_1 + λ·ℰ_3`.

---

## 2. Two things to train (both reuse existing code) — then planning is inference-only

### 2a. Forward model `P` — the articulator-conditioned predictor
Reuse `AudioConditionedPredictor` (`audio_cond.py:90`) **verbatim** with `action_embed_dim = 6`.
Everything downstream is unchanged: `state = arti[t]`, `action = arti[t+1]−arti[t]` via
`to_state_action` (`audio_cond.py:76`), `forward_predictions` teacher-forced + AR rollout,
`rollout_l1` loss. Train with the existing DDP trainer (`aucjepa_train_ddp.py`) — just point
`audio.cache_dir` at an **articulator** cache and set `audio.dim: 6`.
- Requires an `arti-6` stream per clip, aligned to temporal tokens by the **same**
  `pool_audio_to_tokens` window logic (`audio_cond.py:35`). The 6 features are the articulator
  state cache (e.g. tract variables / tracked landmarks). No code change to the pooler.
- Output: `runs/aucjepa_arti6_<res>/latest.pt` — a frozen world model for planning.

### 2b. Phoneme head `C` — the energy bridge (only for Energy 1/2)
Reuse the existing phoneme probe (`configs/eval_phoneme_*.yaml`, `probe.type: tcn|mlp|linear`).
Train `C: z → K` on **encoder latents** with gold/pseudo phoneme labels (the eval pipeline
already does exactly this). Freeze it for planning. For Energy 2, instead compute prototypes
`μ_p` from the same labeled latents — no head needed.

> Neither 2a nor 2b is new training machinery — both already exist. **Planning adds no
> training**; it is a pure test-time optimizer wrapping the two frozen models.

---

## 3. The planner (new code — the only genuinely new module)

New file: `artijepa/aucjepa_plan.py`. Pure inference. Operates **entirely in latent space**
(no pixel decoding — the JEPA advantage).

```
inputs: frozen encoder, frozen P (2a), frozen C (2b), target phoneme p* (or sequence)
        seed: ctx_frames real frames of a clip  -> z_k (latent), s_k (arti state)

CEM(p*):
  init action-seq distribution  N(μ=0, σ=σ0)  over a^{1:T}  (T×6 Gaussians)
  repeat n_iter:
    sample M candidate action seqs  a^{1:T}_m
    for each m:                                # vectorize over M as the batch dim
        ẑ = P._ar_rollout(z_k, s_from(a_m), a_m, start=ctx_frames, n=T)   # audio_cond.py:134
        E_m = energy(ẑ, p*)                    # Energy 1/2/3 from §1
    keep top-k lowest-E candidates -> refit μ, σ
  return argmin_E action seq; execute a_1; (optional) re-encode & re-plan  # receding horizon
```

- `s_from(a)` = cumulative sum of actions from `s_k` (absolute arti states the predictor needs).
- Reuse `P._ar_rollout` (`audio_cond.py:134`) for the forward rollout — it already seeds with
  `start` real frames and rolls the rest from state/action only. Batch the `M` candidates so one
  rollout call scores the whole population.
- **Sequence goal** `p*_{1..L}` (a phoneme *string*): use **sub-goals** (exactly how V-JEPA 2-AC
  handles multi-step pick-and-place) — one energy term per phoneme along the horizon,
  `ℰ = Σ_t energy(ẑ_t, p*_t)`. If the frame↔phoneme alignment is free, score with a **CTC**
  alignment over the rollout instead of a fixed per-frame assignment (the repo already has CTC /
  PER machinery: `phonemes.py:98 reference_sequence`, `:141 phoneme_error_rate`).

---

## 4. Concrete file/diff plan

| File | New/changed | What |
|---|---|---|
| `artijepa/arti_cache.py` | **new** | build the per-clip `arti-6` `.npy` cache + `meta.json` z-score stats (mirror the audio cache). Skip if arti features already cached. |
| `configs/aucjepa_arti6_256.yaml` | **new** | copy `aucjepa_vitl_256_ddp.yaml`; set `audio.dim: 6`, `audio.cache_dir: <arti cache>`, `audio.normalize: zscore`. |
| `scripts/11_train_aucjepa_arti6.sh` | **new** | wrap `10_train_aucjepa_ddp.sh` with the arti config (forward model 2a). |
| phoneme head | **reuse** | train via existing `eval_phoneme_*.yaml`; save `C` (or prototypes `μ_p`). |
| `artijepa/aucjepa_energy.py` | **new** | `energy_classifier_nll`, `energy_prototype`, `energy_arti_target` (§1). Pure functions on `(ẑ, p*)`. |
| `artijepa/aucjepa_plan.py` | **new** | CEM / receding-horizon MPC loop (§3); loads frozen encoder + `P` + `C`; single-phoneme and sub-goal/CTC sequence modes. |
| `configs/aucjepa_plan_256.yaml` | **new** | planner hyperparams: `T`, `ctx_frames`, `cem.{M, top_k, n_iter, sigma0}`, `energy.kind`, `lambda_arti`. |
| `scripts/12_plan_aucjepa.sh` | **new** | run the planner over a goal phoneme / phoneme string. |

No edits to parent-repo code; no edits to `audio_cond.py` (reused as-is with `A=6`).

---

## 5. Evaluation / what "it works" means

1. **Reachability (single phoneme):** for each `p*`, does CEM find an arti-seq whose rolled-out
   latent classifies as `p*`? Report top-1 accuracy of `argmax C(ẑ_T)` vs `p*` across the 39/41
   inventory. Baseline: Energy 3 (articulatory-target only).
2. **Trajectory realism:** compare planned arti-trajectories to **held-out real** arti-6 for the
   same target phoneme (RMSE / DTW). Sanity that the planner finds plausible gestures, not latent
   adversarial junk.
3. **Sequence goal:** plan a phoneme string; measure **PER** (`phonemes.py:141`) of the
   rolled-out latent re-classified by `C`, and arti-trajectory smoothness.
4. **Ablations:** Energy 1 vs 2 vs 3; with/without the `λ·ℰ_3` regularizer; `ctx_frames` (more
   context = easier); CEM `M`/`n_iter` budget vs reachability.

**Reality check (from our earlier discussion):** if you only need *articulation-from-sound*, the
existing audio→video AUC-JEPA already gives it by direct prediction — planning earns its keep
**only** when you add coarticulation (sequences), articulator constraints (velocity/reachability
limits in CEM sampling), or one-to-many inversion (multiple gestures → same phoneme). Make sure
the eval includes at least one of these, or the planner has nothing to do over direct prediction.

---

## 6. Milestones

- **M0** — build `arti-6` cache (`arti_cache.py`); verify `pool_audio_to_tokens` alignment on a
  few clips (token windows match `phonemes.token_center_times`).
- **M1** — train forward model `P` with `A=6` (reuse `aucjepa_train_ddp.py`); confirm AR-rollout
  L1 / teacher-forcing gap in `diagnostics.jsonl` is comparable to the audio model.
- **M2** — train/load phoneme head `C` (or prototypes `μ_p`); confirm probe accuracy on val.
- **M3** — `aucjepa_energy.py` + a **1-step** CEM sanity (T=1) on Energy 3 (closed-form arti
  target) — fastest signal the loop is wired right.
- **M4** — full CEM (§3) with Energy 1; single-phoneme reachability eval (§5.1).
- **M5** — sequence goals via sub-goals/CTC; PER + trajectory realism (§5.2–5.3); ablations.

## 7. Risks / open decisions

- **Do you even need `z`?** If 6 articulators fully describe the state, Energy 3 plans in 6-D with
  a trivial dynamics model — far cheaper. Add the video model (`P`, Energy 1) only if the rtMRI
  latent carries phonetic info the 6 features don't. Decide with an M2 probe: how well does `C`
  predict phonemes from `z` vs. from `arti-6` directly?
- **Single-phoneme planning may be trivial** (one gesture target → "move there"); the research
  value is in sequences / coarticulation / constraints (see §5 reality check).
- **CEM in high-D latent energy can be exploited** — adversarial arti-seqs that fool `C` without
  realistic gestures. Mitigate with the `λ·ℰ_3` arti-target regularizer and velocity-bounded CEM
  sampling.
- **Inventory consistency:** freeze ONE phoneme index order across `C`, the goal table, and the
  prototypes (39 vs the repo's 41-symbol ARPABET).

---

## 8. Empirical finding (2026-06-30): the forward model ignores the articulators

**What we ran.** M1 forward model `P` (frozen T-SSL ViT-L + arti-conditioned AC predictor,
`A=6`, `ctx_frames=8`) trained 8/20 epochs at 256px on the usc_lss sessions (`runs/acjepa_arti6_256_ddp`).

**Result.** The world-model objective trains well — val L1 falls monotonically (TF
`0.629→0.408`, AR `0.642→0.440` over epochs 1→8). **But `arti_gap` (real-vs-shuffled-arti AR
L1, the "does it USE the articulators?" probe) plateaus at `5e-4–1e-3`** — only **~0.1–0.2 %
of the AR L1** — and is non-monotone (rose to `1.0e-3` @ ep7, fell to `6e-4` @ ep8) while LR
was already annealing down. **The predictor reconstructs the future from visual frame dynamics
and barely uses arti-6.** Same failure mode as the trashed acoustic run (`audio_gap≈0`) and the
§11 risk #1 — but here there is a deeper, structural reason:

### 8.1 Root cause — the articulators are a *readout of the observation*, not a hidden action

V-JEPA 2-AC works because the robot **action** (torque / end-effector delta) is **NOT visible
in the image** — conditioning injects genuinely new information. **arti-6 is the opposite: the
six constriction degrees are measured *from the very MRI image the encoder already sees*.** So
given the past frames, the articulatory state is largely *inferable from the pixels*, and
conditioning on it adds little NEW information for next-frame prediction. The conditioning
signal is near-deterministic in the observations ⇒ the gradient to *use* it is weak. Secondary
contributors: `ctx_frames=8` seeds half the clip (the rest is guessable by motion continuity, so
arti is never the bottleneck); only 2 conditioning tokens/frame vs 256 visual tokens; and a
single-speaker shuffle is a weak negative.

### 8.2 Proposed solution (ordered; A+C first)

- **A — Make arti the bottleneck (config-only, cheapest; TRY FIRST).** Drop `ctx_frames` to
  **1–2** and let the arti-only rollout run ~14–15 frames. Over a long arti-only horizon, frame-
  dynamics extrapolation decays, so the model *must* route future prediction through arti to stay
  accurate → forces controllability. Smoke → short run, watch `arti_gap`. (New config:
  `acjepa_arti6_256_ctx2_ddp.yaml`.)
- **C — Answer §7 "do we even need `z`?" with the M2 probe (run in PARALLEL).** Train a phoneme
  head from `z` vs from `arti-6` directly. If arti-6 alone predicts phonemes ≈ as well as `z`,
  the articulators and the video latent are **near-redundant** ⇒ the planner does **not** need the
  video world model: plan directly in 6-D arti space with **Energy 3** (cheap, interpretable,
  ALREADY implemented in `acjepa_energy`/`acjepa_plan`). Under this view, "P ignores arti" is not a
  bug to fix but evidence for the **right** design — and planning earns its keep via
  sequences / coarticulation / velocity constraints (§5 reality check), not via `z`.
- **B — Raise conditioning bandwidth (predictor-adapter change).** FiLM an arti-derived `(γ,β)`
  into *every* spatial token of *every* frame (cf. §3.5) instead of 2 side tokens. Harder to
  ignore; needs a small adapter mod (no parent-repo edit).
- **D — Change the prediction target so arti is causal (heavier; only if A–C inconclusive).**
  Predict skip-`K` future frames (dynamics extrapolation is harder over a gap) or the residual the
  pure-visual model cannot, so arti carries information the past frames don't.

**Recommendation.** Run **A** (`ctx_frames=2`) and **C** (the M2 redundancy probe) next. If
neither makes arti necessary, the honest conclusion is that arti-conditioned *video forward
modeling* is low-value on this data (redundant signal), and the project pivots cleanly to
**Energy-3 arti-space planning**, where the implemented modules already work. The epoch-8 `P`
checkpoint is kept as the ctx_frames=8 baseline for the ablation.

### 8.3 Fix A (`ctx_frames=2`) RESULT — 2026-07-01: it works, but arti is inherently weak

Ran fix **A** from scratch, full 20 epochs, single-GPU V100-32GB (`runs/acjepa_arti6_256_v100_ctx2`,
config `acjepa_arti6_256_v100_ctx2.yaml`; ckpt-off, batch 3 — the 1TF+14AR = 15-call rollout needs
~1.6× the VRAM of ctx=8, bench: bs3=22.7 GB safe / bs4=29 GB too close to 32 GB). Compute-bound,
~3.65 s/step, ~61 min/epoch. (Also ran a ctx=8 V100 baseline `runs/acjepa_arti6_256_v100`, killed
@ep6, that reproduced the ~5e-4 plateau.)

**`arti_gap` broke the plateau and climbed ~10×.** It tracked the ctx=8 ~5e-4 line through ep8,
then rose monotonically from ep9: ep9 1.6e-3 → ep12 4.5e-3 → **peak ep19 6.9e-3**, ep20 5.6e-3
(settled in a noisy ~5–7e-3 band; ep17 dipped to 1.6e-3 — the estimate is noisy, see caveats).
That is **~10–14× the ctx=8 ceiling** ⇒ **forcing a long arti-only horizon does make the predictor
use the articulators.** Meanwhile the world model kept sharpening (val_ar_l1 0.66→0.428).

**BUT the gap is still small in absolute terms (~1.3 % of AR L1)** — exactly as §8.1 predicts (arti
is a *readout*, so there's a low ceiling on how much conditioning can help next-frame prediction).
A fairer denominator is the **AR-rollout penalty** `ar_real − tf` = 0.428 − 0.404 = 0.025: the right
articulators recover **~22 %** of it, so arti is not negligible for the rollout, just capped. Two
measurement effects likely *understate* it: the diagnostic shuffles **within-batch** (batch 3 → each
"wrong" arti drawn from only 2 others, weak negative), and averages only `probe_max_batches=8` val
batches (noisy). A global shuffle + all-val re-measure on the ep20 ckpt would tighten it.

**Verdict / next.** Fix A succeeded directionally but confirms arti-conditioned video forward
modeling has limited headroom here. **Do NOT chase a bigger `arti_gap`** — run **C (M2 redundancy
probe)** to decide the pivot: if arti-6 predicts phonemes ≈ as well as `z`, go to Energy-3 arti-space
planning and treat the small gap as evidence for that design. **The M2 utterance↔session alignment
blocker is now RESOLVED** (whole-sequence frame-match of the per-utt `.avi` to the cached session
`IMAGE` → offset → slice frame-exact arti-6; validated). See `TODO_acjepa.md` M2.
