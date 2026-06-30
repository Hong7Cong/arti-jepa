# AC-JEPA (Articulator-Conditioned JEPA) — Progress & TODO

Implementation of [`aucjepa_plans_new.md`](aucjepa_plans_new.md): freeze the (T-SSL
domain-adapted) V-JEPA 2 ViT-L encoder, train **only an articulator-conditioned
predictor** that rolls the latent state forward given the synchronized **6-D
articulator vector** `arti-6` (articulators play the role of robot *actions* in
V-JEPA 2-AC), then **plan** a sequence of articulator movements that drives the
rolled-out latent toward a **target phoneme** (CEM / receding-horizon MPC). Only edit
`dev_artiJEPA/`; parent `src/`/`app/` are read-only (reuse `src/models/ac_predictor.py`
verbatim with `action_embed_dim = 6`).

> **This REPLACES the acoustic AUC-JEPA** (`audio → video` forward prediction). The
> old plan + its entire implementation were moved to [`trash/`](trash/) on 2026-06-28
> (full reset). The acoustic 256px DDP run on 2×P100 (`runs/aucjepa_vitl_256_ddp`) is
> **SUPERSEDED / abandoned** — its on-disk artifacts are left untouched but no longer
> tracked here. Naming pivot: old `aucjepa` (**A**co**u**stic) → new **`acjepa`**
> (**A**rticulator-**C**onditioned). The new-plan body keeps an `aucjepa_` file prefix;
> we use `acjepa_` consistently to distinguish from the trashed acoustic code.

---

## ⚠️ Critical data discovery (2026-06-28) — corrects the plan's §0.5 assumption

The plan assumed `arti-6` could be bolted onto the **75-speaker** training manifest
(`speaker75/sub0NN`, ~83 fps, 2371 clips). **It cannot** — those videos have no
articulator tracking. The arti `.mat` files are a **different, self-contained corpus**:

- **Corpus = `/scratch1/hongn/usc_lss`** — the single-speaker (`usc_s1`) gold-phoneme
  OOD set (`phonemes.py:8`; 104×104 @ ~100 Hz, 16 kHz audio, gold ARPABET timestamps).
- **`articulators/usc_s1_<NN>_mview.mat`** (71 **session** files) each hold, all at
  **100 Hz and frame-exact with each other**:
  `AUDIO (16 kHz)`, `IMAGE [104,104,T] uint8` (the MRI frames), and the **6 constriction
  signals** `Bilabial, Alveolar, Palatal, Velum, Pharyngeal, Larynx` (`[T,1]` each).
- ⇒ **arti-6 and the MRI frames live in the SAME file at the SAME rate.** Reading both
  from the one `.mat` makes them frame-exact **by construction** — the plan's
  make-or-break alignment risk (§1.1: fps mismatch, trim offset) is **eliminated**, not
  merely mitigated. We do NOT use the per-utterance `.avi` for the forward model.

**Consequences baked into the redesign:**
- Forward model `P` (M0–M1) trains on **usc_lss sessions** (video+arti from `.mat`),
  `target_fps = 100` (native, no consonant aliasing).
- Splits are **session-disjoint** (one speaker ⇒ speaker-disjoint impossible; documented
  fallback per plan §1.3). 71 sessions → ~70/15/15% train/val/test.
- **Phoneme head `C` / planner goal-labels (M2–M5) have an OPEN [VERIFY]:** the gold
  phoneme `.json` are per-**utterance** (`usc_s1_<NN>_<rep>.json`, 684) while the cache
  is per-**session** (`usc_s1_<NN>_mview`, 71). Mapping utterance phoneme timing into
  session frame time (offset within the session recording) is **not yet resolved** — it
  blocks Energy-1/2 and the phoneme-reachability eval, NOT the forward model.

---

## Build order (plan §6 milestones) — status

- [x] **M0 — arti cache + manifest BUILT (2026-06-28).** `bash scripts/11_build_arti_cache.sh`
      cached all **71 sessions** (0 fail) → `/scratch1/hongn/artijepa/arti_feats/usc_lss/`
      `<stem>.arti.npy` `[T,6]` fp16 + `<stem>.image.npy` `[T,104,104]` uint8 (**3.4 GB**)
      + `meta.json` (dim 6, 100 Hz, per-dim z-score over **228,323** train frames).
      Session-disjoint manifest `arti_manifest.csv`: **49 train / 11 val / 11 test**.
      (NB `du -sh` under-reports on BeeGFS; use `--apparent-size`.)
- [x] **Predictor + dataset + trainers wired.** `arti_cond.ArtiConditionedPredictor`
      (AC predictor verbatim, `A=6`); `acjepa_dataset.RTMRIArtiDataset` (IMAGE+arti from
      the same `.mat`); single-GPU `acjepa_train.py` + DDP `acjepa_train_ddp.py`
      (`static_graph=True` kept — same gotcha as the acoustic DDP).
- [x] **Planner stack (code).** `acjepa_energy.py` (3 energies + bridge builders);
      `acjepa_plan.py` (CEM / receding-horizon, latent-space rollout via `P.rollout`).
- [x] **Smoke test.** `tests/test_acjepa_smoke.py` — CPU/ViT-tiny: pooling, state/action
      round-trip, predictor+rollout shapes, **frozen-encoder guarantee** (all encoder
      grads None), 3 energies, and a closed-form **CEM convergence** check.
      `PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_acjepa_smoke.py`
- [~] **M1 — 256px pipeline SMOKE PASSED (2026-06-28), full run pending.** DDP smoke on
      **2× P100-16GB** (`13_train_acjepa_ddp.sh acjepa_arti6_256_ddp.yaml --max-steps 30`,
      exit 0): encoder loaded **clean at 256px** (`292 tensors, 0 missing, 0 skipped` —
      RoPE handles 104→256), frozen 303.9M + trainable predictor **22.1M**, tokens/frame
      256, A=6, ctx_frames=8, act-ckpt ON. **Fits 16 GB, no OOM/NaN**, ~16 s/step (1st
      25.8 s = cuDNN autotune). Loss sane at random init ~**2.18** (TF 1.09 + AR 1.09);
      diagnostics + atomic `latest.pt` (88 MB, resumable) both exercised. 30 warmup steps
      is a pipeline check, NOT convergence. (Smoke artifacts archived in
      `runs/acjepa_arti6_256_ddp/_smoke/`.)
- [x] **M1 — FULL RUN STOPPED EARLY @ epoch 8/20 (2026-06-30).** Ran 8 epochs on 2× P100
      (`runs/acjepa_arti6_256_ddp`, `latest.pt` = ep8 loss 0.84, `.prev` = ep7; smoke in
      `_smoke/`). **World model trains well** — val L1 falls monotonically (TF 0.629→0.408,
      AR 0.642→0.440). **BUT the key `arti_gap` diagnostic plateaus at 5e-4–1e-3 (~0.1–0.2 %
      of AR L1) and fell back at ep8 → the predictor IGNORES the articulators** (predicts
      future frames from visual dynamics). Stopped early — ~12 more epochs at the same
      `ctx_frames=8` would keep lowering L1 but not make `P` arti-controllable, so a planner
      over it has no lever. **Root cause + fix written up in `aucjepa_plans_new.md` §8.**

      ### ⚠️ The problem & proposed solution (see plan §8 for the full write-up)
      **Why:** unlike a robot action (invisible in the image), the 6 articulators are
      *constriction degrees read off the very MRI image the encoder sees* → given the past
      frames the arti state is already inferable from pixels, so conditioning on it adds
      little NEW info for next-frame prediction (near-redundant signal). Compounded by
      `ctx_frames=8` (rest of clip guessable by motion), 2 cond-tokens/frame vs 256 visual,
      and a single-speaker shuffle (weak negative).
      **Next (A + C first):**
      - **A — LAUNCHED 2026-06-30.** `acjepa_arti6_256_ctx2_ddp.yaml` (`ctx_frames: 2`,
        rollout 8→14 frames, ~280 ms arti-only). Running on 2× P100, ~18 s/step (~5 h/epoch),
        act-ckpt ON fits 16 GB. `runs/acjepa_arti6_256_ctx2_ddp/`. **Watch `diagnostics.jsonl`
        `arti_gap`** — must climb well above the ctx=8 ~5e-4 plateau to confirm arti is now used.
      - **C** — M2 redundancy probe (phoneme-from-`z` vs phoneme-from-`arti-6`). If arti-6
        alone ≈ as good, `z` is redundant → **pivot to Energy-3 arti-space planning** (already
        implemented in `acjepa_energy`/`acjepa_plan`); "P ignores arti" becomes evidence for
        the right design, planning value comes from sequences/coarticulation/constraints.
      - **B** — FiLM arti into every token (adapter change). **D** — predict skip-K/residual.
- [ ] **M2** — phoneme head `C` (or prototypes `μ_p`) **[blocked on the utterance↔session
      alignment above]**. Decide via a probe: does `C` predict phonemes better from `z`
      than from `arti-6` directly? (plan §7 — "do you even need `z`?").
- [ ] **M3** — `aucjepa_energy` + 1-step CEM sanity on **Energy 3** (closed-form arti
      target) — the smoke test already exercises the CEM mechanics; M3 runs it against a
      real trained `P` and a real `arti_targets.npy` table.
- [ ] **M4** — full CEM with **Energy 1**; single-phoneme reachability eval (plan §5.1).
- [ ] **M5** — sequence goals (sub-goals / CTC); PER + arti-trajectory realism; ablations
      (Energy 1/2/3, λ·Energy-3 regulariser, `ctx_frames`, CEM budget).

---

## Files (plan §4)

| File | Status | What |
|---|---|---|
| `artijepa/arti_cache.py` | ✅ new | `.mat` → arti-6 (+IMAGE) `.npy` + `meta.json`; session-disjoint manifest |
| `artijepa/arti_cond.py` | ✅ new | `ArtiConditionedPredictor` (AC predictor verbatim, A=6), pooling, state/action, `rollout`, `rollout_l1` |
| `artijepa/acjepa_dataset.py` | ✅ new | `RTMRIArtiDataset` — IMAGE+arti from one `.mat`, frame-exact |
| `artijepa/acjepa_train.py` | ✅ new | single-GPU trainer (frozen encoder + arti predictor) |
| `artijepa/acjepa_train_ddp.py` | ✅ new | DDP trainer (`static_graph=True`); ckpt interchangeable w/ single-GPU |
| `artijepa/acjepa_energy.py` | ✅ new | `energy_classifier_nll` / `energy_prototype` / `energy_arti_target` + builders |
| `artijepa/acjepa_plan.py` | ✅ new | CEM / receding-horizon MPC; `make_energy`, `load_world_model` |
| `configs/acjepa_arti6_128.yaml` / `_256.yaml` / `_256_ddp.yaml` | ✅ new | forward-model configs (`arti.dim:6`, `target_fps:100`) |
| `configs/acjepa_plan_256.yaml` | ✅ new | planner hyperparams (energy kind, ctx/horizon, CEM `M`/`top_k`/`n_iter`/`sigma0`/`a_clip`) |
| `scripts/11_build_arti_cache.sh` … `14_plan_acjepa.sh` | ✅ new | cache / train / ddp / plan launchers |
| `tests/test_acjepa_smoke.py` | ✅ new | CPU smoke (incl. energies + CEM) |
| phoneme head `C` | ⬜ reuse | train via existing `eval_phoneme_*.yaml`; **needs M2 alignment** |

Reused unchanged: `src/models/ac_predictor.py` (AC predictor), `model.build_models`,
`checkpoint.*`, `rtmri_dataset` preproc helpers (`_intensity_norm`/`_spatial`),
`phonemes.*` (inventory + token alignment + PER).

## Design decisions locked
- Trainable = AC predictor + its 6→D state/action Linear projections only; encoder **frozen**.
- Target = the frozen encoder itself (no EMA, no collapse risk).
- Conditioning = `state = arti[t]`, `action = arti[t+1] − arti[t]` (Euclidean ⇒ plain
  diff; no SO(3)), distinct `Linear(6, D)`, AC predictor verbatim (`add_tokens=2`).
- Objective = **temporal** world-model, teacher-forced + context-prefix AR rollout
  (`ctx_frames`, makes the articulators causally necessary).
- Frozen index order = `ARTICULATORS = [Bilabial, Alveolar, Palatal, Velum, Pharyngeal,
  Larynx]` (`arti_cond.py`) — the cache, predictor, energy targets, planner all agree.
- Phoneme inventory = the repo's 41-symbol ARPABET (`phonemes.py`).
- Encoder init = T-SSL `tssl_vitl_256/latest.pt` `target_encoder` (RoPE ⇒ res-flexible,
  loads clean at 128/256 even though usc_lss frames are 104×104).

## How to run
```bash
source dev_artiJEPA/scripts/_env.sh

# M0: cache arti-6 (+IMAGE) from usc_lss .mat sessions + manifest (scipy env, no GPU)
bash dev_artiJEPA/scripts/11_build_arti_cache.sh          # add --limit 5 for a subset

# M1: train forward model P. 128px smoke (P100) or 256px primary / DDP:
bash dev_artiJEPA/scripts/12_train_acjepa.sh dev_artiJEPA/configs/acjepa_arti6_128.yaml
#   quick:  ... acjepa_arti6_128.yaml --max-steps 50
#   resume: ... --resume /scratch1/hongn/artijepa/runs/acjepa_arti6_128/latest.pt
#   DDP:    bash dev_artiJEPA/scripts/13_train_acjepa_ddp.sh dev_artiJEPA/configs/acjepa_arti6_256_ddp.yaml

# M3+: plan toward a goal phoneme (needs a trained P + arti_targets.npy for Energy 3)
bash dev_artiJEPA/scripts/14_plan_acjepa.sh \
    /scratch1/hongn/artijepa/runs/acjepa_arti6_256/latest.pt m --seed-clip 0

# CPU smoke (no GPU/transformers): all checks
PYTHONPATH=.:dev_artiJEPA python dev_artiJEPA/tests/test_acjepa_smoke.py
```
Outputs per run under `meta.folder`: `train_log.csv`, `diagnostics.jsonl` (TF/AR L1 +
`arti_gap`), `latest.pt` (predictor/opt/scaler/epoch — resumable; encoder reloaded
frozen from `model.checkpoint`).

## Open questions / [VERIFY]
- **Utterance↔session phoneme alignment** (blocks M2/M4/M5): map per-utterance gold
  phoneme timestamps onto session frame time. Options: (a) find each utterance's start
  offset within its `_mview` session; (b) re-segment sessions into utterances and cache
  per-utterance; (c) force-align phonemes to the session audio directly. Decide before C.
- **Do we even need `z`?** (plan §7) — if the 6 articulators fully describe the state,
  Energy 3 plans in 6-D with trivial dynamics. Add the video model only if `z` carries
  phonetic info the 6 features don't. Resolve with the M2 probe.
- **Single-phoneme planning may be trivial** — research value is in sequences /
  coarticulation / velocity-bounded reachability (plan §5 reality check). Ensure the eval
  includes ≥1 of these.
- **CEM exploiting the latent energy** — adversarial arti-seqs that fool `C`. Mitigate via
  `lambda_arti` (Energy-3 regulariser) + `a_clip` velocity bound (both already in the
  planner config).
- **`grayscale_stats`** for usc_lss frames not yet fit — relying on per-clip z-score
  (mean 0 / std 1 defaults), which dominated for the 75-speaker corpus too.
```
