# Arti-JEPA — TODO: Evaluation

Evaluate frozen encoder features (our T-SSL checkpoints + public image models) on
**phoneme prediction** from the silent rtMRI video. Metrics: frame-level
**Cohen's κ** + **PER**. Completed numbers live in `RESULTS.md`; pretraining plans
in `TODO_pretraining.md`; file map in `Master.md`.

**Eval task pivot (2026-06-07):** the weak stimulus-group classification was
**removed** (clip-type says nothing about articulation). Two label sources:
- **Task 1 — pseudo** phonemes from an audio model (wav2vec2/WavLM CTC) on the
  75-speaker corpus's paired audio (in-domain speakers).
- **Task 2 — gold** phonemes w/ timestamps for one **OOD** speaker
  (`/scratch1/hongn/usc_lss`, 104×104 @ 99 fps, 684 utts, 41 ARPABET).

**Env caveat (audio model):** `transformers 5.10.2` needs torch≥2.7 but this env
has 2.6 → CTC import fails. Task-1 label-gen (`build_pseudo_labels`) is a
**decoupled** batch step: run in a compatible env (`pip install 'transformers<5'`
or torch≥2.7); it writes `.npy` label streams the artijepa eval reads.

**Alignment** is in **seconds** (frame-rate agnostic): clips resampled to
target_fps (25), audio phoneme stream ~50 Hz, 4 audio units ≈ 1 JEPA token
(tubelet 2 → 80 ms). The 99-fps OOD speaker needs no special-casing.

**Eval infra (DONE):** `eval_phoneme.py` (freeze encoder → per-temporal-token
features → per-token probe → κ + PER), `phonemes.py` (41-ARPABET, seconds-based
alignment, CTC-collapse, PER, κ), `usc_lss.py` (OOD manifest+dataset),
`audio_phoneme.py` (Task-1 pseudo pipeline), `baselines.py` (public image models).
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

### 3. Task-1 pseudo labels — **NOT STARTED (TBA)**
The in-domain transfer probe (no gold alignment → where CTC should pay off).
- Run `audio_phoneme.build_pseudo_labels(manifest_split.csv)` in a
  transformers-compatible env (`transformers<5` or torch≥2.7) → cache to
  `/scratch1/hongn/artijepa/pseudo_phonemes/`.
- Then eval with `configs/eval_phoneme_pseudo.yaml` (kind: pseudo) on pretrained
  + tssl_128/256. **Result rows in `RESULTS.md` = TBA.**
- Re-test CTC here (the head×loss ablation found CTC worse *only because* gold
  alignment was available; pseudo is the no-alignment case CTC is built for).

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
| dinov2-L/14 (DINOv3 unavailable) | 518 | ✅ κ 0.291 | ❌ NOT STARTED |
| resnet-50 | 224 | ✅ κ 0.304 | ❌ NOT STARTED |

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
