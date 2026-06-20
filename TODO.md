# Arti-JEPA — Progress & TODO

V-JEPA 2 → rtMRI vocal-tract video. Plan: `Arti-JEPA-Plans.md`. Headline track =
**T-SSL** (domain-adaptive SSL on unlabeled rtMRI) + a **phoneme-prediction** eval
of the lift. Only edit `dev_artiJEPA/`; parent repo is the read-only reference.

**Eval task pivot (2026-06-07):** the weak stimulus-group classification was
**removed** (not meaningful — clip-type says nothing about articulation). The
downstream task is now **phoneme prediction** from the silent rtMRI video, scored
by frame-level **Cohen's κ** + **PER**. Two label sources (Plan B.4 / Part C):
- **Task 1 — pseudo** phonemes from an audio model (wav2vec2/WavLM CTC) on the
  75-speaker corpus's paired audio (in-domain speakers).
- **Task 2 — gold** phonemes w/ timestamps for one **OOD** speaker
  (`/scratch1/hongn/usc_lss`, 104×104 @ 99 fps, 684 utts, 41 ARPABET).

**Env / node:** conda `artijepa` (torch 2.6+cu124). `source dev_artiJEPA/scripts/_env.sh`.
Current node = **L40S (46 GB, Ada cc 8.9, bf16)**; was V100 — check `nvidia-smi`.
Eval uses bf16; T-SSL configs use float16 (V100-safe, also fine on L40S).
**Audio-model caveat:** `transformers 5.10.2` needs torch≥2.7, but this env has
2.6 → the CTC import fails. Task-1 label-gen (`build_pseudo_labels`) is a
**decoupled** batch step: run it in a compatible env (`pip install 'transformers<5'`
or torch≥2.7); it writes `.npy` label streams that the artijepa eval reads.

Artifacts under `/scratch1/hongn/artijepa/` (never `/project2`):
`manifest_split.csv` (train 1808 / val 279 / test 284, subject-disjoint),
`grayscale_stats.json`, `checkpoints/vitl.pt`, `runs/<name>/`,
`feat_cache/phoneme/<tag>/`, `eval/`. usc_lss manifest:
`/scratch1/hongn/usc_lss/phoneme_manifest.csv`.

---

## DONE
- [x] **Phase 0 — data engineering (A.1–A.9).** manifest, subject-disjoint splits,
      grayscale stats, decord linear-interp resample (`crop`/`tile`), safe aug.
- [x] **T-SSL trainer (B.3).** `tssl_train.py` (EMA target, L1 feat loss,
      multiblock masks per grid). Loads pretrained ViT-L clean (292/292). Now logs
      **label-free** collapse metrics only (stimulus probe removed). L40S step
      ~0.65 s @128px bs16 → ~1.5 h/8k steps, ~4.5 h full.
- [x] **Removed stimulus-group eval** from code (`eval_probe.py`/config/script
      deleted; `collapse.linear_probe` dropped; `tssl_train` diagnostics → label-free).
- [x] **Phoneme infra — NEW.**
      - `phonemes.py` — 41-ARPABET inventory, **seconds-based** alignment to JEPA
        tokens (frame-rate agnostic), CTC-collapse, PER (edit distance), Cohen's κ.
        Validated on usc_lss (self κ=1.0; inventory covers data).
      - `usc_lss.py` — OOD manifest builder + dataset (104×104→resize, 99→25 fps,
        per-token gold labels, tile+pad, PER reassembly). 684 utts (581/34/69).
      - `eval_phoneme.py` — freeze encoder → per-temporal-token features
        (`[B,N,D]→[B,T',D]`, temporal-major pooling) → per-token probe
        (`linear`/`mlp`/`tcn`) → κ + PER. Vocab/drop-set sourced from dataset.
        Config `eval_phoneme_usc_lss.yaml`, script `04_eval_phoneme.sh`.
      - `audio_phoneme.py` — Task-1 pseudo pipeline (ffmpeg 16 kHz audio →
        wav2vec2 CTC → `.npy` streams; `PseudoPhonemeDataset`). Audio decode +
        dataset verified; **model step blocked by env (see caveat) → decoupled**.
- [x] **Task-2 baseline (gold, OOD, pretrained, 128px, tcn probe):**
      **test κ 0.222 / PER_micro 0.760** (val κ 0.240 / PER 0.745), frame-acc
      0.259 vs 1/41 chance. Real phonetic signal out-of-the-box. JSON:
      `…/eval/phoneme_usc_lss_pretrained128_*.json`; features cached at
      `…/feat_cache/phoneme/pretrained128_*/`.

## ✅ 256px T-SSL — COMPLETE (ep50) + FINAL eval DONE (2026-06-16)
**Training finished cleanly to epoch 50 on V100-32GB (d13-07, job 9402610): `[tssl] epoch 50
avg loss 0.4548 … done`, diagnostics clean (feature_std 1.25, eff_rank 75.8, mean_abs_cosine
0.505 — no collapse). `runs/tssl_vitl_256/latest.pt` = epoch-50 checkpoint.**
**FINAL eval DONE (2026-06-16 09:17, gold/OOD usc_lss, 256px, CE, tag `tssl256` = fresh e50
features): HEADLINE = tssl_256 @e50 + attentive test κ 0.530 / PERµ 0.486 / frame-acc 0.549.**
T-SSL lift over frozen pretrained-256 at the same res: tcn 0.303→0.382, tcn_spatial 0.407→0.488,
attentive **0.446→0.530 (+0.084, +19%)**. Finishing training past e22 added +0.034 κ (attentive
0.496→0.530). **κ 0.530 is the best result anywhere** — beats tssl_128 (0.475), the e22 snapshot
(0.496), and the best image baseline (sup ViT-L/16, 0.368) by a wide margin. Driver
`eval/run_256_e50_eval.sh`, log `eval/eval_256_e50.log`, JSONs `eval/phoneme_usc_lss_tssl256*_*.json`.
Encoder loaded clean (`292 miss 0 ... (epoch 50)`). Caches: `feat_cache/phoneme/tssl256_c2c3132725`
(pooled, tcn) + `tssl256sp_9ee0a341a1` (spatial grid, tcn_spatial+attentive).
**Next candidates:** ≥3 seeds for the spatial heads; re-run image baselines w/ a spatial probe for
a fully fair fight (`baselines.py` tubelet-pools spatial away today); Task-1 pseudo labels.

<details><summary>(historical) resume log while training was in progress</summary>

**Status (2026-06-14 21:59 PDT): RE-RESUMED @ epoch 32/50, RUNNING on V100-32GB (d13-07),
interactive SLURM job 9402610 (~47 h alloc, TIME_LEFT ~1-22h at launch).** History: prior
allocs kept dying on SSH drop; the run before this reached e33 step ~300 (~10:16 Jun-14) then
died; `latest.pt` = end of epoch 32 (09:14, epoch-end saves only), so this resume redoes e33
from step 0 (lost ~300 micro-steps, ~1 h, no harm). Resume verified THIS launch:
`[tssl] RESUMED latest.pt @ epoch 32 (8000 opt-updates fast-forwarded); continuing to epoch 50`,
first steps `[e33 0/500] loss=0.4529 … m=0.9980` → `[e33 20/500] loss=0.4509 … m=0.9993`
— continuous, no re-warmup (`m=0.9980` at step 0 is the display init; corrects to 0.9993).
GPU: **SOLO this time** (no shared proc) → training ~24.1 GB / 100 %, step time **~12 s/micro-step**.
**18 epochs left (32→50) × ~1.7 h ≈ ~31 h < the ~47 h alloc → should finish to epoch 50 in THIS
single allocation, no further resume needed if SSH holds.** (Released a spare idle alloc job
9402630/d14-11 at launch — was holding 2 GPUs.) Launched DETACHED with `setsid`
(+ `PYTHONUNBUFFERED=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), stdout appended to
`train_stdout.log` — survives SSH disconnection within the alloc.
Re-launch (verbatim, if it dies again): from `$REPO_ROOT` (= `/project2/shrikann_35/hongn/vjepa2`),
`setsid env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_256.yaml --resume /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt < /dev/null >> /scratch1/hongn/artijepa/runs/tssl_vitl_256/train_stdout.log 2>&1 &`
Diagnostics clean through epoch 32 (feature_std 0.175→~1.27–1.31, mean_abs_cosine 0.99→~0.45↓,
eff_rank dipped to ~28 @ep5 then recovering to ~70.8 @ep32 — anti-collapse, no sign of collapse).

**Epoch-22 eval — DONE (2026-06-13 15:43).** 256px phoneme eval of the halfway checkpoint
ran against a FROZEN snapshot (now deleted, reclaimed 5.1 GB); all 6 combos complete, results
in the log table + the 2026-06-13 UPDATE note below. Headline: **tssl_256 @e22 + attentive
κ 0.496 > fully-trained tssl_128 (0.475) > best image baseline (0.368)**; T-SSL lift ~+0.05 κ
across all heads. Driver `eval/run_256_e22_eval.sh`, log `eval/eval_256_e22_rerun.log`.
**CACHE-KEY CAVEAT (still applies to the epoch-50 eval):** `eval_phoneme._tag()` omits the
checkpoint path from the cache hash, so the FINAL (epoch-50) tssl eval MUST use tag `tssl256`
(NOT `tssl256e22`) or first `rm feat_cache/phoneme/tssl256e22*`, else it silently re-reads the
epoch-22 features cached here.
**Status (2026-06-10): RUNNING on Tesla V100-32GB at batch 32, training cleanly.**
`tssl_vitl_256.yaml` is now **batch_size 32 / ipe 500 / accum 2** (was bs8/ipe2000/accum8);
`eff_batch=64`, `oue=250`, clips/epoch=16k are **all unchanged**, so the recipe + resume +
schedules are byte-identical — only the micro-batch changed. ~**12.5 s/micro-step → 0.39 s/clip**
(vs bs8's 0.44, ~11 % faster) → full run `ipe500×50ep=25k micro-steps` ≈ **~87 h**. Peak VRAM
**~24.5/32 GB** (run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Loss 0.51→0.37.

**VRAM tuning (why bs32, not bigger):** at 256px on 32 GB, **activation checkpointing is
MANDATORY** — `use_activation_checkpointing: false` OOMs even at bs8 (target encoder sees all
4096 tokens → ~30 GB activations). So the recompute can't be removed; we stay **compute-bound**
(6 vs 2 dataloader workers = same speed). The only lever is a bigger batch under ckpt: bs8≈12 GB,
**bs32≈24.5 GB (~11 % faster/clip, the sweet spot)**, bs64 OOMs (hits 32 GB). Micro-batch must
divide eff_batch 64 → {8,16,32,64}, so 32 is the largest that fits. P100 was ~28 s/step (infeasible);
L40S ~40 h (flip `dtype: bfloat16`).

**The SLURM allocation is 47 h → ~23 epochs fit per allocation; RESUME continues it:**
```bash
source dev_artiJEPA/scripts/_env.sh
bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_256.yaml \
     --resume /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt
```
`latest.pt` is saved atomically every epoch (with `.prev` backup) + schedules
fast-forward, so resume picks up exactly where it stopped. Watch
`runs/tssl_vitl_256/{train_log.csv,diagnostics.jsonl,train_stdout.log}` for collapse
(clean at 128px). **Eval intermediate checkpoints** to get the 256px datapoint before
the full run completes (the spatial probe below makes any `latest.pt` worth checking).
On an L40S/A100/H100 this budget is ~40 h and you can flip `meta.dtype: bfloat16`.

**What's already done (validated):**
- `tssl_train.py` now has **gradient accumulation** (`optimization.effective_batch`;
  schedules step per *optimizer update*) + **resume** (`--resume <ckpt>`, atomic save,
  `.prev` backup, scaler+schedule restore). Both smoke-tested.
- `tssl_vitl_256.yaml` set to **batch 8 / effective_batch 64 (accum 8) / ipe 2000**
  (16k chunks/epoch ≥ the 128px 8k parity). GPU peak only ~5 GB/16 (act-ckpt); host
  RAM (16 GB cap) is the real limit → `num_workers 2`, `pin_mem false`.
- `runs/tssl_vitl_256/` is **clean** (no stale ckpt) → first launch starts fresh.

**Do, on a fast GPU node:**
1. `source dev_artiJEPA/scripts/_env.sh`; `nvidia-smi` (confirm it's NOT a P100/V100; want L40S/A100/H100).
2. Full run (fresh): `bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_256.yaml`
   — watch `runs/tssl_vitl_256/diagnostics.jsonl` for collapse (clean at 128px).
   `latest.pt` saved every epoch (atomic). On L40S+ flip `meta.dtype: bfloat16`.
3. **If the allocation dies, resume:** same command + `--resume /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt`.
4. Phoneme eval, baseline + adapted, at 256px (NOTE: need `configs/eval_phoneme_usc_lss_256.yaml`
   — may not exist yet; copy `eval_phoneme_usc_lss.yaml` and set `spatial_size: 256`, `dtype` per GPU):
   `bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml --tag pretrained256`
   `bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt --tag tssl256`
   → fill the tssl_256 / pretrained_256 rows. **256px is the fair comparison vs the image
   baselines** (which ran at native 224–518; supervised ViT-L led at κ 0.368 vs V-JEPA@128 0.222).

</details>

## TODO (after 256px)
- [x] **Pretrained vision-encoder baselines (2026-06-09).** Added `artijepa/baselines.py`
      (timm, per-frame → tubelet-pool to V-JEPA's token grid → same probe+labels) +
      `image_baseline` path in `eval_phoneme.py`, config `eval_phoneme_usc_lss_baseline.yaml`,
      `scripts/05_eval_baselines.sh`. Ran all five at gold/OOD, tcn, native res, "each its
      best shot" (minmax→[0,1] + model's own mean/std). **DINOv3 ViT-L unavailable in this
      env's timm (only ViT-7B/ConvNeXt) → used DINOv2 ViT-L/14.** transformers 5.x is broken
      here (vision class imports fail) → timm-only. **Result: at native res ALL beat frozen
      V-JEPA@128px; supervised ViT-L/16 best (κ 0.368 vs 0.222/0.247).** BUT res confound
      (224–518 vs 128) → the 256px V-JEPA run is the fair comparison. Rows in results log.
- [ ] **Task-1 pseudo labels.** `audio_phoneme.build_pseudo_labels(manifest_split.csv)`
      in a transformers-compatible env (this env's transformers 5.x needs torch≥2.7;
      use `transformers<5` or a torch≥2.7 env) → then eval with `eval_phoneme_pseudo.yaml`.

## Probe-head & loss ablation (NEW — requested)
Question: does a stronger temporal read-out + a sequence loss recover
coarticulation-scale dependency the `tcn` (±2-token conv) misses? Two axes
(Plan B.4 / Part D). Run on the **already-cached** `[N,16,D]` features of
`pretrained128` + `tssl128` (and baselines), so it is cheap.

**Axis 1 — probe head** (add to `eval_phoneme.TokenProbe`, CE-trained, drop-in):
- `lstm` — bi-LSTM over the 16 tokens (full-sequence recurrence).
- `transformer` — small self-attention encoder (global context) + learned/sinusoid
  positional encoding over the 16 tokens.
- Sweep `linear`/`mlp`/`tcn`/`lstm`/`transformer` × {pretrained128, tssl128}, ≥3 seeds → κ + PER.
- **Low risk; do first.** No new data path; metric stays κ-native.

**Axis 2 — CTC loss** (alignment-free; bigger lift):
- New per-utterance path: feed whole utterances (variable `T'`, pad + `input_lengths`),
  vocab += blank (→ 42 classes), target = collapsed phoneme sequence (`target_lengths`),
  `nn.CTCLoss`. Reassemble per-utt from the cached `meta=(utt,chunk,n_chunks)` rather
  than re-extracting features.
- Decode greedy (and optional beam) → **PER (primary)**; κ only via CTC forced
  alignment (optional). Pairs with `lstm`/`transformer`.
- **Caveats (documented in Plan B.4):** CTC still emits ≤1 label/token → does NOT beat
  the 80 ms token-rate ceiling; metric emphasis shifts CE→κ, CTC→PER.
- Config: `probe.loss: {ce|ctc}`, `probe.type: {…|lstm|transformer}`. CLI `--loss`.

Tasks:
- [x] Added `lstm` + `transformer` heads to `TokenProbe` (CE) and a CTC mode
      (`--loss ctc`: per-utt collate from cache, blank vocab, `nn.CTCLoss`, greedy
      decode → PER). `scripts/06_probe_sweep.sh` runs the head×loss×encoder grid on
      cached features. **(2026-06-09)**
- [x] **Spatial-aware heads `tcn_spatial` + `attentive` (2026-06-10).** Un-pooled
      `[B,T',S',D]` extraction (`pool_spatial` derived from probe type, `…sp_<hash>`
      cache), V-JEPA `AttentivePooler` reused. `scripts/07_probe_spatial.sh`. **Keystone
      result above — spatial probe ~doubles the T-SSL κ lift and beats the baselines.**
- [ ] ≥3 seeds for the winning combos (single-seed); re-run image baselines w/ a spatial
      probe (needs `baselines.py` to emit the spatial grid); optional beam decode for CTC.

**Sweep result (2026-06-09, gold/OOD usc_lss, 128px, single seed) — head × loss:**

| encoder | head | CE test κ | CE PERµ | CTC PERµ |
|---|---|---|---|---|
| pretrained128 | **tcn** | **0.224** | **0.759** | 0.811 |
| pretrained128 | lstm | 0.196 | 0.778 | 0.820 |
| pretrained128 | transformer | 0.186 | 0.804 | 0.841 |
| tssl128 | **tcn** | **0.255** | **0.739** | 0.792 |
| tssl128 | lstm | 0.249 | 0.738 | 0.811 |
| tssl128 | transformer | 0.247 | 0.752 | 0.824 |

**Findings:** (1) **`tcn`+CE wins** — `lstm`/`transformer` *overfit* the small data on
frozen 128px features (κ drops with head capacity). (2) **CTC is worse on PER
everywhere** (+0.05–0.08): we have gold alignment, so per-token CE uses more signal;
CTC's payoff is the **Task-1 pseudo** labels (no alignment), not here. (3) **T-SSL still
helps across every head/loss** — the lift is in the features, not the read-out. So the
`tcn`+CE / κ headline stands; the temporal-head/CTC ablation is a negative-but-informative
result. Re-test CTC once Task-1 pseudo labels exist.

### ▶▶ Spatial-aware probe — KEYSTONE (2026-06-10): mean-pooling was hiding most of the signal
The default eval **mean-pools the S'=(res/patch)² spatial tokens** away before the probe
(`[B,N,D]→[B,T',D]`). But *where* in the vocal tract the signal sits (tongue/lip/velum
position) is exactly the phonetic information. Two **spatial-aware** heads consume the
**un-pooled `[B,T',S',D]` grid** instead (`eval_phoneme.py`, CE only):
- **`tcn_spatial`** — learned additive attention-pool over S' per temporal step → `[B,T',D]`,
  then the same kernel-3 TCN over time (the *only* change vs `tcn` is mean→learned spatial pool).
- **`attentive`** — V-JEPA's exact `AttentivePooler` (cross-attn, 1 query) over S' per
  temporal step → `[B,T',D]` → linear (no temporal mixing). `src/models/attentive_pooler.py`.
Extraction caches the un-pooled grid under a `…sp_<hash>` tag (4.1 GB/encoder, 128px;
existing pooled caches untouched). Run: `bash scripts/07_probe_spatial.sh` (default 128px,
both heads, pretrained + T-SSL). `--probe {tcn_spatial,attentive}` auto-selects the un-pooled cache.

**Result (gold/OOD usc_lss, 128px, CE, single seed):**

| encoder | head | test κ | frame-acc | test PERµ | vs mean-pool κ |
|---|---|---|---|---|---|
| pretrained128 | tcn (mean-pool S') | 0.222 | 0.259 | 0.760 | — |
| pretrained128 | **tcn_spatial** | **0.344** | 0.372 | 0.653 | **+0.122** |
| pretrained128 | attentive | 0.327 | 0.355 | 0.694 | +0.105 |
| tssl128 | tcn (mean-pool S') | 0.247 | 0.280 | 0.755 | — |
| tssl128 | tcn_spatial | 0.433 | 0.458 | 0.587 | +0.186 |
| tssl128 | **attentive** | **0.475** | **0.497** | **0.523** | **+0.228** |

**Findings (this changes the headline):**
1. **Spatial structure carries most of the phonetic signal.** Just *not* mean-pooling lifts
   the FROZEN pretrained V-JEPA from κ 0.222→0.344 (+55 %) and T-SSL from 0.247→0.475 (+92 %).
2. **The T-SSL lift is far larger than mean-pooling revealed.** Mean-pool lift was +0.025
   (0.222→0.247); with the attentive spatial probe it is **+0.148** (0.327→0.475). T-SSL
   adds a lot of *spatially-localized* phonetic structure the pooled probe was blind to.
3. **T-SSL + spatial probe (κ 0.475) beats the best image baseline** (supervised ViT-L/16,
   κ 0.368) — and that is still at 128px, the "unfair low-res" V-JEPA condition. So the earlier
   "baselines > frozen V-JEPA" was a **probe artifact (mean-pool), not a resolution gap.**
4. **`attentive` > `tcn_spatial` on the T-SSL features** (0.475 vs 0.433) but ≈/below on
   pretrained (0.327 vs 0.344): the richer cross-attn pooler pays off once the features are
   adapted. Caveat: single seed, small OOD speaker — needs ≥3 seeds (below).

**Follow-ups:** (a) ≥3 seeds for the spatial heads. (b) **Re-run the image baselines with a
spatial probe** for a fair fight (`baselines.py` currently tubelet-pools spatial away → it must
emit the `[B,T',S',D]` grid first). (c) Spatial probe at **256px** once a `tssl_256` checkpoint
lands (S'=256 there). (d) The κ headline (with-vs-without T-SSL) should now be quoted with the
**spatial** probe, not mean-pool `tcn`.

### ▶ IN PROGRESS (2026-06-16, requested) — robustness + fully-fair baseline fight
Two tasks on the 256px spatial probe (now that tssl_256 @e50 is done, κ 0.530):
1. **≥3 seeds for the spatial heads.** Put error bars on the headline. The frozen-encoder
   feature cache is **seed-independent** (deterministic `@torch.no_grad` extraction), so each seed
   only re-trains the cheap probe → reuse `pretrained256sp` / `tssl256sp` caches. Added
   `eval_phoneme.py --seed` (overrides `meta.seed`; probe init/shuffle only) and put the seed in
   the result-JSON name (`…_s{seed}.json`) so seeds don't clobber. Combos: {pretrained256, tssl256}
   × {tcn_spatial, attentive} × seeds {0,1,2}. Driver `eval/run_seeds_spatial.sh`.
2. **Image baselines WITH the spatial probe** (the fair fight). Taught `baselines.py` to emit the
   per-frame **patch-token grid** when `pool_spatial=False`: ViT → `forward_features` minus
   CLS/reg tokens; ResNet → the conv feature map; tubelet-pooled over time, flattened
   temporal-major to `[B,T'*S',D]` so `extract()` recovers `[B,T',S',D]`. Grid side capped at 16
   (adaptive-pool) so **dinov2@518 37×37→16×16** stays tractable (clip/siglip 16×16, vit-L/16
   14×14, resnet 7×7 are ≤cap, untouched). Each model at its native res ("its best shot"). Combos:
   {clip, siglip, dinov2, vitl, resnet} × {tcn_spatial, attentive} × seeds {0,1,2}. Driver
   `eval/run_baselines_spatial.sh`. **The decisive check:** does sup ViT-L/16 + spatial probe stay
   below tssl_256 + spatial probe? If so, the keystone "T-SSL > baselines" holds under a fair probe.

#### PARTIAL RESULTS — ⚠ RUN INTERRUPTED by GPU de-allocation (state verified 2026-06-17)
The fair-fight baseline run (`run_baselines_spatial.sh`) **died mid-training of `base_siglip |
attentive | seed 1`** (log only reached `[probe e1/40]`) when the GPU was lost. Last clean JSON
written = `base_siglip | attentive | seed 0` at 20:42. So nothing meaningful landed past the
~20:16 snapshot except siglip-attentive s0. **No GPU is currently dedicated to artijepa**
(d13-03 / job 9451141 is busy with the HIS `train_real.py seamless_v100` job, ~19.5 GB).
All JSONs persist on /scratch1 (`eval/phoneme_usc_lss_*_s{0,1,2}.json`); log `eval/eval_256_fairfight.log`.

**Per-model completion (verified by re-aggregating the JSONs, 2026-06-17):**
- ✅ `pretrained256` tcn_spatial + attentive — **3/3 seeds**
- ✅ `tssl256 @e50`  tcn_spatial + attentive — **3/3 seeds**  ← headline, DONE & solid
- ✅ `base_clip`     tcn_spatial + attentive — **3/3 seeds**
- ✅ `base_siglip`   tcn_spatial — **3/3 seeds**;  attentive — **1/3** (s0 only; s1 died, s2 not started)
- ❌ `base_vitl` (supervised ViT-L/16, **the key competitor**) — **0/3, NOT STARTED**
- ❌ `base_dinov2` — 0/3;  ❌ `base_resnet` — 0/3

**3-seed mean±sd, gold/OOD usc_lss, 256px, CE (test κ / test PERµ / val κ):**

| encoder | head | seeds | test κ (mean±sd) | test PERµ | val κ |
|---|---|---|---|---|---|
| pretrained256 | tcn_spatial | 0,1,2 | 0.411 ± 0.007 | 0.601 | 0.425 |
| pretrained256 | **attentive** | 0,1,2 | **0.449 ± 0.005** | 0.556 | 0.458 |
| **tssl256 @e50** | tcn_spatial | 0,1,2 | 0.481 ± 0.005 | 0.528 | 0.497 |
| **tssl256 @e50** | **attentive** | 0,1,2 | **0.527 ± 0.004** | 0.485 | 0.544 |
| base: clip-L/14 | tcn_spatial | 0,1,2 | 0.345 ± 0.010 | 0.662 | 0.375 |
| base: clip-L/14 | attentive | 0,1,2 | 0.282 ± 0.002 | 0.676 | 0.313 |
| base: siglip-L/16 | tcn_spatial | 0,1,2 | 0.363 ± 0.009 | 0.636 | 0.405 |
| base: siglip-L/16 | attentive | 0 *(1/3, partial)* | 0.296 | 0.671 | 0.320 |

**Findings so far (1) seeds tighten the headline:** tssl256+attentive **0.527±0.004** vs pretrained256+attentive
0.449±0.005 → T-SSL lift **+0.078**, sd ~0.004–0.005 (tiny → highly significant; single-seed 0.530 was representative).
**(2) Keystone holds emphatically under the fair probe so far:** even the best image baseline with the spatial
probe (siglip tcn_spatial 0.363, clip tcn_spatial 0.345) is **far below** frozen pretrained V-JEPA (0.449), let
alone tssl256 (0.527). Note `attentive` *hurts* clip (0.282 < its tcn_spatial 0.345) — the cross-attn pooler
needs V-JEPA-style features. **STILL PENDING (the decisive check): supervised ViT-L/16** (mean-pool
winner at 0.368, never ran here), plus dinov2, resnet, and siglip-attentive s1/s2. The keystone is
not airtight until vitl runs — it was the one competitive baseline under the old mean-pool probe.

**▶ RESUME (needs a free GPU — none allocated to artijepa as of 2026-06-17):** `source dev_artiJEPA/scripts/_env.sh` then
`setsid env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash /scratch1/hongn/artijepa/eval/run_baselines_spatial.sh < /dev/null >> /scratch1/hongn/artijepa/eval/eval_256_fairfight.log 2>&1 &`
— clip+siglip feature caches (`base_clipsp`,`base_siglipsp`, 17 GB ea) already exist → those extractions cache-hit
and finished probes overwrite idempotently. **To skip the redo and finish only what's missing, edit the script's
`for m in …` list to `vitl dinov2 resnet siglip`** (siglip will cache-hit its grid and just (re)train the
attentive seeds it's missing). ⚠ Do NOT co-launch on d13-03 (job 9451141) while the HIS job runs there — the
16 GB SLURM host-RAM cap will likely OOM both. Then aggregate all `_s{0,1,2}.json`
(`eval/aggregate_spatial.py`) and fill the table above.

## DONE (T-SSL phase)
- [x] **128px T-SSL trained to completion (50 ep, 2026-06-07).** Clean, **no
      collapse** — all three metrics moved *anti*-collapse over training:
      feature_std 0.30→1.12, effective_rank 79→**102**/1024, mean_abs_cosine
      0.957→**0.560**. Loss settled ~0.40 (L1 vs moving EMA target). Ckpt
      `runs/tssl_vitl_128/latest.pt` (5.1 GB).
- [x] **Headline: with vs without T-SSL (gold OOD phoneme, 128px, tcn).** T-SSL
      **lifts** representation phonetics and it transfers to the unseen speaker:
      test κ **0.222→0.247** (+0.025, +11%), frame-acc 0.259→0.280, PERµ
      0.760→0.755; val κ 0.240→0.277 (+15%). κ shows the gain; PER is capped by
      the 80 ms token rate. JSON `…/eval/phoneme_usc_lss_tssl128_*.json`.
- [ ] **Task-1 pseudo labels.** Run `audio_phoneme.build_pseudo_labels(manifest_split.csv)`
      in a transformers-compatible env → cache to `/scratch1/hongn/artijepa/pseudo_phonemes/`,
      then `eval_phoneme.py` with `eval_phoneme_pseudo.yaml` (kind: pseudo). Needs
      a small `configs/eval_phoneme_pseudo.yaml` (TODO) + the env fix.
- [ ] **Probe ablations:** see **Probe-head & loss ablation** section below
      (`linear`/`mlp`/`tcn`/`lstm`/`transformer` × CE/CTC); 128 vs 256 px; ≥3 seeds
      (cheap once features cached). Optional: 25 fps frame-level head (×2 upsample)
      to beat the 80 ms token resolution ceiling on fast phonemes.
- [ ] **(usc_lss bonus) dense heads later:** it ships tongue contours / SAM seg /
      kinematics — enables segmentation/landmark probes on the OOD speaker (Plan B.4).

## DEFERRED
- T1–T3 fine-tuning ladder; ViT-g scale-up; bf16+grad-accum for 256px T-SSL.

---

## How to run
```bash
source dev_artiJEPA/scripts/_env.sh
# Task 2 (gold OOD) phoneme eval, pretrained baseline:
bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml --tag pretrained128
# T-SSL train (128px), bounded:
bash dev_artiJEPA/scripts/03_train_tssl.sh dev_artiJEPA/configs/tssl_vitl_128.yaml --max-steps 8000
# with-T-SSL phoneme eval:
bash dev_artiJEPA/scripts/04_eval_phoneme.sh dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml \
     --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt --tag tssl128
```

## Results log (phoneme; κ↑ better, PER↓ better)
| date | encoder | task | res | probe | test κ | test PERµ | val κ | frame-acc | chance |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-07 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | tcn | 0.222 | 0.760 | 0.240 | 0.259 | 0.024 |
| 2026-06-07 | **tssl_128** (V-JEPA) | gold/OOD usc_lss | 128 | tcn | **0.247** | 0.755 | 0.277 | 0.280 | 0.024 |
| 2026-06-10 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | tcn_spatial | 0.344 | 0.653 | 0.349 | 0.372 | 0.024 |
| 2026-06-10 | pretrained (V-JEPA) | gold/OOD usc_lss | 128 | attentive | 0.327 | 0.694 | 0.345 | 0.355 | 0.024 |
| 2026-06-10 | tssl_128 (V-JEPA) | gold/OOD usc_lss | 128 | tcn_spatial | 0.433 | 0.587 | 0.450 | 0.458 | 0.024 |
| 2026-06-10 | **tssl_128** (V-JEPA) | gold/OOD usc_lss | 128 | **attentive** | **0.475** | **0.523** | 0.483 | 0.497 | 0.024 |
| 2026-06-09 | base: dinov2-L/14 | gold/OOD usc_lss | 518 | tcn | 0.291 | 0.710 | 0.317 | 0.321 | 0.024 |
| 2026-06-09 | base: clip-L/14 | gold/OOD usc_lss | 224 | tcn | 0.293 | 0.710 | 0.331 | 0.323 | 0.024 |
| 2026-06-09 | base: resnet-50 | gold/OOD usc_lss | 224 | tcn | 0.304 | 0.694 | 0.345 | 0.331 | 0.024 |
| 2026-06-09 | base: siglip-L/16 | gold/OOD usc_lss | 256 | tcn | 0.313 | 0.685 | 0.326 | 0.342 | 0.024 |
| 2026-06-09 | **base: vit-L/16 sup** | gold/OOD usc_lss | 224 | tcn | **0.368** | **0.620** | 0.420 | 0.393 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | tcn | 0.303 | 0.693 | 0.320 | 0.335 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | tcn_spatial | 0.407 | 0.590 | 0.404 | 0.432 | 0.024 |
| 2026-06-13 | pretrained (V-JEPA) | gold/OOD usc_lss | 256 | attentive | 0.446 | 0.564 | 0.453 | 0.469 | 0.024 |
| 2026-06-13 | tssl_256 **@e22** (V-JEPA) | gold/OOD usc_lss | 256 | tcn | 0.356 | 0.643 | 0.397 | 0.383 | 0.024 |
| 2026-06-13 | tssl_256 **@e22** (V-JEPA) | gold/OOD usc_lss | 256 | tcn_spatial | 0.461 | 0.544 | 0.482 | 0.483 | 0.024 |
| 2026-06-13 | **tssl_256 @e22** (V-JEPA) | gold/OOD usc_lss | 256 | **attentive** | **0.496** | **0.508** | 0.509 | 0.517 | 0.024 |
| 2026-06-16 | tssl_256 **@e50** (final) | gold/OOD usc_lss | 256 | tcn | 0.382 | 0.639 | 0.416 | 0.407 | 0.024 |
| 2026-06-16 | tssl_256 **@e50** (final) | gold/OOD usc_lss | 256 | tcn_spatial | 0.488 | 0.527 | 0.508 | 0.508 | 0.024 |
| 2026-06-16 | **tssl_256 @e50 (final)** (V-JEPA) | gold/OOD usc_lss | 256 | **attentive** | **0.530** | **0.486** | 0.539 | 0.549 | 0.024 |
| _pending_ | pretrained | pseudo/75-spk | 128 | tcn | | | | | |

> **UPDATE 2026-06-10 — "baselines > V-JEPA" was a PROBE artifact, not res.** With the
> spatial-aware probe, **tssl_128 (κ 0.475) > supervised ViT-L/16 (0.368)** at the *same*
> 128px — the gap was the mean-pool probe discarding spatial structure, not the resolution.
> For a fully fair fight the **image baselines also need a spatial probe** (`baselines.py`
> tubelet-pools spatial away today). The 256px V-JEPA run + spatial probe is the eventual
> apples-to-apples headline. (Old caveat: image baselines ran native 224–518, V-JEPA at 128.)

> **UPDATE 2026-06-13 — 256px eval of the HALFWAY (epoch-22) T-SSL checkpoint** (6 combos,
> frozen snapshot, gold/OOD usc_lss). Three takeaways:
> 1. **T-SSL lifts ~+0.05 κ across every head at 256px**, even at epoch 22/50:
>    tcn 0.303→0.356, tcn_spatial 0.407→0.461, attentive 0.446→**0.496**.
> 2. **tssl_256 @e22 + attentive (κ 0.496) already beats the fully-trained tssl_128 (0.475)**
>    and the best image baseline (sup ViT-L/16, 0.368). The epoch-50 finish should go higher.
> 3. **Resolution helps the frozen baseline hard**: frozen attentive 128px 0.327 → 256px 0.446
>    (+0.119); so 256px is the right res. `attentive > tcn_spatial > tcn` holds at both encoders.
> **CACHE CAVEAT for the epoch-50 eval:** the @e22 features are cached as `tssl256e22*` /
> `tssl256e22sp*`; `_tag()` ignores the ckpt path, so the FINAL eval MUST use tag `tssl256`
> (not `tssl256e22`) or first `rm feat_cache/phoneme/tssl256e22*`, else it silently re-reads
> epoch-22 features. JSONs: `eval/phoneme_usc_lss_{pretrained256,tssl256e22}*_*.json`.
