# Arti-JEPA — RUNME (reproduce the numbers in `RESULTS.md`)

This reproduces every phoneme row in `RESULTS.md` **inference-only**: the encoder
checkpoints are **frozen** (`@torch.no_grad` feature extraction) — we never
re-train an encoder here. Only the small per-token probe trains (cheap, minutes on
any GPU). Encoder pretraining itself is in `TODO_pretraining.md`; settings tables
are in `RESULTS.md`; file map in `Master.md`.

> **Frozen encoder ⇒ deterministic, seed-independent feature caches.** Re-running an
> eval reuses `feat_cache/phoneme/<tag>_<hash>/` and only retrains the probe, so
> seeds/heads are cheap to sweep. Results land as `eval/phoneme_usc_lss_<tag>_*.json`.

---

## 0. Prerequisites

```bash
cd /project2/shrikann_35/hongn/vjepa2          # = $REPO_ROOT
source dev_artiJEPA/scripts/_env.sh            # conda artijepa (torch 2.6+cu124)
nvidia-smi                                     # any CUDA GPU; bf16 wants L40S/A100/H100, fp16 ok on V100
```

Data/artifacts are on `/scratch1` (never `/project2`):
- gold OOD eval data: `/scratch1/hongn/usc_lss/phoneme_manifest.csv`
- grayscale stats: `/scratch1/hongn/artijepa/grayscale_stats.json` (+ `_128.json`)
- outputs: `/scratch1/hongn/artijepa/{eval/,feat_cache/phoneme/}`

---

## 1. Trained encoder checkpoints (SAVE / verify these first)

The headline rows depend on these encoder checkpoints. **Keep them backed up** —
they are the only non-reproducible artifact here (each is one full T-SSL run,
~5.1 GB). Verify they exist before evaluating:

```bash
ls -lh /scratch1/hongn/artijepa/runs/{tssl_vitl_128,tssl_vitl_256,tssl_vitl_256_combined}/latest.pt
```

| tag | checkpoint | what | status |
|---|---|---|---|
| `pretrained*` | stock V-JEPA2 `vitl.pt` (auto-downloaded to `runs`/`checkpoints`) | no T-SSL baseline | always available |
| `tssl128` | `runs/tssl_vitl_128/latest.pt` | 128px T-SSL, ep50 | ✅ saved (5.1 GB) |
| `tssl256` | `runs/tssl_vitl_256/latest.pt` | **256px T-SSL, ep50 (headline)** | ✅ saved (5.1 GB) |
| `tssl256comb` | `runs/tssl_vitl_256_combined/latest.pt` | +longitudinal, ep215 | ⏳ training (see `TODO_pretraining.md`) |

> **Back up a checkpoint** (recommended before reclaiming scratch):
> `cp /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt <durable>/tssl_vitl_256_ep50.pt`
> Training saves `latest.pt` atomically every epoch (+`.prev`); to also keep history
> set `meta.snapshot_freq` in the config (writes `epoch_NN.pt`).

> **⚠ Cache-tag rule:** `eval_phoneme._tag()` hashes the config but **not** the
> `--encoder` path. Always pass a **distinct `--tag` per encoder**, or first
> `rm -r feat_cache/phoneme/<tag>*`, else it silently reuses another encoder's
> cached features.

---

## 2. Headline — T-SSL lift at 256px (gold/OOD `usc_lss`, attentive spatial probe)

Reproduces the **0.530 / 0.527±0.004** headline + its frozen-pretrained reference
(`RESULTS.md` → "256px T-SSL" and "3-seed fair-fight").

```bash
CFG=dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml
EV=dev_artiJEPA/scripts/04_eval_phoneme.sh

# (a) frozen pretrained-256 reference
bash $EV $CFG --tag pretrained256 --probe attentive

# (b) T-SSL 256 @e50 — the headline encoder
bash $EV $CFG --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt \
      --tag tssl256 --probe attentive
```

3 seeds (error bars; reuses the frozen cache, only retrains the probe):
```bash
for s in 0 1 2; do
  bash $EV $CFG --tag pretrained256 --probe attentive --seed $s
  bash $EV $CFG --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256/latest.pt \
        --tag tssl256 --probe attentive --seed $s
done
# aggregate -> mean±sd: python /scratch1/hongn/artijepa/eval/aggregate_spatial.py
```
Swap `--probe attentive` → `tcn_spatial` or `tcn` for the other rows. Each run
prints test κ / PERµ and writes `eval/phoneme_usc_lss_<tag>_..._s<seed>.json`.

---

## 3. 128px T-SSL lift (`RESULTS.md` → "128px T-SSL", spatial-probe keystone)

```bash
CFG=dev_artiJEPA/configs/eval_phoneme_usc_lss.yaml
# mean-pool tcn (original headline) + spatial heads:
for P in tcn tcn_spatial attentive; do
  bash $EV $CFG --tag pretrained128 --probe $P
  bash $EV $CFG --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_128/latest.pt \
        --tag tssl128 --probe $P
done
```

Head × loss ablation (`RESULTS.md` table): `scripts/06_probe_sweep.sh` runs the
`linear/mlp/tcn/lstm/transformer × {CE,CTC} × {pretrained128,tssl128}` grid on the
cached features. Spatial heads alone: `scripts/07_probe_spatial.sh`.

---

## 4. Public image baselines (the fair fight)

Frozen off-the-shelf timm encoders, same probe/labels (`RESULTS.md` → "Image
baselines" + "3-seed fair-fight").

```bash
BCFG=dev_artiJEPA/configs/eval_phoneme_usc_lss_baseline.yaml
# native-res mean-pool tcn (one per model):
for m in clip siglip dinov2 vitl resnet; do
  bash $EV $BCFG --model $m --tag base_$m            # --batch 2 for dinov2 @518px
done
# spatial-probe fair fight (3 seeds) — driver handles caches + seeds:
setsid env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash /scratch1/hongn/artijepa/eval/run_baselines_spatial.sh < /dev/null \
  >> /scratch1/hongn/artijepa/eval/eval_256_fairfight.log 2>&1 &
```
⚠ This run is **PARTIAL** in `RESULTS.md` (sup ViT-L/16, dinov2, resnet still
pending). To finish only the missing models, edit the driver's `for m in …` list →
`vitl dinov2 resnet siglip`. Don't co-launch on a node already running another job
(16 GB host-RAM cap → OOM). See `TODO_eval.md` §2.

---

## 5. Combined (+longitudinal) checkpoint — TBA (blocked on training)

Once `runs/tssl_vitl_256_combined/latest.pt` finishes (`TODO_pretraining.md`):
```bash
bash $EV dev_artiJEPA/configs/eval_phoneme_usc_lss_256.yaml \
      --encoder /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/latest.pt \
      --tag tssl256comb --probe attentive        # fresh tag = no cache collision
```
Compare vs `tssl256` (0.527) and `pretrained256` (0.449) → fills the TBA row.

---

## 6. Task-1 pseudo labels — TBA (env-blocked)

Decoupled (audio model needs `transformers<5` or torch≥2.7):
```bash
# in a transformers-compatible env:
python -c "from artijepa.audio_phoneme import build_pseudo_labels; \
           build_pseudo_labels('/scratch1/hongn/artijepa/manifest_split.csv')"
# back in artijepa env:
bash $EV dev_artiJEPA/configs/eval_phoneme_pseudo.yaml --tag pseudo_pretrained128
```

---

## Where results go
- Per-run JSON: `/scratch1/hongn/artijepa/eval/phoneme_usc_lss_<tag>_*.json`
  (κ, PER, frame-acc, config snapshot).
- Feature caches: `/scratch1/hongn/artijepa/feat_cache/phoneme/<tag>_<hash>/`
  (pooled) and `<tag>sp_<hash>/` (spatial grid).
- Aggregate seeds → mean±sd: `eval/aggregate_spatial.py`. Transcribe into
  `RESULTS.md` (results log + 3-seed table).
