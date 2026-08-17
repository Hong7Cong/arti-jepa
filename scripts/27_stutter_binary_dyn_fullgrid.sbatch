#!/bin/bash
#SBATCH --job-name=stut_dyngrid
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/stutter_binary/slurm_%x_%j.log
#
# Stuttering BINARY, DYNAMIC length x FULL SPATIAL GRID -- `seq_attentive_lstm` probe
# on the combined T-SSL ckpt_215 encoder (tag tssl256comb215).
#
# Why: §12 phase 2c showed that at FIXED 32f, keeping the S' grid lifts the rt-MRI
# fine-tune (+0.032) while hurting the pretrained encoder -- the fine-tune's gain lives
# in spatial tokens that mean-pooling destroys. The dynamic path (§9) so far only ever
# ran SPATIALLY-POOLED (seq_attentive 0.767 / seq_lstm 0.744), i.e. exactly the reduction
# that hides that gain. This run combines the two: real temporal extent AND spatial detail.
#
# The cost is the cache, which scales with sample_fps (ragged [ΣL, S'=256, D=1024] fp16):
#     25 fps -> 62 GiB  |  50 fps -> 110 GiB  |  native ~99 fps -> 204 GiB
# (vs 0.24-0.80 GiB spatially pooled, and 31 GiB for the fixed-32f full grid).
#
# That is the phoneme Phase-2 problem (RESULTS_phonepred.md "Loss landscape (Phase 2)":
# a 149 GiB spatial cache + utterance-mode loaders OOM'ing a 64 GB cgroup), so this path
# reuses its four fixes -- madvise'd extraction, PADDED-token batching, eval loaders at
# 2 workers x prefetch_factor=1, fp16 dataset items. See the module docstring of
# artijepa/eval_stutter_binary_dynamic.py.
#
# WHY --mem=64G IS ENOUGH FOR A 204 GiB CACHE. Host RAM here is decoupled from the cache
# size: it is (train_workers x train_prefetch + eval_workers x 1) x one padded batch
# = (4x2 + 2x1) x 512 MiB = 5.0 GiB of loader buffers, + ~5 GiB torch/CUDA. The cache is
# never resident -- extraction madvise'es its dirty pages away, and the probe's sequential
# memmap reads leave only RECLAIMABLE page cache (the kernel evicts it under pressure
# rather than OOM-killing). This is the same reason the phoneme probe survived its 149 GiB
# cache. NOTE the phoneme job itself ran --mem=96G; 64 GB was the cgroup it OOM'd in
# BEFORE the fixes, so this is the arithmetic above, not a precedent -- raise to 96G if a
# fold ever trips, or halve probe.max_tokens (batch granularity only, not the result).
#
# Seed 0 pays the extraction; seeds 1,2 are cache hits (frozen encoder -> seed-independent
# features). Run seeds as separate jobs if extraction + 3 probes overruns the wall.
#
#   sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch              # native fps, seeds 0 1 2
#   FPS=25 sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch       # cheaper 62 GiB cache
#   SEEDS=0 sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch      # extraction-only job
#   SEEDS="1 2" sbatch dev_artiJEPA/scripts/27_stutter_binary_dyn_fullgrid.sbatch  # cache-hit probes
set -euo pipefail
# Code root. Unlike 23_stutter_binary_attentive.sbatch this defaults to the LIVE checkout:
# the wt_stutter_sweep worktree is pinned to branch stutter-binary-sweep, which does NOT
# contain this eval's full-grid support -- running there would silently execute stale code.
# For run-isolation, commit this branch, `git worktree add <dir> <commit>`, and pass WT=<dir>.
WT="${WT:-/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA}"
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/artijepa
export PYTHONPATH="/project2/shrikann_35/hongn/vjepa2:${WT}"
export HF_HOME=/scratch1/hongn/huggingface_checkpoints
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# compute-node /tmp is RAM-backed and charged to the job's memory cgroup -- keep TMPDIR
# on scratch so nothing competes with the 96G budget the OOM fixes are sized against.
export TMPDIR=/scratch1/hongn/tmp; mkdir -p "$TMPDIR"
cd "$WT"

CFG="$WT/configs/eval_stutter_binary_dyn_fullgrid.yaml"
FPS="${FPS:-native}"
# FOLDS="PWS3" runs a SUBSET of the 7 LOSO folds in this job so the native cell can be
# split across 7 parallel jobs (each ~5h vs ~35h serial past any wall). Val splits stay
# bit-identical to the all-folds run (the eval loop consumes rng for every speaker); each
# job writes its own JSON (`..._loso_PWS3_s0.json`). Merge the 7 with agg after. Empty =>
# the normal all-folds run (writes `..._loso_full_s0.json`).
FOLDARG=""; [ -n "${FOLDS:-}" ] && FOLDARG="--folds ${FOLDS}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h /scratch1 | tail -1

for seed in ${SEEDS:-0 1 2}; do
  echo "############ dyn-fullgrid tssl256comb215 fps=${FPS} folds=${FOLDS:-all} seed=${seed} | $(date '+%F %T') ############"
  python -m artijepa.eval_stutter_binary_dynamic --config "$CFG" \
      --probe seq_attentive_lstm --sample-fps "$FPS" --split loso --seed "$seed" $FOLDARG
done
echo "done $(date '+%F %T')"
