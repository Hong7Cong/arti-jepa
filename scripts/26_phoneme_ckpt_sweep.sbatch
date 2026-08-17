#!/bin/bash
#SBATCH --job-name=ph_cksweep
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G          # spatial cache ~200 GiB mmap page-cache headroom (mirrors 25_*.sbatch)
#SBATCH --time=30:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/cksweep/slurm_%x_%j.log
#
# KAPPA-vs-CKPT SWEEP -- Phase-1 headline probe (`attentive` x CE) run across the
# tssl_vitl_256_combined pretraining checkpoints, so we can plot in-domain /
# x-domain phoneme kappa as a function of pretraining epoch.
#
# WHY: RESULTS_phonepred.md Phase-1 headline has only ckpt_100 (tssl256comb100) and
#   ckpt_215 (tssl256comb215). This fills in the intermediate checkpoints
#   (60, 150, 200) with the SAME probe/loss/config so all points are comparable.
#
# PROBE: plain `attentive` (V-JEPA AttentivePooler over the 256 spatial tokens per
#   temporal step -> linear), chunk-mode, CE. This is the Phase-1 HEADLINE head
#   (reports kappa + PER) -- NOT the Phase-2 `attentive_lstm`. Config default already
#   is attentive/ce; passed explicitly here for clarity.
#
# HASH/CACHE: the feature-cache hash (eval_phoneme._tag) keys on encoder `spec`
#   (= the --encoder path, which includes the ckpt filename) + size/fps/manifest, so
#   each ckpt gets its OWN cache + hash automatically -- no stale-cache collision.
#   These caches do NOT exist yet (unlike comb100/comb215 from Phase 1/2), so each
#   job does a full spatial feature extraction (~200 GiB) ONCE, then 3 seeds share it
#   (only the ~1-2 min probe stage re-runs per seed).
#
# HOW TO RUN (one job per ckpt epoch; seeds default "0 1 2"):
#   cd /project2/shrikann_35/hongn/vjepa2
#   S=dev_artiJEPA/scripts/26_phoneme_ckpt_sweep.sbatch
#   for E in 60 150 200; do sbatch $S $E; done
#   # optional 2nd arg = seed list, e.g.  sbatch $S 60 "0"
# MONITOR:
#   squeue -u $USER -o "%.12i %.14j %.8T %.10M %R" | grep ph_cksweep
#   tail -f /scratch1/hongn/artijepa/eval/cksweep/slurm_ph_cksweep_<jobid>.log
# OUTPUTS (meta.out = /scratch1/hongn/artijepa/eval):
#   phoneme_usc_lss_tssl256comb<E>sp_<hash>_attentive_ce_s<seed>.{json,pt}

set -uo pipefail
EPOCH="${1:?usage: sbatch 26_phoneme_ckpt_sweep.sbatch <epoch: 60|100|150|200|215> [seeds]}"

source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=dev_artiJEPA/configs/eval_phoneme_annot16_combined.yaml
TSSL_DIR=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined
CKPT="$TSSL_DIR/ckpt_${EPOCH}.pt"
TAG="tssl256comb${EPOCH}"
mkdir -p /scratch1/hongn/artijepa/eval/cksweep

[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 2; }

SEEDS="${2:-0 1 2}"

echo "########## phoneme ckpt-sweep: ckpt_${EPOCH} (tag=$TAG) attentive x ce seeds='$SEEDS' job ${SLURM_JOB_ID:-local} $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for SEED in $SEEDS; do
  echo "===== [comb${EPOCH}] seed $SEED START $(date) ====="
  srun python -u -m artijepa.eval_phoneme \
      --config "$CONFIG" --dtype float16 \
      --encoder "$CKPT" --tag "$TAG" --probe attentive --loss ce --seed "$SEED"
  echo "===== [comb${EPOCH}] seed $SEED END   $(date) ====="
done
echo "########## ckpt_${EPOCH} done $(date) ##########"
