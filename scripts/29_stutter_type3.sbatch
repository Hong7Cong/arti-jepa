#!/bin/bash
#SBATCH --job-name=stut_type3
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/stutter_type/slurm_%x_%j.log
#
# Stuttering DISFLUENCY-TYPE (Task 8c): block / rep / pro, frozen T-SSL ckpt_215
# (combined) + attentive_lstm probe, LOSO over the 7 PWS, probe weights saved per fold.
#
# Feature cache is SHARED with the binary eval: tag `tssl256_215` @ 256px/32f/full-grid
# already exists (3901 x 4096 x 1024 fp16, 31 GiB) -> every run below is a cache HIT
# and does zero video decoding. Do NOT change data.{neg_per_pos,build_seed,min_dur,
# max_dur,merge_gap,frames_per_clip,spatial_size,event_pad_s} or the hash changes and
# it re-extracts (~13 min GPU + 31 GiB).
#
#   sbatch dev_artiJEPA/scripts/29_stutter_type3.sbatch                # type3, seeds 0 1 2
#   sbatch dev_artiJEPA/scripts/29_stutter_type3.sbatch type4          # + fluent class
#   TASKS="type3 type4" SEEDS=0 sbatch dev_artiJEPA/scripts/29_stutter_type3.sbatch
set -euo pipefail
REPO=/project2/shrikann_35/hongn/vjepa2
DEV="$REPO/dev_artiJEPA"
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/artijepa
export PYTHONPATH="$REPO:$DEV"
export HF_HOME=/scratch1/hongn/huggingface_checkpoints
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd "$DEV"

CFG="$DEV/configs/eval_stutter_type.yaml"
mkdir -p /scratch1/hongn/artijepa/eval/stutter_type

TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(${TASKS_ENV:-type3})
for task in "${TASKS[@]}"; do
  for seed in ${SEEDS:-0 1 2}; do
    echo "############ ${task} attentive_lstm tssl256_215 seed=${seed} | $(date '+%F %T') ############"
    python -m artijepa.eval_stutter_type --config "$CFG" \
        --task "$task" --probe attentive_lstm --split loso --seed "$seed"
  done
done
echo "== stutter type sweep done =="
