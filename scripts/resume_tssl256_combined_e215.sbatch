#!/bin/bash
#SBATCH --job-name=tssl256c
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=47:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/slurm_%j.log
#
# Resume the ViT-L 256px T-SSL pretraining on the COMBINED corpus from latest.pt
# to the full 215 epochs. Snapshots ckpt_150.pt + ckpt_200.pt fall out of the
# config's meta.snapshot_freq=50 (tssl_train saves ckpt_<epoch>.pt whenever the
# just-finished epoch is a multiple of 50) -- 50/100 already exist, 150 & 200 are
# the two remaining in-range, 250 > 215. No config edit needed for the requested saves.
#
# ~70 epochs left @ ~60 min/ep (~70 h) > the 48 h wall cap, so this job resubmits
# ITSELF ~10 min before the wall limit (SIGUSR1 from --signal). Each launch resumes
# from the newest latest.pt (saved atomically every epoch); the LR/EMA cosine
# schedule is fast-forwarded to the resumed step, so it picks up exactly where it
# stopped. The chain self-terminates once epoch 215 is reached (guard below).
#
# !! DO NOT run this while the current INTERACTIVE training is still live -- two
#    processes writing runs/tssl_vitl_256_combined/latest.pt will corrupt the
#    checkpoint. Stop the interactive run first (it's the b11-14 / d11-02 alloc),
#    then submit this. Verify nothing is writing: `ls -l .../latest.pt` twice.
#
# HOW TO RUN
#   sbatch dev_artiJEPA/scripts/resume_tssl256_combined_e215.sbatch
# MONITOR
#   squeue -u $USER
#   tail -f /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/slurm_<jobid>.log
#   tail -f /scratch1/hongn/artijepa/runs/tssl_vitl_256_combined/train_log.csv
# CANCEL (also stops the self-resubmit chain)
#   scancel <jobid>

set -uo pipefail
SELF=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/resume_tssl256_combined_e215.sbatch
CONFIG=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/configs/tssl_vitl_256_combined.yaml
FOLDER=/scratch1/hongn/artijepa/runs/tssl_vitl_256_combined
RESUME="$FOLDER/latest.pt"
TARGET_EPOCHS=215

# Last finished epoch (1-indexed count) from diagnostics; -1 if unreadable.
last_epoch() { tail -1 "$FOLDER/diagnostics.jsonl" 2>/dev/null | sed -n 's/.*"epoch": *\([0-9]*\).*/\1/p'; }

# Resubmit ~10 min before the wall limit (SIGUSR1 from --signal), but only if
# there is still training left to do -- avoids an infinite chain of instant-exit
# jobs once epoch 215 is complete.
resubmit() {
    ep=$(last_epoch)
    if [ -n "$ep" ] && [ "$ep" -ge "$((TARGET_EPOCHS - 1))" ]; then
        echo "[$(date)] SIGUSR1 but epoch $ep >= target -> DONE, not resubmitting"; exit 0
    fi
    echo "[$(date)] SIGUSR1 -> resubmitting $SELF (last epoch=$ep)"; sbatch "$SELF"; exit 0
}
trap resubmit USR1

# env: conda artijepa + PYTHONPATH (REPO_ROOT + dev_artiJEPA); cd's to REPO_ROOT.
source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh

# Match the original launch: expandable segments to avoid 256px fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "########## T-SSL 256 combined resume (job $SLURM_JOB_ID) $(date) on $(hostname) ##########"
echo "config=$CONFIG"
echo "resume=$RESUME  last finished epoch=$(last_epoch)  target=$TARGET_EPOCHS"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

if [ ! -f "$RESUME" ]; then
    echo "FATAL: $RESUME not found -- nothing to resume from."; exit 1
fi

# tssl_train reads meta.resume (set here via --resume), fast-forwards the schedule,
# and continues to optimization.epochs (215). snapshot_freq=50 handles ckpt_150/200.
srun python -u -m artijepa.tssl_train --config "$CONFIG" --resume "$RESUME" &
wait
echo "########## job $SLURM_JOB_ID exited $(date) last epoch=$(last_epoch) ##########"
