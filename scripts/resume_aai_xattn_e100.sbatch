#!/bin/bash
#SBATCH --job-name=aai_xattn
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=47:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_xattn/slurm_%j.log
#
# Resume AAI-JEPA (L9, ctx1, CROSS-ATTENTION) from latest.pt to 100 epochs.
# ~82 epochs left @ ~56 min/ep > 48h cap, so this job resubmits itself ~10 min
# before the wall limit; each launch resumes from the newest latest.pt (atomic,
# saved every epoch). Extending 20->100 re-stretches the cosine LR over 100 ep.
#
# HOW TO RUN
#   sbatch dev_artiJEPA/scripts/resume_aai_xattn_e100.sbatch
# MONITOR
#   squeue -u $USER
#   tail -f /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_xattn/slurm_<jobid>.log
#   tail -f /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_xattn/train_log.csv
# CANCEL (also stop the self-resubmit chain)
#   scancel <jobid>

set -uo pipefail
SELF=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/resume_aai_xattn_e100.sbatch
CONFIG=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/configs/aai_wavlm_256_combined_L9_ctx1_xattn_e100.yaml
FOLDER=/scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_xattn
RESUME="$FOLDER/latest.pt"

# resubmit ~10 min before the wall limit (SIGUSR1 from --signal)
resubmit() { echo "[$(date)] SIGUSR1 -> resubmitting $SELF"; sbatch "$SELF"; exit 0; }
trap resubmit USR1

# env: conda artijepa + PYTHONPATH (REPO_ROOT + dev_artiJEPA); cd's to REPO_ROOT
source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh

# job 10132117 OOM'd at ep18 during the AR rollout: 30 MiB short with 2.2 GB
# reserved-but-fragmented on the V100-32GB (peak was predicted ~22.7 GB but hit
# 29 GB allocated). expandable_segments defrags the caching allocator and recovers
# far more than the 30 MiB needed -- low-risk, no batch/eff_batch change.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "########## AAI xattn resume (job $SLURM_JOB_ID) $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "resume=$RESUME  config=$CONFIG"

srun python -u -m artijepa.aai_train --config "$CONFIG" --resume "$RESUME" &
wait
echo "########## job $SLURM_JOB_ID exited $(date) ##########"
