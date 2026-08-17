#!/bin/bash
#SBATCH --job-name=aai_film
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=47:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_film/slurm_%j.log
#
# Resume AAI-JEPA (L9, ctx1, FiLM conditioning) from latest.pt @ ep20 to 50 epochs.
# ~30 epochs left @ ~<1 h/ep may exceed the 48h cap, so this job resubmits itself
# ~10 min before the wall limit; each launch resumes from the newest latest.pt
# (atomic, saved every epoch). Extending 20->50 re-stretches the cosine LR over 50 ep.
#
# HOW TO RUN
#   sbatch dev_artiJEPA/scripts/resume_aai_film_e50.sbatch
# MONITOR
#   squeue -u $USER
#   tail -f /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_film/slurm_<jobid>.log
#   tail -f /scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_film/train_log.csv
# CANCEL (also stop the self-resubmit chain)
#   scancel <jobid>

set -uo pipefail
SELF=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/resume_aai_film_e50.sbatch
CONFIG=/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/configs/aai_wavlm_256_combined_L9_ctx1_film_e50.yaml
FOLDER=/scratch1/hongn/artijepa/runs/aai_wavlm_256_combined_L9_ctx1_film
RESUME="$FOLDER/latest.pt"

# resubmit ~10 min before the wall limit (SIGUSR1 from --signal)
resubmit() { echo "[$(date)] SIGUSR1 -> resubmitting $SELF"; sbatch "$SELF"; exit 0; }
trap resubmit USR1

# env: conda artijepa + PYTHONPATH (REPO_ROOT + dev_artiJEPA); cd's to REPO_ROOT
source /project2/shrikann_35/hongn/vjepa2/dev_artiJEPA/scripts/_env.sh

# same allocator defrag guard used for the xattn resume (AR rollout can fragment
# the V100-32GB caching allocator near peak); low-risk, no batch/eff_batch change.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "########## AAI film resume (job $SLURM_JOB_ID) $(date) on $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "resume=$RESUME  config=$CONFIG"

srun python -u -m artijepa.aai_train --config "$CONFIG" --resume "$RESUME" &
wait
echo "########## job $SLURM_JOB_ID exited $(date) ##########"
