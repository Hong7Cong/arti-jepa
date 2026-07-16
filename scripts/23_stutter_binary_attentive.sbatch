#!/bin/bash
#SBATCH --job-name=stut_attn
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/stutter_binary/slurm_%x_%j.log
#
# Stuttering BINARY, full-grid ATTENTIVE probe (NO spatial pooling) @ fixed 32f.
# Hypothesis: the per-frame SPATIAL token structure carries articulatory-dynamics
# information that pooled_attentive (mean over S') destroys. This keeps the full
# [T'*S', D] = 4096-token grid and lets one AttentivePooler attend over ALL tokens
# (space+time jointly), so a spatial-detail gain -- if real -- can surface.
#
# Encoders: FAIR-pretrained V-JEPA2 (vjepa_pt) and combined T-SSL ckpt_100 (tssl256),
# both 256px/32f/z-score. 3 seeds each; features extracted ONCE per encoder
# (pool_mode=none, ~31 GiB fp16 cache) then reused across seeds (frozen -> seed-indep).
# Compare against the pooled_attentive CIs from run_3seed_pooled.sh.
#
# Runs from the ISOLATED worktree of branch stutter-binary-sweep so a concurrent
# session editing the main checkout cannot change the code mid-job.
#   sbatch dev_artiJEPA/scripts/23_stutter_binary_attentive.sbatch
#   sbatch dev_artiJEPA/scripts/23_stutter_binary_attentive.sbatch tssl256   # one encoder
set -euo pipefail
WT=/scratch1/hongn/artijepa/wt_stutter_sweep
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/artijepa
export PYTHONPATH="/project2/shrikann_35/hongn/vjepa2:${WT}"
export HF_HOME=/scratch1/hongn/huggingface_checkpoints
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$WT"
CFG="$WT/configs/eval_stutter_binary.yaml"
VITL=/scratch1/hongn/artijepa/checkpoints/vitl.pt

run_enc () {
  local tag="$1"; shift
  for seed in ${SEEDS:-0 1 2}; do   # SEEDS env overrides; seed 0 extracts the 31 GiB grid, 1,2 are cache hits
    echo "############ ${tag} attentive seed=${seed} | $(date '+%F %T') ############"
    python -m artijepa.eval_stutter_binary --config "$CFG" --probe attentive \
        --split loso --seed "$seed" --tag "$tag" "$@"
  done
}

WHICH=("$@"); [ ${#WHICH[@]} -eq 0 ] && WHICH=(vjepa_pt tssl256)
for tag in "${WHICH[@]}"; do
  case "$tag" in
    tssl256)  run_enc tssl256 ;;                                   # config default ckpt_100
    vjepa_pt) run_enc vjepa_pt --spec pretrained --checkpoint "$VITL" ;;
    *) echo "unknown tag $tag" >&2; exit 1 ;;
  esac
done
echo "== attentive full-grid sweep done =="
