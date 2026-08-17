#!/bin/bash
#SBATCH --job-name=stut_attn_vmae
#SBATCH --account=shrikann_35
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch1/hongn/artijepa/eval/stutter_binary/slurm_%x_%j.log
#
# Stuttering BINARY, full-grid ATTENTIVE probe (NO spatial pooling) -- VideoMAE arm.
# Extends docs/STUTTERING.md §12 phase 2c (which covered only vjepa_pt + tssl256) to the
# two VideoMAE encoders, so the spatial-detail dissociation can be tested on a second
# encoder family with the SAME contrast (Kinetics SSL vs rt-MRI continue-pretrain):
#   videomae_pt    MCG-NJU/videomae-large  (Kinetics SSL)
#   videomae_tssl  rt-MRI continue-pretrain checkpoint-214
#
# Matched to the JEPA full-grid runs on everything the protocol controls: same rows /
# duration-matched negatives (build_seed 0), same LOSO folds over the 7 PWS, balanced CE,
# probe `attentive` over the full [T'*S', D] grid, macro-F1 primary, 3 seeds (0 1 2).
#
# GEOMETRY CAVEAT (why this is not literally "32f"). The JEPA arm runs 256px/32f/tubelet2
# -> T'=16, S'=256 -> 4096 tokens. VideoMAE-L is a FIXED 16-frame / 224px model: its
# HF backbone resamples any input to 16 frames, so load_frozen_encoder pins
# frames_per_clip=16, spatial_size=224, intensity_norm=minmax (see eval_stutter_binary.py
# "videomae" branch) -- passing --frames 32 would be silently overridden. Each encoder
# therefore runs at its OWN native geometry, exactly as in the published pooled_attentive
# cross-encoder table. Grid here: T'=8 x S'=196 = 1568 tokens/clip.
#   feature cache = 3901 x 1568 x 1024 fp16 ~= 12 GiB per encoder (vs ~31 GiB for JEPA).
# Both clips still span the SAME event window (the loader samples num_frames uniformly
# across the event), so temporal coverage is matched even though token counts are not.
#
# Seed 0 pays the extraction; seeds 1 and 2 are cache hits (frozen encoder -> features are
# seed-independent, only the val split + probe init change). 3 seeds run serially per job;
# the 12 h wall is generous (the 4096-token JEPA grid needed ~25 min/seed on cache hit).
#
# --save-probe writes the best-val probe weights for EVERY LOSO fold to
#   /scratch1/hongn/artijepa/eval/stutter_binary/probes/<feature-tag>_attentive_binary_loso_s<seed>/fold_<PWS>.pt
# each holding state_dict + probe_kwargs + dim/t_steps/classes + val macro-F1 + best epoch
# + the feature-cache tag, i.e. everything needed to rebuild and re-run the probe later.
#
#   sbatch dev_artiJEPA/scripts/29_stutter_binary_attentive_videomae.sbatch                 # both encoders
#   sbatch dev_artiJEPA/scripts/29_stutter_binary_attentive_videomae.sbatch videomae_tssl   # one
#   SEEDS=0 sbatch dev_artiJEPA/scripts/29_stutter_binary_attentive_videomae.sbatch         # extraction + 1 seed
set -euo pipefail
# Code root. Like 27_*.sbatch this defaults to the LIVE checkout: the wt_stutter_sweep
# worktree used by 23_*.sbatch is pinned to branch stutter-binary-sweep, which does NOT
# contain --save-probe -- running there would silently execute stale code. For run
# isolation, commit this branch, `git worktree add <dir> <commit>`, and pass WT=<dir>.
WT="${WT:-/project2/shrikann_35/hongn/vjepa2/dev_artiJEPA}"
source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh
conda activate /scratch1/hongn/conda/envs/artijepa
export PYTHONPATH="/project2/shrikann_35/hongn/vjepa2:${WT}"
export HF_HOME=/scratch1/hongn/huggingface_checkpoints
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# compute-node /tmp is RAM-backed and charged to the job's memory cgroup
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd "$WT"
CFG="$WT/configs/eval_stutter_binary.yaml"
VMAE_TSSL=/scratch1/hongn/videomae_ct/runs/vitl_rtmri_combined_ct/checkpoint-214.pth

run_enc () {
  local tag="$1"; shift
  for seed in ${SEEDS:-0 1 2}; do   # seed 0 extracts the ~12 GiB grid; 1,2 are cache hits
    echo "############ ${tag} attentive seed=${seed} | $(date '+%F %T') ############"
    python -m artijepa.eval_stutter_binary --config "$CFG" --probe attentive \
        --split loso --seed "$seed" --tag "$tag" --save-probe "$@"
  done
}

WHICH=("$@"); [ ${#WHICH[@]} -eq 0 ] && WHICH=(videomae_pt videomae_tssl)
for tag in "${WHICH[@]}"; do
  case "$tag" in
    videomae_pt)   run_enc videomae_pt   --videomae-name MCG-NJU/videomae-large ;;
    videomae_tssl) run_enc videomae_tssl --videomae-checkpoint "$VMAE_TSSL" ;;
    *) echo "unknown tag $tag (expected videomae_pt | videomae_tssl)" >&2; exit 1 ;;
  esac
done
echo "== videomae full-grid attentive sweep done =="
