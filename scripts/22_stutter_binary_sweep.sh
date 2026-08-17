#!/usr/bin/env bash
# Cross-ENCODER binary fluent-vs-disfluent sweep (docs/STUTTERING.md §12, phase 2).
# Frozen encoder -> pooled_attentive probe (mean spatial -> attend time; tiny cache),
# leave-one-speaker-out over the 7 PWS, seed 0. OpenCV `stutter_binary` loader
# (pal8-safe -- do NOT use the decord eval_disfluency path on this corpus).
#
# Matched protocol across every encoder: same rows/negatives (build_seed 0), same
# LOSO folds, balanced CE, macro-F1 primary. Each encoder runs at its OWN native
# geometry/normalisation (the ported multi-encoder load_frozen_encoder handles it):
#   tssl256        Arti-JEPA T-SSL ViT-L, 256px/32f, z-score  (rt-MRI fine-tune)
#   vjepa_pt       FAIR V-JEPA2 ViT-L pretrained, 256px/32f, z-score
#   videomae_pt    MCG-NJU VideoMAE-L, 224px/16f, minmax      (Kinetics SSL)
#   videomae_tssl  VideoMAE-L rt-MRI continue-pretrain ckpt-214, 224px/16f, minmax
#   vitl           supervised ViT-L/16 (timm augreg), per-frame, minmax
#   dinov2         DINOv2 ViT-L/14 (timm), per-frame, minmax
#
#   bash scripts/22_stutter_binary_sweep.sh                 # all six, GPU 0
#   bash scripts/22_stutter_binary_sweep.sh tssl256 vitl    # a subset
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/scratch1/hongn/huggingface_checkpoints}"
export PYTHONUNBUFFERED=1                 # stream progress to the tee'd logs live
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CFG="${DEV_DIR}/configs/eval_stutter_binary.yaml"
PROBE="${PROBE:-pooled_attentive}"      # tiny spatial-pooled cache; matched setting
VITL_PT="/scratch1/hongn/artijepa/checkpoints/vitl.pt"
VMAE_TSSL="/scratch1/hongn/videomae_ct/runs/vitl_rtmri_combined_ct/checkpoint-214.pth"
LOG_DIR="${ARTI_OUT}/eval/stutter_binary/sweep_logs"; mkdir -p "${LOG_DIR}"

run_one () {
  local tag="$1"; shift
  echo "############################################################"
  echo "## ${tag} | probe=${PROBE} | $(date '+%F %T')"
  echo "############################################################"
  python -m artijepa.eval_stutter_binary --config "${CFG}" \
      --probe "${PROBE}" --split loso --tag "${tag}" "$@" \
      2>&1 | tee "${LOG_DIR}/${tag}_${PROBE}.log"
}

encoder_args () {   # tag -> the encoder-selecting flags
  case "$1" in
    tssl256)       echo "" ;;                                                   # config default ckpt
    vjepa_pt)      echo "--spec pretrained --checkpoint ${VITL_PT}" ;;
    videomae_pt)   echo "--videomae-name MCG-NJU/videomae-large" ;;
    videomae_tssl) echo "--videomae-checkpoint ${VMAE_TSSL}" ;;
    vitl)          echo "--model vitl" ;;
    dinov2)        echo "--model dinov2" ;;
    *) echo "__UNKNOWN__" ;;
  esac
}

WHICH=("$@")
[ ${#WHICH[@]} -eq 0 ] && WHICH=(tssl256 vjepa_pt videomae_pt videomae_tssl vitl dinov2)

for tag in "${WHICH[@]}"; do
  args="$(encoder_args "${tag}")"
  if [ "${args}" = "__UNKNOWN__" ]; then echo "unknown encoder tag: ${tag}" >&2; exit 1; fi
  # shellcheck disable=SC2086
  run_one "${tag}" ${args}
done
echo "== sweep done. results in ${ARTI_OUT}/eval/stutter_binary/ =="
