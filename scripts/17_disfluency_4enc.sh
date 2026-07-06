#!/usr/bin/env bash
# 4-encoder frozen disfluency-type comparison (2026-06-30 request):
# V-JEPA2 pretrained / VideoMAE-L / Google ViT-L / DINOv3 ViT-L, frozen features,
# attentive-probe head only, 50 fps, 100-frame clips (VideoMAE 16f), ONE fixed
# subject-disjoint split (test=PWS5, val=PWS10). Each encoder extracts its feature
# cache once (task-independent), then scores type3 and type5 from that cache.
#   bash scripts/17_disfluency_4enc.sh              # all four, GPU 0
#   GPU=1 bash scripts/17_disfluency_4enc.sh dinov3 # one encoder on GPU 1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HOME="${HF_HOME:-/data2/foundation_models_checkpoints}"
CFG="${DEV_DIR}/configs/eval_disfluency_100f.yaml"
LOG_DIR="${ARTI_OUT}/eval/disfluency_4enc"
mkdir -p "${LOG_DIR}"

# alias -> eval_disfluency invocation flags + result tag
run_encoder () {
  local name="$1"; shift
  for task in type3 type5; do
    echo "############################################################"
    echo "## ${name} | ${task} | $(date '+%F %T')"
    echo "############################################################"
    python -m artijepa.eval_disfluency --config "${CFG}" --task "${task}" --tag "${name}" "$@"
  done
}

WHICH=("$@")
[ ${#WHICH[@]} -eq 0 ] && WHICH=(vjepa2_pt videomae vitl_google dinov3)

for enc in "${WHICH[@]}"; do
  case "${enc}" in
    vjepa2_pt)   run_encoder vjepa2_pt   ;;                       # frozen pretrained V-JEPA2
    videomae)    run_encoder videomae    --model videomae ;;      # VideoMAE-L (16f native)
    vitl_google) run_encoder vitl_google --model vitl ;;          # Google supervised ViT-L
    dinov3)      run_encoder dinov3      --model dinov3 ;;         # DINOv3 ViT-L
    *) echo "unknown encoder alias: ${enc}" >&2; exit 1 ;;
  esac
done
echo "== done. results in ${LOG_DIR}/ =="
