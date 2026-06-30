#!/usr/bin/env bash
# Run the AC-JEPA planner: CEM search over articulator actions toward a goal phoneme,
# scored by an energy on the frozen world model's rolled-out latent (plan §3-§5).
#   bash scripts/14_plan_acjepa.sh <world_model.pt> <target_phoneme> [--seed-clip N] [config]
# e.g. bash scripts/14_plan_acjepa.sh \
#        /scratch1/hongn/artijepa/runs/acjepa_arti6_256/latest.pt m
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

WM="${1:?need world-model checkpoint (P latest.pt)}"
TARGET="${2:?need goal phoneme (ARPABET, e.g. m) or comma string}"
shift 2 || true
CONFIG="${DEV_DIR}/configs/acjepa_plan_256.yaml"

export PYTHONUNBUFFERED=1
python -m artijepa.acjepa_plan --config "${CONFIG}" \
  --world-model "${WM}" --target "${TARGET}" "$@"
