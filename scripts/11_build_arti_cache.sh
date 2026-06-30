#!/usr/bin/env bash
# Build the articulator (+ MRI frame) cache from usc_lss *_mview.mat sessions and the
# session-disjoint manifest (aucjepa_plans_new.md M0). Runs in the plain `artijepa`
# env -- scipy only, NO transformers (unlike the trashed WavLM audio cache).
#   bash scripts/11_build_arti_cache.sh [--limit N]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

PYTHONPATH="${REPO_ROOT}:${DEV_DIR}" python -m artijepa.arti_cache \
  --mat-dir /scratch1/hongn/usc_lss/articulators \
  --out "${ARTI_OUT}/arti_feats/usc_lss" \
  --manifest "${ARTI_OUT}/arti_manifest.csv" "$@"
