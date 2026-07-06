#!/usr/bin/env bash
# Phase 0 (extra corpus): add the longitudinal rtMRI set to the T-SSL pre-training
# pool. Builds a longitudinal manifest, concatenates it with the 75-speaker
# all-train manifest, and recomputes global grayscale stats over the combination.
#
# Prereqs: scripts/01_prepare_data.sh already produced manifest_alltrain.csv
#          (the 75-speaker all-data pretrain manifest the 256 config uses).
#
# Outputs (under $ARTI_OUT):
#   manifest_longitudinal.csv   longitudinal corpus, every row split=train
#   manifest_combined.csv       75-speaker all-train  +  longitudinal  (one CSV)
#   grayscale_stats_combined.json   channel-norm stats over the combined train set
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

LONGI_ROOT="${1:-/data1/span_data/longitudinal}"
SPEAKER75_MANIFEST="${2:-${ARTI_OUT}/manifest_alltrain.csv}"
mkdir -p "${ARTI_OUT}"

if [[ ! -f "${SPEAKER75_MANIFEST}" ]]; then
  echo "ERROR: ${SPEAKER75_MANIFEST} not found -- run scripts/01_prepare_data.sh first" >&2
  exit 1
fi

echo "== build longitudinal manifest (n_frames/fps from filename tokens) =="
# add --probe --workers 16 to verify n_frames/fps against decord (slow).
python -m artijepa.build_manifest_longitudinal \
  --data-root "${LONGI_ROOT}" \
  --out "${ARTI_OUT}/manifest_longitudinal.csv"

echo "== merge 75-speaker + longitudinal -> combined manifest =="
python -m artijepa.merge_manifests \
  --inputs "${SPEAKER75_MANIFEST}" "${ARTI_OUT}/manifest_longitudinal.csv" \
  --out "${ARTI_OUT}/manifest_combined.csv"

echo "== global grayscale stats over the COMBINED train set (256px) =="
python -m artijepa.compute_stats \
  --manifest "${ARTI_OUT}/manifest_combined.csv" \
  --out "${ARTI_OUT}/grayscale_stats_combined.json" \
  --spatial-size 256 --max-clips 300

echo "DONE -> ${ARTI_OUT}/manifest_combined.csv , grayscale_stats_combined.json"
echo "      point configs/tssl_vitl_256.yaml at these (already done on main)."
