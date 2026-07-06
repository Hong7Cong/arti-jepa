#!/usr/bin/env bash
# Phase 0 data engineering: manifest -> subject-disjoint splits -> grayscale stats.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"

DATA_ROOT="${1:-/data1/span_data/rtmri75s}"
mkdir -p "${ARTI_OUT}"

echo "== build manifest (with decord probe of fps/n_frames) =="
python -m artijepa.build_manifest \
  --data-root "${DATA_ROOT}" \
  --out "${ARTI_OUT}/manifest.csv" --probe

echo "== subject-disjoint splits =="
python -m artijepa.splits \
  --manifest "${ARTI_OUT}/manifest.csv" \
  --out "${ARTI_OUT}/manifest_split.csv" \
  --val-frac 0.12 --test-frac 0.12 --seed 0

echo "== global grayscale stats (256px) =="
python -m artijepa.compute_stats \
  --manifest "${ARTI_OUT}/manifest_split.csv" \
  --out "${ARTI_OUT}/grayscale_stats.json" \
  --spatial-size 256 --max-clips 300

echo "DONE -> ${ARTI_OUT}/manifest_split.csv , grayscale_stats.json"
