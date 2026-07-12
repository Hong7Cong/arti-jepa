#!/usr/bin/env bash
# Build the stuttering disfluency manifest (Task 8): scan every PWS TextGrid,
# canonicalize <phoneme>_<type> labels, emit one row per labeled event.
#   bash scripts/15_build_stutter_manifest.sh                     # primary tier only
#   bash scripts/15_build_stutter_manifest.sh --fluent-per-file 3 # + fluent negatives
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
ROOT="${STUTTER_ROOT:-/data1/span_data/stuttering}"
FLUENT=0
[ "${1:-}" = "--fluent-per-file" ] && FLUENT="${2:-3}"
python - "$ROOT" "$FLUENT" <<'PY'
import sys
from artijepa.stutter import build_manifest
root, fluent = sys.argv[1], int(sys.argv[2])
build_manifest(root=root, fluent_per_file=fluent)
PY
