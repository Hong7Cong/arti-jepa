#!/usr/bin/env bash
# Fast end-to-end validation: unit checks + a 3-step ViT-tiny T-SSL run on CPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_env.sh"
python "${DEV_DIR}/tests/test_smoke.py"
