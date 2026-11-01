#!/usr/bin/env bash
set -euo pipefail
BASE=/home/chengyetai/workspace/current_exp/chunk_manual_20260927
if [[ "${1:-}" == start ]]; then
  bash "$BASE/reference_service.sh" "${2:-23}"
  WEIGHT_SOURCE=1 DEEPEP_FIX=0 WEIGHT_REFERENCE_URL=http://127.0.0.1:30122 bash "$BASE/serve.sh" "$@"
else
  bash "$BASE/serve.sh" "$@"
fi
