#!/usr/bin/env bash
set -euo pipefail
REFERENCE=/home/chengyetai/workspace/current_exp/chunk_manual_20260927/reference_service.sh
SERVE=/home/chengyetai/workspace/current_exp/mtp_full_20260928/serve.sh
if [[ "${1:-}" == start ]]; then
  bash "$REFERENCE" "${2:-23}"
  WEIGHT_SOURCE=1 DEEPEP_FIX=0 WEIGHT_REFERENCE_URL=http://127.0.0.1:30122 bash "$SERVE" "$@"
else
  bash "$SERVE" "$@"
fi
