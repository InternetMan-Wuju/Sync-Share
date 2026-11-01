#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EP_SIZE="${EP_SIZE:-64}"
ENABLE_EPLB="${ENABLE_EPLB:-1}"
PATCH12="${PATCH12:-1}"
PATCH2="${PATCH2:-1}"
NZ_OFF="${NZ_OFF:-0}"
INSTRUMENT="${INSTRUMENT:-0}"
PREFIX_CACHE="${PREFIX_CACHE:-0}"
MTP="${MTP:-0}"

case "${EP_SIZE}" in
  64)
    HOSTS=(10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32)
    NNODES=4
    ;;
  32)
    HOSTS=(10.120.72.23 10.120.72.25)
    NNODES=2
    ;;
  *)
    echo "unsupported EP_SIZE=${EP_SIZE}" >&2
    exit 1
    ;;
esac

RANK=0
for H in "${HOSTS[@]}"; do
  echo "===== push and start ${H} rank=${RANK} ====="
  scp "${SCRIPT_DIR}/patches/expert_location_updater_patched.py" "${H}:/tmp/expert_location_updater_patched.py"
  scp "${SCRIPT_DIR}/patches/torch_npu_distributed_c10d_patched.py" "${H}:/tmp/torch_npu_distributed_c10d_patched.py"
  if [ "${INSTRUMENT}" = "1" ]; then
    scp "${SCRIPT_DIR}/patches/tokenizer_manager_instrumented.py" "${H}:/tmp/tokenizer_manager_instrumented.py"
  fi
  scp "${SCRIPT_DIR}/serve_glm52_ep64_decode.sh" "${H}:/tmp/serve_glm52_ep64_decode.sh"
  ssh -t "${H}" "EP_SIZE=${EP_SIZE} NNODES=${NNODES} ENABLE_EPLB=${ENABLE_EPLB} PATCH12=${PATCH12} PATCH2=${PATCH2} NZ_OFF=${NZ_OFF} INSTRUMENT=${INSTRUMENT} PREFIX_CACHE=${PREFIX_CACHE} MTP=${MTP} bash /tmp/serve_glm52_ep64_decode.sh ${RANK}"
  RANK=$((RANK + 1))
done

echo "===== all nodes submitted ====="
echo "查看状态:"
echo "  ssh 10.120.72.23 'sudo docker ps --filter name=glm52-ep64-decode-chengyetai --format \"{{.Names}} {{.Status}}\"'"
echo "健康检查:"
echo "  ssh 10.120.72.23 'curl -s -m 5 -o /dev/null -w \"%{http_code}\" http://127.0.0.1:30000/health'"
