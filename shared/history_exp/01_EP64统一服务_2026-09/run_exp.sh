#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_ID="${1:?usage: $0 <exp_id 1..10>}"

DURATION="${DURATION:-1800}"
SAMPLE="${SAMPLE:-2000}"
STRATIFY="${STRATIFY:-dataset}"
SEED="${SEED:-0}"
CONCURRENCY="${CONCURRENCY:-64}"

case "${EXP_ID}" in
  1) EP_SIZE=64; ENABLE_EPLB=0; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  2) EP_SIZE=64; ENABLE_EPLB=1; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  3) EP_SIZE=32; ENABLE_EPLB=0; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  4) EP_SIZE=32; ENABLE_EPLB=1; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  5) EP_SIZE=64; ENABLE_EPLB=1; PATCH12=0; PATCH2=1; NZ_OFF=0 ;;
  6) EP_SIZE=32; ENABLE_EPLB=1; PATCH12=0; PATCH2=1; NZ_OFF=0 ;;
  7) EP_SIZE=64; ENABLE_EPLB=1; PATCH12=0; PATCH2=1; NZ_OFF=1 ;;
  8) EP_SIZE=32; ENABLE_EPLB=1; PATCH12=0; PATCH2=1; NZ_OFF=1 ;;
  9) EP_SIZE=64; ENABLE_EPLB=1; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  10) EP_SIZE=32; ENABLE_EPLB=1; PATCH12=1; PATCH2=1; NZ_OFF=0 ;;
  *) echo "unknown exp_id=${EXP_ID}" >&2; exit 1 ;;
esac

if [ "${EP_SIZE}" = "32" ]; then
  NODES="10.120.72.23 10.120.72.25"
else
  NODES="10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32"
fi

echo "===== run exp ${EXP_ID}: EP=${EP_SIZE} EPLB=${ENABLE_EPLB} patch12=${PATCH12} patch2=${PATCH2} nz_off=${NZ_OFF} ====="

# 停掉旧服务（4 台都尝试，没容器的会自动忽略）
bash "${SCRIPT_DIR}/stop_ep64_decode.sh" || true

# 启动
EP_SIZE="${EP_SIZE}" \
  ENABLE_EPLB="${ENABLE_EPLB}" \
  PATCH12="${PATCH12}" \
  PATCH2="${PATCH2}" \
  NZ_OFF="${NZ_OFF}" \
  bash "${SCRIPT_DIR}/start_ep64_decode.sh"

# 等健康检查 200
echo "waiting for health..."
HEALTHY=0
for i in $(seq 1 40); do
  CODE=$(ssh -o BatchMode=yes 10.120.72.23 \
    'curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health' 2>/dev/null || echo 000)
  if [ "${CODE}" = "200" ]; then
    HEALTHY=1
    echo "healthy after ${i} checks"
    break
  fi
  echo "  [${i}/40] health=${CODE}, waiting..."
  sleep 15
done

if [ "${HEALTHY}" != "1" ]; then
  echo "ERROR: service not healthy in time" >&2
  exit 1
fi

# 跑 Dolly 压测（同时收集性能和专家热度）
RUN_NAME="exp${EXP_ID}_ep${EP_SIZE}_eplb${ENABLE_EPLB}_p12_${PATCH12}_p2_${PATCH2}_nz${NZ_OFF}_$(date +%m%d_%H%M%S)"
  NODES="${NODES}" \
  RUN_NAME="${RUN_NAME}" \
  EP_SIZE="${EP_SIZE}" \
  DURATION="${DURATION}" \
  SAMPLE="${SAMPLE}" \
  STRATIFY="${STRATIFY}" \
  SEED="${SEED}" \
  CONCURRENCY="${CONCURRENCY}" \
  bash "${SCRIPT_DIR}/ep_lab/run_multi_ep_cell.sh"
