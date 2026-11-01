#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${URL:-http://127.0.0.1:30000}"
CONTAINER=glm52-ep64-decode-chengyetai
LIMIT="${LIMIT:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/ep_metrics}"

mkdir -p "${RUN_DIR}"

echo "===== 0. health ====="
curl -s -m 5 -o /dev/null -w 'health_http=%{http_code}\n' "${URL}/health"

echo "===== 1. start expert distribution record ====="
curl -s -X POST "${URL}/start_expert_distribution_record"
echo

echo "===== 2. run Dolly benchmark ====="
python3 "${SCRIPT_DIR}/bench_dolly.py" \
  --url "${URL}" \
  --jsonl "${SCRIPT_DIR}/databricks-dolly-15k.jsonl" \
  --out "${RUN_DIR}/dolly_bench.json" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

echo "===== 3. stop expert distribution record ====="
curl -s -X POST "${URL}/stop_expert_distribution_record"
echo

echo "===== 4. dump expert distribution ====="
curl -s -X POST "${URL}/dump_expert_distribution_record" >/dev/null
sleep 2

NEWEST=$(sudo docker exec "${CONTAINER}" bash -c "ls -t /tmp/expert_distribution_recorder_*.pt 2>/dev/null | head -1")
if [ -n "${NEWEST}" ]; then
  sudo docker cp "${CONTAINER}:${NEWEST}" "${RUN_DIR}/expert_distribution.pt"
  echo "[dump] ${NEWEST} -> ${RUN_DIR}/expert_distribution.pt"
else
  echo "[dump] WARNING: no expert_distribution_recorder_*.pt found in container /tmp"
fi

echo "===== 5. save Prometheus metrics ====="
curl -s "${URL}/metrics" -o "${RUN_DIR}/metrics.prom"
echo "[metrics] -> ${RUN_DIR}/metrics.prom"

echo "===== 6. save npu-smi snapshot ====="
npu-smi info > "${RUN_DIR}/npusmi_after.txt" 2>&1 || true
echo "[npu-smi] -> ${RUN_DIR}/npusmi_after.txt"

echo "===== done ====="
echo "结果目录: ${RUN_DIR}"
find "${RUN_DIR}" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
