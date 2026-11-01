#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/../dataset" && pwd)"

URL="${URL:-http://127.0.0.1:30000}"
CONTAINER=glm52-ep64-decode-chengyetai
NODES="${NODES:-10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32}"
PORT="${PORT:-30000}"

LIMIT="${LIMIT:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUN_NAME="${RUN_NAME:-dolly_ep64_$(date +%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${DATASET_DIR}/ep_runs/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"

echo "===== run dir ====="
echo "${RUN_DIR}"

echo "===== 0. health ====="
curl -s -m 5 -o /dev/null -w 'health_http=%{http_code}\n' "${URL}/health"

echo "===== 1. start expert distribution record ====="
curl -s -X POST "${URL}/start_expert_distribution_record"
echo

echo "===== 2. run Dolly benchmark ====="
python3 "${DATASET_DIR}/bench_dolly.py" \
  --url "${URL}" \
  --jsonl "${DATASET_DIR}/databricks-dolly-15k.jsonl" \
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
  sudo docker cp "${SCRIPT_DIR}/analyze_expert_distribution.py" "${CONTAINER}:/tmp/analyze_expert_distribution.py"
  sudo docker exec "${CONTAINER}" python3 /tmp/analyze_expert_distribution.py \
    --pt "${NEWEST}" \
    --out /tmp/expert_distribution.json
  sudo docker cp "${CONTAINER}:/tmp/expert_distribution.json" "${RUN_DIR}/expert_distribution.json"
  echo "[dump] expert_distribution.pt -> ${RUN_DIR}/"
else
  echo "[dump] WARNING: no expert_distribution_recorder_*.pt found"
fi

echo "===== 5. save Prometheus metrics ====="
for ip in ${NODES}; do
  curl -s -m 10 "http://${ip}:${PORT}/metrics" -o "${RUN_DIR}/metrics_${ip}.prom"
  echo "[metrics] -> ${RUN_DIR}/metrics_${ip}.prom"
done

echo "===== 6. save npu-smi snapshot ====="
for ip in ${NODES}; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${ip}" "npu-smi info" \
    > "${RUN_DIR}/npusmi_${ip}.txt" 2>&1 || true
  echo "[npu-smi] -> ${RUN_DIR}/npusmi_${ip}.txt"
done

echo "===== 7. summarize run ====="
python3 "${SCRIPT_DIR}/summarize_dolly_run.py" \
  --run-dir "${RUN_DIR}"

echo "===== done ====="
find "${RUN_DIR}" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
