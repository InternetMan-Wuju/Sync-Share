#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/../dataset" && pwd)"

URL="${URL:-http://127.0.0.1:30000}"
CONTAINER=glm52-ep64-decode-chengyetai
NODES="${NODES:-10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32}"
PORT="${PORT:-30000}"
EP_SIZE="${EP_SIZE:-64}"

DURATION="${DURATION:-1800}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SAMPLE="${SAMPLE:-2000}"
STRATIFY="${STRATIFY:-dataset}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"
RUN_NAME="${RUN_NAME:-multi_ep_$(date +%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${DATASET_DIR}/ep_runs/${RUN_NAME}}"

FILES=(
  "${DATASET_DIR}/databricks-dolly-15k.jsonl"
  "${DATASET_DIR}/KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl"
  "${DATASET_DIR}/alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl"
)

mkdir -p "${RUN_DIR}"

echo "===== run dir ====="
echo "${RUN_DIR}"

echo "===== 0. health ====="
curl -s -m 5 -o /dev/null -w 'health_http=%{http_code}\n' "${URL}/health"

echo "===== 1. start expert distribution record ====="
curl -s -X POST "${URL}/start_expert_distribution_record"
echo

echo "===== 2. run mixed benchmark ====="
ARGS=(
  --url "${URL}"
  --jsonl "${FILES[@]}"
  --out "${RUN_DIR}/multi_bench.json"
  --progress "${RUN_DIR}/multi_bench.progress.jsonl"
  --duration "${DURATION}"
  --concurrency "${CONCURRENCY}"
  --max-tokens "${MAX_TOKENS}"
  --sample "${SAMPLE}"
  --stratify "${STRATIFY}"
  --seed "${SEED}"
)
if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi
python3 "${DATASET_DIR}/bench_multi.py" "${ARGS[@]}"

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

echo "===== 4b. compute rank load from global logical_count ====="
if [ -n "${NEWEST}" ]; then
  sudo docker cp "${SCRIPT_DIR}/analyze_rank_load_global.py" \
    "${CONTAINER}:/tmp/analyze_rank_load_global.py"
  sudo docker exec "${CONTAINER}" python3 /tmp/analyze_rank_load_global.py \
    --pt "${NEWEST}" \
    --ep-size "${EP_SIZE}" \
    --out /tmp/rank_load_global.json || true
  sudo docker cp "${CONTAINER}:/tmp/rank_load_global.json" \
    "${RUN_DIR}/rank_load_global.json" 2>/dev/null || true
  echo "[rank-load] -> ${RUN_DIR}/rank_load_global.json"
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

echo "===== 7. dataset token stats ====="
python3 "${SCRIPT_DIR}/dataset_stats.py" \
  --in "${RUN_DIR}/multi_bench.json" \
  --out "${RUN_DIR}/dataset_stats.json"

echo "===== 8. save server log + rebalance timing ====="
sudo docker logs "${CONTAINER}" > "${RUN_DIR}/server.log" 2>&1 || true
python3 "${SCRIPT_DIR}/parse_rebalance_timing.py" \
  --log "${RUN_DIR}/server.log" \
  --out "${RUN_DIR}/rebalance_timing.json" || true

echo "===== done ====="
find "${RUN_DIR}" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
