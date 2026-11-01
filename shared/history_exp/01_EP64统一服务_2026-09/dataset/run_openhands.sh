#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER=glm52-ep64-decode-chengyetai
URL="${URL:-http://127.0.0.1:30000}"
LIMIT="${LIMIT:-32}"
MAX_TURNS="${MAX_TURNS:-20}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "===== 1. copy parquet into container ====="
sudo docker cp "${SCRIPT_DIR}/trajectories.parquet" "${CONTAINER}:/tmp/trajectories.parquet"

echo "===== 2. copy prepare script into container ====="
sudo docker cp "${SCRIPT_DIR}/prepare_openhands.py" "${CONTAINER}:/tmp/prepare_openhands.py"

echo "===== 3. convert first ${LIMIT} trajectories to JSON ====="
sudo docker exec "${CONTAINER}" python3 /tmp/prepare_openhands.py \
  --parquet /tmp/trajectories.parquet \
  --out /tmp/openhands_sessions.json \
  --limit "${LIMIT}" \
  --max-turns "${MAX_TURNS}"

echo "===== 4. copy JSON back to host ====="
sudo docker cp "${CONTAINER}:/tmp/openhands_sessions.json" "${SCRIPT_DIR}/openhands_sessions.json"

echo "===== 5. send requests to service ====="
python3 "${SCRIPT_DIR}/bench_openhands.py" \
  --url "${URL}" \
  --json "${SCRIPT_DIR}/openhands_sessions.json" \
  --tools "${SCRIPT_DIR}/tools.json" \
  --out "${SCRIPT_DIR}/openhands_bench.json" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

echo "===== done ====="
echo "结果文件: ${SCRIPT_DIR}/openhands_bench.json"
