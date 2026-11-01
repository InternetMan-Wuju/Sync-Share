#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${URL:-http://127.0.0.1:30000}"
LIMIT="${LIMIT:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"

python3 "${SCRIPT_DIR}/bench_dolly.py" \
  --url "${URL}" \
  --jsonl "${SCRIPT_DIR}/databricks-dolly-15k.jsonl" \
  --out "${SCRIPT_DIR}/dolly_bench.json" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

echo "===== done ====="
echo "结果文件: ${SCRIPT_DIR}/dolly_bench.json"
