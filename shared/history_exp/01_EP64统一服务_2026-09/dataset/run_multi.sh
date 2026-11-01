#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

URL="${URL:-http://127.0.0.1:30000}"
DURATION="${DURATION:-3600}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SAMPLE="${SAMPLE:-0}"
STRATIFY="${STRATIFY:-category}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"
RUN_NAME="${RUN_NAME:-multi_$(date +%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/multi_runs/${RUN_NAME}}"

mkdir -p "${OUT_DIR}"

# 接续顺序：先跑已有 dolly，再跑中文 dolly，最后跑英文 alpaca-gpt4。
FILES=(
  "${SCRIPT_DIR}/databricks-dolly-15k.jsonl"
  "${SCRIPT_DIR}/KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl"
  "${SCRIPT_DIR}/alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl"
)

ARGS=(
  --url "${URL}"
  --jsonl "${FILES[@]}"
  --out "${OUT_DIR}/multi_bench.json"
  --progress "${OUT_DIR}/multi_bench.progress.jsonl"
  --duration "${DURATION}"
  --concurrency "${CONCURRENCY}"
  --max-tokens "${MAX_TOKENS}"
)

if [ "${SAMPLE}" -gt 0 ]; then
  ARGS+=(--sample "${SAMPLE}" --seed "${SEED}" --stratify "${STRATIFY}")
fi

if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

echo "===== multi-bench ====="
echo "out dir: ${OUT_DIR}"
echo "duration: ${DURATION}s  sample: ${SAMPLE}  stratify: ${STRATIFY}"

python3 "${SCRIPT_DIR}/bench_multi.py" "${ARGS[@]}"
