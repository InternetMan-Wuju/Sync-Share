#!/bin/bash
set -euo pipefail

# 测试4 前缀缓存压测：三档并发 10/20/30，每档请求数 = 并发*4。
# 用法: URL=... [INPUT=16384 OUTPUT=1024 PREFIX_RATIO=0.9] bash run_test4_prefix.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${URL:-http://127.0.0.1:18623}"
INPUT="${INPUT:-16384}"
OUTPUT="${OUTPUT:-1024}"
PREFIX_RATIO="${PREFIX_RATIO:-0.9}"
RESULTS="${RESULTS:-${DIR}/results_prefix}"
mkdir -p "$RESULTS"

for CONC in 10 20 30; do
  REQ=$((CONC * 4))
  PREFIX_NUM=$((CONC / 2))
  OUT="$RESULTS/test4_in${INPUT}_conc${CONC}.json"
  echo "===== input=${INPUT} output=${OUTPUT} concurrency=${CONC} requests=${REQ} ====="
  python3 "$DIR/bench_pd_prefix.py" \
    --url "$URL" \
    --input-tokens "$INPUT" \
    --output-tokens "$OUTPUT" \
    --concurrency "$CONC" \
    --requests "$REQ" \
    --prefix-ratio "$PREFIX_RATIO" \
    --prefix-num "$PREFIX_NUM" \
    --out "$OUT"
done

echo "===== all done, results in ${RESULTS} ====="
