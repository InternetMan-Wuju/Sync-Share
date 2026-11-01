#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${URL:-http://127.0.0.1:18623}"
INPUT="${INPUT:-16384}"
OUTPUT="${OUTPUT:-1024}"
PREFIX_RATIO="${PREFIX_RATIO:-0.9}"
RESULTS="${RESULTS:-${DIR}/results}"
mkdir -p "${RESULTS}"

for CONC in 10 20 30; do
  REQ=$((CONC * 4))
  OUT="${RESULTS}/test4_in${INPUT}_conc${CONC}.json"
  echo "===== run input=${INPUT} output=${OUTPUT} concurrency=${CONC} requests=${REQ} ====="
  python3 "${DIR}/bench_pd.py" \
    --url "${URL}" \
    --input-tokens "${INPUT}" \
    --output-tokens "${OUTPUT}" \
    --concurrency "${CONC}" \
    --requests "${REQ}" \
    --prefix-ratio "${PREFIX_RATIO}" \
    --out "${OUT}"
done

echo "===== all done, results in ${RESULTS} ====="
