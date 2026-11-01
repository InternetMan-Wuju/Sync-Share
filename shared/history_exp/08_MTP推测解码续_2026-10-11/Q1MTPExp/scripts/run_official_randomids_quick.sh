#!/usr/bin/env bash
# Short ON-only comparison; the reference scripts are never executed.
set -euo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
RUN_ID="${1:?run id required}"
OUT="$DIR/results/$RUN_ID"
LOGDIR="$DIR/logs/$RUN_ID"
mkdir -p "$OUT" "$LOGDIR"
exec >>"$OUT/run.log" 2>&1
export NAME=q1-mtp-23
owned=0
cleanup() {
  if [[ "$owned" = 1 ]]; then
    bash "$DIR/scripts/serve_mtp_q1.sh" stop
  fi
}
trap cleanup EXIT
export LOADER="" IMG=quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3
export MTP=on BACKEND=deepep DEEPEP=normal GRAPH=on MEM=0.90 MRR=8 MAXPREF=32768
echo "Starting $RUN_ID on node23, MTP ON, 1K/256 x2 per input type"
bash "$DIR/scripts/serve_mtp_q1.sh" start "$RUN_ID"
owned=1
ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if curl -fsS --max-time 5 http://127.0.0.1:30080/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  if [[ "$(sudo -n docker inspect "$NAME" --format '{{.State.Running}}')" != true ]]; then
    echo "Service exited; inspect $LOGDIR/server.log"
    exit 1
  fi
  sleep 10
done
[[ "$ready" = 1 ]] || { echo "Service health timed out"; exit 1; }
echo "health=200; exporting datasets with the installed official SGLang code"
sudo -n docker exec -i -e PYTHONDONTWRITEBYTECODE=1 -e HF_HOME=/exp-logs/hf-cache \
  "$NAME" python3 - /exp-logs/datasets < "$DIR/scripts/export_official_randomids.py"
mkdir -p "$OUT/datasets"
cp "$LOGDIR/datasets/"*.json "$OUT/datasets/"
python3 - "$DIR/datasets/random_1k256.json" "$OUT/datasets/uniform_legacy.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
d['name'] = 'uniform_legacy_subset_first2'
d['cases'] = d['cases'][:2]
d['n_requests'] = len(d['cases'])
d['source_dataset'] = sys.argv[1]
Path(sys.argv[2]).write_text(json.dumps(d, ensure_ascii=False, indent=1))
PY
for kind in native_exact_ids uniform_legacy native_default_text; do
  echo "Testing $kind: 2 requests, concurrency=1, output=256"
  timeout 600 python3 -u "$DIR/scripts/quick_accept_test.py" \
    --url http://127.0.0.1:30080 --dataset "$OUT/datasets/$kind.json" \
    --out "$OUT/$kind.json" --concurrency 1 --flush-cache --timeout 300
done
echo "COMPLETED $OUT"
