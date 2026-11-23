#!/usr/bin/env bash
# Re-test only online EPLB +32 with a 1000-forward-pass rebalance period.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-/data_lib/cyt/ep32_eplb1000_20261123}"
PORT="${PORT:-18023}"
WAIT="${WAIT:-900}"
MEM="${MEM:-0.85}"
CTX="${CTX:-65536}"
MAXPREF="${MAXPREF:-32768}"
CHUNKED="${CHUNKED:-8192}"
REBAL_ITERS="${REBAL_ITERS:-1000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
COUNTS=(4 3 3)

wait_health() {
  local end=$((SECONDS + WAIT))
  while (( SECONDS < end )); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/health" || true)" == 200 ]]; then return 0; fi
    sleep 5
  done
  echo "health timeout" >&2; return 1
}
stop32() { (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" ./scripts/serve_ep32.sh stop >/dev/null 2>&1 || true); }
ensure_inputs() {
  local source="${AGENTX_SOURCE:-/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json}"
  (cd "$ROOT" && python3 scripts/make_agentx_inputs.py --source "$source" --outdir inputs)
  (cd "$ROOT" && python3 - <<'PY2'
import hashlib,json
from pathlib import Path
src=json.loads(Path('inputs/agentx_8192.json').read_text())
ids=src['input_ids'][:64]
out={'input_tokens':64,'input_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'source':'derived from agentx_8192 first 64 ids','materialization':'AgentX-derived calibration traffic; not original human text','input_ids':ids}
Path('inputs/agentx_64_cal.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
PY2
  )
}
req() {
  local phase="$1" round="$2" repeat="$3" size="$4" input="$5" out="$6"
  python3 "$ROOT/scripts/bench_request.py" --url "http://127.0.0.1:${PORT}" --input "$input" --out "$out" \
    --arm eplb --phase "$phase" --round "$round" --repeat "$repeat" --size "$size" --max-new-tokens 2
}
collect() {
  local run="$1" dest="$2"
  mkdir -p "$ROOT/logs/$dest/node23" "$ROOT/logs/$dest/node25"
  cp -a "$DEPLOY_ROOT/logs/ep32_${run}/node23/." "$ROOT/logs/$dest/node23/"
  scp -q -r -o BatchMode=yes "10.120.72.25:$DEPLOY_ROOT/logs/ep32_${run}/node25/." "$ROOT/logs/$dest/node25/"
}
run_round() {
  local round="$1"
  local count="${COUNTS[$((round-1))]}"
  local run="perf_eplb1000_r${round}"
  stop32
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM="$MEM" CTX="$CTX" MAXPREF="$MAXPREF" CHUNKED="$CHUNKED" \
    OBSERVE=0 REBAL_ITERS="$REBAL_ITERS" ./scripts/serve_ep32.sh start eplb_online "$run")
  wait_health
  local log="$DEPLOY_ROOT/logs/ep32_${run}/node23/server.log"
  local before after i=0
  before=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
  echo "round $round calibration begins with $before prior rebalance lines"
  for i in $(seq 1 1500); do
    req calibration "$round" "$i" 64 "$ROOT/inputs/agentx_64_cal.json" \
      "$ROOT/artifacts/raw/calibration/r${round}_cal_${i}.json" >/dev/null
    after=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
    if (( after > before )); then
      echo "round $round calibration reached new rebalance end after $i requests ($before -> $after)"
      break
    fi
  done
  after=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
  if (( after <= before )); then echo "round $round failed to trigger rebalance" >&2; return 1; fi
  mkdir -p "$ROOT/artifacts/raw/warmup" "$ROOT/artifacts/raw/formal"
  for size in 8192 16384 32768; do
    req warmup "$round" 0 "$size" "$ROOT/inputs/agentx_${size}.json" "$ROOT/artifacts/raw/warmup/r${round}_${size}.json"
    for i in $(seq 1 "$count"); do
      req formal "$round" "$i" "$size" "$ROOT/inputs/agentx_${size}.json" "$ROOT/artifacts/raw/formal/r${round}_${size}_${i}.json"
    done
  done
  collect "$run" "r${round}_eplb"
  stop32
}
analyze() {
  (cd "$ROOT" && python3 scripts/analyze_eplb1000.py)
}
cleanup() { stop32; }
trap cleanup EXIT
case "${1:-all}" in
  all) ensure_inputs; for r in 1 2 3; do run_round "$r"; done; analyze ;;
  round1) run_round 1 ;;
  round2) run_round 2 ;;
  round3) run_round 3 ;;
  inputs) ensure_inputs ;;
  analyze) analyze ;;
  cleanup) cleanup ;;
  *) echo "usage: $0 {all|round1|round2|round3|inputs|analyze|cleanup}" >&2; exit 2 ;;
esac
