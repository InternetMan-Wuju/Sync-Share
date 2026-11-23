#!/usr/bin/env bash
# EP32 three-arm repeated performance matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-/data_lib/cyt/ep32_perf_repeat_20261123}"
PORT="${PORT:-18023}"
WAIT="${WAIT:-900}"
MEM="${MEM:-0.85}"
CTX="${CTX:-65536}"
MAXPREF="${MAXPREF:-32768}"
CHUNKED="${CHUNKED:-8192}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REQ_COUNTS=(4 3 3)

wait_health() {
  local end=$((SECONDS + WAIT))
  while (( SECONDS < end )); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/health" || true)" == 200 ]]; then return 0; fi
    sleep 5
  done
  echo "health timeout after ${WAIT}s" >&2
  return 1
}

stop32() {
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" ./scripts/serve_ep32.sh stop >/dev/null 2>&1 || true)
}

ensure_inputs() {
  local source="${AGENTX_SOURCE:-/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json}"
  (cd "$ROOT" && python3 scripts/make_agentx_inputs.py --source "$source" --outdir inputs)
}

collect() {
  local run="$1" dest="$2"
  mkdir -p "$ROOT/logs/$dest/node23" "$ROOT/logs/$dest/node25"
  cp -a "$DEPLOY_ROOT/logs/ep32_${run}/node23/." "$ROOT/logs/$dest/node23/"
  scp -q -r -o BatchMode=yes "10.120.72.25:$DEPLOY_ROOT/logs/ep32_${run}/node25/." "$ROOT/logs/$dest/node25/"
}

start_arm() {
  local arm="$1" run="$2"
  stop32
  case "$arm" in
    original)
      (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM="$MEM" CTX="$CTX" MAXPREF="$MAXPREF" CHUNKED="$CHUNKED" \
        OBSERVE=0 ./scripts/serve_ep32.sh start off "$run") ;;
    eplb)
      (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM="$MEM" CTX="$CTX" MAXPREF="$MAXPREF" CHUNKED="$CHUNKED" \
        OBSERVE=0 ./scripts/serve_ep32.sh start eplb_online "$run") ;;
    k2)
      (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM="$MEM" CTX="$CTX" MAXPREF="$MAXPREF" CHUNKED="$CHUNKED" \
        OBSERVE=0 KCONFIG="$ROOT/control/k2_ep32_perf.json" ./scripts/serve_ep32.sh start k2 "$run") ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
}

req() {
  local arm="$1" phase="$2" round="$3" repeat="$4" size="$5" input="$6" out="$7" maxnew="${8:-2}"
  python3 "$ROOT/scripts/bench_request.py" \
    --url "http://127.0.0.1:${PORT}" --input "$input" --out "$out" \
    --arm "$arm" --phase "$phase" --round "$round" --repeat "$repeat" --size "$size" --max-new-tokens "$maxnew"
}

wait_eplb_after_calibration() {
  local run="$1" log="$DEPLOY_ROOT/logs/ep32_${run}/node23/server.log"
  local before after
  before=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
  req eplb calibration 0 0 12288 "$ROOT/inputs/agentx_12288_test.json" "$ROOT/artifacts/raw/calibration/${run}_12288_cal0.json" 2
  for i in $(seq 1 90); do
    after=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
    if (( after > before )); then return 0; fi
    sleep 2
  done
  # One more calibration request is allowed if the first request did not cross a window.
  req eplb calibration 0 1 12288 "$ROOT/inputs/agentx_12288_test.json" "$ROOT/artifacts/raw/calibration/${run}_12288_cal1.json" 2
  for i in $(seq 1 90); do
    after=$(grep -c '\[EPLBManager\] rebalance end' "$log" 2>/dev/null || true)
    if (( after > before )); then return 0; fi
    sleep 2
  done
  echo "EPLB calibration did not produce a new rebalance end in $log" >&2
  return 1
}

run_arm() {
  local round="$1" arm="$2" count="${REQ_COUNTS[$((round-1))]}"
  local run="perf_${RUN_ID}_r${round}_${arm}"
  start_arm "$arm" "$run"
  wait_health
  if [[ "$arm" == eplb ]]; then wait_eplb_after_calibration "$run"; fi
  mkdir -p "$ROOT/artifacts/raw/warmup" "$ROOT/artifacts/raw/formal"
  for size in 8192 16384 32768; do
    req "$arm" warmup "$round" 0 "$size" "$ROOT/inputs/agentx_${size}.json" \
      "$ROOT/artifacts/raw/warmup/r${round}_${arm}_${size}.json" 2
    for i in $(seq 1 "$count"); do
      req "$arm" formal "$round" "$i" "$size" "$ROOT/inputs/agentx_${size}.json" \
        "$ROOT/artifacts/raw/formal/r${round}_${arm}_${size}_${i}.json" 2
    done
  done
  collect "$run" "r${round}_${arm}"
  stop32
}

validate_k2() {
  local run="perf_${RUN_ID}_validation"
  stop32
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM="$MEM" CTX="$CTX" MAXPREF="$MAXPREF" CHUNKED="$CHUNKED" \
    OBSERVE=0 KCONFIG="$ROOT/control/k2_ep32.json" ./scripts/serve_ep32.sh start k2 "$run")
  wait_health
  req k2 validation 0 0 8192 "$ROOT/inputs/agentx_8192.json" "$ROOT/artifacts/raw/validation/k2_native_8k.json" 2
  collect "$run" k2_validation
  stop32
  (cd "$ROOT" && python3 scripts/analyze_k2.py --logs logs/k2_validation --out artifacts/VALIDATION_K2_NATIVE_TASK_ACCOUNTING.json --ep 32 --main-per-rank 8 --owners 0,1 --hots 163,133)
}

analyze() {
  (cd "$ROOT" && python3 scripts/analyze_eplb_calibration.py --out artifacts/EPLB_CALIBRATION_EVIDENCE.json)
  (cd "$ROOT" && python3 scripts/analyze_perf.py)
}

cleanup() { stop32; }
trap cleanup EXIT

case "${1:-all}" in
  all)
    ensure_inputs
    for round in 1 2 3; do
      case "$round" in
        1) order=(original eplb k2) ;;
        2) order=(eplb k2 original) ;;
        3) order=(k2 original eplb) ;;
      esac
      echo "=== round $round: ${order[*]} ==="
      for arm in "${order[@]}"; do run_arm "$round" "$arm"; done
    done
    validate_k2
    analyze ;;
  round1) for arm in original eplb k2; do run_arm 1 "$arm"; done ;;
  round2) for arm in eplb k2 original; do run_arm 2 "$arm"; done ;;
  round3) for arm in k2 original eplb; do run_arm 3 "$arm"; done ;;
  validate-k2) validate_k2 ;;
  analyze) analyze ;;
  inputs) ensure_inputs ;;
  cleanup) cleanup ;;
  *) echo "usage: $0 {all|inputs|round1|round2|round3|validate-k2|analyze|cleanup}" >&2; exit 2 ;;
esac
