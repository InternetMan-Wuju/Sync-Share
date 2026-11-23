#!/usr/bin/env bash
# Reproduction entry point; requires npu1-23 and npu1-25 and the image/model
# described in MANIFEST.json. Generated logs/inputs are intentionally untracked.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-18023}"
WAIT="${WAIT:-900}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/data_lib/cyt/ep32_native_reroute_20261123}"
LOG23="$DEPLOY_ROOT/logs"

wait_health() {
  local end=$((SECONDS + WAIT))
  while (( SECONDS < end )); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/health" || true)" == 200 ]]; then
      return 0
    fi
    sleep 5
  done
  echo "health timeout after ${WAIT}s" >&2
  return 1
}

stop16() { (cd "$ROOT" && ./scripts/serve_ep16.sh stop all >/dev/null 2>&1 || true); }
stop32() { (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" ./scripts/serve_ep32.sh stop >/dev/null 2>&1 || true); }

inputs() {
  local source="${AGENTX_SOURCE:-/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json}"
  (cd "$ROOT" && python3 scripts/make_agentx_inputs.py --source "$source" --outdir inputs)
}

run_ep16() {
  local tag="$1" cfg="$2" input="$3" out="$4" run_name="${5:-${RUN_TAG}_${tag}}"
  stop16
  (cd "$ROOT" && KCONFIG="$ROOT/control/$cfg" MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 CHUNKED=8192 \
    ./scripts/serve_ep16.sh k2 "$run_name")
  wait_health
  (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
    --input "$input" --out "$out" --max-new-tokens 2)
  stop16
}

collect_ep32() {
  local run="$1" dest="$2"
  mkdir -p "$ROOT/logs/$dest/node23" "$ROOT/logs/$dest/node25"
  cp -a "$LOG23/$run/node23/." "$ROOT/logs/$dest/node23/"
  scp -q -r -o BatchMode=yes "10.120.72.25:$LOG23/$run/node25/." "$ROOT/logs/$dest/node25/"
}

run_ep32_original() {
  stop32
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" MEM=0.85 CTX=65536 MAXPREF=32768 CHUNKED=8192 \
    ./scripts/serve_ep32.sh start off "${RUN_TAG}_ep32_original")
  wait_health
  for s in 8192 16384 32768; do
    (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
      --input "inputs/agentx_${s}.json" --out "artifacts/ep32_original_${s}.json" --max-new-tokens 2)
  done
  collect_ep32 "${RUN_TAG}_ep32_original" ep32_original
  stop32
}

run_ep32_eplb() {
  stop32
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" OBSERVE=1 MEM=0.85 CTX=65536 MAXPREF=32768 CHUNKED=8192 \
    ./scripts/serve_ep32.sh start eplb_online "${RUN_TAG}_ep32_eplb_online")
  wait_health
  (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
    --input inputs/agentx_12288_test.json --out artifacts/ep32_eplb_online_calibration_12288.json --max-new-tokens 2)
  # Wait until the patched updater has completed at least one real rebalance.
  for _ in $(seq 1 120); do
    grep -q '\[EPLBManager\] rebalance end' "$LOG23/${RUN_TAG}_ep32_eplb_online/node23/server.log" && break
    sleep 5
  done
  grep -q '\[EPLBManager\] rebalance end' "$LOG23/${RUN_TAG}_ep32_eplb_online/node23/server.log"
  for s in 8192 16384 32768; do
    (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
      --input "inputs/agentx_${s}.json" --out "artifacts/ep32_eplb_online_${s}.json" --max-new-tokens 2)
  done
  collect_ep32 "${RUN_TAG}_ep32_eplb_online" ep32_eplb_online
  stop32
}

run_ep32_k2() {
  stop32
  (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" KCONFIG="$ROOT/control/k2_ep32.json" MEM=0.85 CTX=65536 MAXPREF=32768 CHUNKED=8192 \
    ./scripts/serve_ep32.sh start k2 "${RUN_TAG}_ep32_k2_single")
  wait_health
  (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
    --input inputs/agentx_32768.json --out artifacts/ep32_k2_single_32768_isolated.json --max-new-tokens 2)
  for s in 8192 16384 32768; do
    (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:${PORT}" \
      --input "inputs/agentx_${s}.json" --out "artifacts/ep32_k2_single_seq_${s}.json" --max-new-tokens 2)
  done
  collect_ep32 "${RUN_TAG}_ep32_k2_single" ep32_k2_single
  stop32
}

analyze() {
  (cd "$ROOT" && python3 scripts/analyze_slots.py \
    --same-layer-logs logs/k2_ep16_k2_same_layer_verify/k_service \
    --cross-layer-logs logs/k2_ep16_k2_single_v2/k_service \
    --q0-logs logs/k2_ep16_k2_q0/k_service \
    --out artifacts/EP16_SLOT_CONSISTENCY.json)
  (cd "$ROOT" && python3 scripts/analyze_eplb.py \
    --node23-server logs/ep32_eplb_online/node23/server.log \
    --node25-server logs/ep32_eplb_online/node25/server.log \
    --node23-observer logs/ep32_eplb_online/node23/observer \
    --node25-observer logs/ep32_eplb_online/node25/observer \
    --out artifacts/EP32_EPLB_ONLINE_EVIDENCE.json)
  (cd "$ROOT" && python3 scripts/analyze_k2.py --logs logs/k1_ep16_k1_verify --out artifacts/EP16_K1_NATIVE_TASK_ACCOUNTING.json --ep 16 --main-per-rank 16 --owners 0 --hots 163)
  (cd "$ROOT" && python3 scripts/analyze_k2.py --logs logs/k2_ep16_k2_same_layer_verify --out artifacts/EP16_K2_NATIVE_TASK_ACCOUNTING.json --ep 16 --main-per-rank 16 --owners 0,1 --hots 163,133)
  (cd "$ROOT" && python3 scripts/analyze_k2.py --logs logs/ep32_k2_single --out artifacts/EP32_K2_NATIVE_TASK_ACCOUNTING.json --ep 32 --main-per-rank 8 --artifacts artifacts)
  (cd "$ROOT" && python3 scripts/summarize_results.py)
}

case "${1:-help}" in
  inputs) inputs ;;
  ep16-k1) run_ep16 ep16-k1 k1_verify.json inputs/agentx_8192.json artifacts/ep16_k1_verify_8192.json ep16_k1_verify ;;
  ep16-k2) run_ep16 ep16-k2 k2.json inputs/agentx_8192.json artifacts/ep16_k2_single_v2_8192.json ep16_k2_single_v2 ;;
  same-layer) run_ep16 same-layer k2_l49.json inputs/agentx_12288_test.json artifacts/ep16_k2_same_layer_verify.json ep16_k2_same_layer_verify ;;
  q0) run_ep16 q0 k2_q0.json inputs/agentx_8192.json artifacts/ep16_k2_q0_8192.json ep16_k2_q0 ;;
  ep32-original) run_ep32_original ;;
  ep32-eplb-online) run_ep32_eplb ;;
  ep32-k2) run_ep32_k2 ;;
  validate) analyze ;;
  status) (cd "$ROOT" && DEPLOY_ROOT="$DEPLOY_ROOT" ./scripts/serve_ep32.sh status || true); sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cyt-(threeway|global-k)' || true ;;
  cleanup) stop16; stop32 ;;
  all) inputs; run_ep16 ep16-k1 k1_verify.json inputs/agentx_8192.json artifacts/ep16_k1_verify_8192.json ep16_k1_verify; run_ep16 ep16-k2 k2.json inputs/agentx_8192.json artifacts/ep16_k2_single_v2_8192.json ep16_k2_single_v2; run_ep16 same-layer k2_l49.json inputs/agentx_12288_test.json artifacts/ep16_k2_same_layer_verify.json ep16_k2_same_layer_verify; run_ep16 q0 k2_q0.json inputs/agentx_8192.json artifacts/ep16_k2_q0_8192.json ep16_k2_q0; run_ep32_original; run_ep32_eplb; run_ep32_k2; analyze ;;
  *) cat >&2 <<EOF
usage: $0 {inputs|ep16-k1|ep16-k2|same-layer|q0|ep32-original|ep32-eplb-online|ep32-k2|validate|status|cleanup|all}
EOF
    exit 2 ;;
esac
