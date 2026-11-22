#!/usr/bin/env bash
# Reproduction entry point for the 2026-11-23 native-dispatch service run.
# The service stages require the two-node Ascend environment described in
# MANIFEST.json. Generated logs and large inputs are intentionally untracked.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-18023}"
WAIT="${WAIT:-900}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

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

stop_ep16() {
  (cd "$ROOT" && ./scripts/serve_ep16.sh stop >/dev/null 2>&1 || true)
}

stop_ep32() {
  (cd "$ROOT" && ./scripts/serve_ep32.sh stop >/dev/null 2>&1 || true)
}

inputs() {
  local source="${AGENTX_SOURCE:-/data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json}"
  (cd "$ROOT" && python3 scripts/make_agentx_inputs.py --source "$source" --outdir inputs)
}

run_ep16() {
  local mode="$1" cfg="$2" name="$3" input="$4" out="$5"
  stop_ep16
  (cd "$ROOT" && \
    KCONFIG="$ROOT/control/$cfg" MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 CHUNKED=8192 \
    ./scripts/serve_ep16.sh "$mode" "${RUN_TAG}_${name}")
  wait_health
  (cd "$ROOT" && python3 scripts/request_agentx.py \
    --url "http://127.0.0.1:${PORT}" --input "$input" --out "$out" --max-new-tokens 2)
  stop_ep16
}

run_ep16_prompt() {
  local mode="$1" cfg="$2" name="$3" prompt="$4" out="$5"
  stop_ep16
  (cd "$ROOT" && \
    KCONFIG="$ROOT/control/$cfg" MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 CHUNKED=8192 \
    ./scripts/serve_ep16.sh "$mode" "${RUN_TAG}_${name}")
  wait_health
  (cd "$ROOT" && python3 scripts/request_service.py \
    --url "http://127.0.0.1:${PORT}" --prompt "$prompt" --out "$out" --max-tokens 2)
  stop_ep16
}

run32_arm() {
  local arm="$1" mode="$2" outprefix="$3"
  local kconfig="${4:-}"
  local layout="${5:-}"
  stop_ep32
  local env_args=()
  [[ -n "$kconfig" ]] && env_args+=("KCONFIG=$kconfig")
  [[ -n "$layout" ]] && env_args+=("LAYOUT=$layout")
  (cd "$ROOT" && env "${env_args[@]}" ./scripts/serve_ep32.sh start "$mode" "${RUN_TAG}_${arm}")
  wait_health
  for size in 8192 16384 32768; do
    (cd "$ROOT" && python3 scripts/request_agentx.py \
      --url "http://127.0.0.1:${PORT}" --input "inputs/agentx_${size}.json" \
      --out "artifacts/${outprefix}_${size}.json" --max-new-tokens 2)
  done
  stop_ep32
}

run32_k2() {
  stop_ep32
  (cd "$ROOT" && KCONFIG="$ROOT/control/k2_ep32.json" \
    ./scripts/serve_ep32.sh start k2 "${RUN_TAG}_k2")
  wait_health
  (cd "$ROOT" && python3 scripts/request_agentx.py \
    --url "http://127.0.0.1:${PORT}" --input inputs/agentx_32768.json \
    --out artifacts/repair_ep32_k2_32768_isolated.json --max-new-tokens 2)
  for size in 8192 16384 32768; do
    (cd "$ROOT" && python3 scripts/request_agentx.py \
      --url "http://127.0.0.1:${PORT}" --input "inputs/agentx_${size}.json" \
      --out "artifacts/repair_ep32_k2_seq_${size}.json" --max-new-tokens 2)
  done
  stop_ep32
}

case "${1:-help}" in
  inputs) inputs ;;
  ep16-k0)
    run_ep16_prompt off k0.json k0 prompts/prompt_8k.txt artifacts/reproduce_ep16_k0.json ;;
  ep16-k1)
    run_ep16_prompt k1 k1_verify.json k1 prompts/prompt_8k.txt artifacts/reproduce_ep16_k1.json ;;
  ep16-k2)
    run_ep16_prompt k2 k2_verify.json k2 prompts/prompt_8k.txt artifacts/reproduce_ep16_k2.json ;;
  same-layer)
    run_ep16 k2 k2_l49.json same_layer inputs/agentx_12288_test.json artifacts/reproduce_same_layer.json ;;
  q0)
    run_ep16_prompt k2 k2_q0.json q0 prompts/prompt_8k.txt artifacts/reproduce_q0.json ;;
  ep32-off)
    run32_arm off off reproduce_ep32_off ;;
  ep32-eplb-static)
    run32_arm eplb_static eplb_static reproduce_ep32_eplb_static \
      "" "$ROOT/layouts/eplb32_static.json" ;;
  ep32-k2)
    run32_k2 ;;
  ep32-eplb-online)
    # This stage is expected to expose the recorded native online-rebalance
    # blocker. Keep the service alive long enough to collect its failure log.
    stop_ep32
    (cd "$ROOT" && ./scripts/serve_ep32.sh start eplb_online "${RUN_TAG}_eplb_online")
    wait_health || true
    (cd "$ROOT" && python3 scripts/request_agentx.py \
      --url "http://127.0.0.1:${PORT}" --input inputs/agentx_8192.json \
      --out artifacts/reproduce_ep32_eplb_online_8192.json --max-new-tokens 2 || true)
    stop_ep32 ;;
  validate) (cd "$ROOT" && python3 scripts/validate_service.py) ;;
  status)
    (cd "$ROOT" && ./scripts/serve_ep32.sh status || true)
    sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cyt-(threeway|global-k)' || true ;;
  cleanup) stop_ep16; stop_ep32 ;;
  all)
    inputs
    run_ep16_prompt off k0.json k0 prompts/prompt_8k.txt artifacts/reproduce_ep16_k0.json
    run_ep16_prompt k1 k1_verify.json k1 prompts/prompt_8k.txt artifacts/reproduce_ep16_k1.json
    run_ep16_prompt k2 k2_verify.json k2 prompts/prompt_8k.txt artifacts/reproduce_ep16_k2.json
    run_ep16 k2 k2_l49.json same_layer inputs/agentx_12288_test.json artifacts/reproduce_same_layer.json
    run_ep16_prompt k2 k2_q0.json q0 prompts/prompt_8k.txt artifacts/reproduce_q0.json
    run32_arm off off reproduce_ep32_off
    run32_arm eplb_static eplb_static reproduce_ep32_eplb_static "" "$ROOT/layouts/eplb32_static.json"
    run32_k2
    (cd "$ROOT" && python3 scripts/validate_service.py) ;;
  *)
    cat >&2 <<EOF
usage: $0 {inputs|ep16-k0|ep16-k1|ep16-k2|same-layer|q0|ep32-off|ep32-eplb-static|ep32-k2|ep32-eplb-online|validate|status|cleanup|all}
EOF
    exit 2 ;;
esac
