#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PORT="${PORT:-18023}"; WAIT="${WAIT:-900}"
wait_health(){ local end=$((SECONDS+WAIT)); while ((SECONDS<end)); do [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health || true)" == 200 ]] && return 0; sleep 5; done; echo 'health timeout' >&2; return 1; }
stop16(){ sudo -n docker rm -f "cyt-threeway-ep16-$1" >/dev/null 2>&1 || true; }
run16(){ local mode="$1" cfg="$2" out="$3"; stop16 "$mode"; (cd "$ROOT" && KCONFIG="$ROOT/control/$cfg" MEM_FRACTION=0.88 CTX=16384 MAX_PREFILL=8192 CHUNKED=8192 ./scripts/serve_ep16.sh "$mode" "repro_$mode"); wait_health; (cd "$ROOT" && python3 scripts/request_service.py --url "http://127.0.0.1:$PORT" --prompt prompts/prompt_8k.txt --out "$out" --max-tokens 2); stop16 "$mode"; }
run32(){ local mode="$1"; shift; (cd "$ROOT" && ./scripts/serve_ep32.sh start "$mode" "repro_$mode"); wait_health; for s in 8192 16384 32768; do (cd "$ROOT" && python3 scripts/request_agentx.py --url "http://127.0.0.1:$PORT" --input "inputs/agentx_${s}.json" --out "artifacts/repro_ep32_${mode}_${s}.json" --max-new-tokens 2 || true); done; (cd "$ROOT" && ./scripts/serve_ep32.sh stop); }
case "${1:-help}" in
 inputs) cd "$ROOT" && python3 scripts/make_agentx_inputs.py --source /data_lib/cyt/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json --outdir inputs ;;
 ep16-k0) run16 off k0.json artifacts/repro_k0_8k.json ;;
 ep16-k1) run16 k1 k1_verify.json artifacts/repro_k1_8k.json ;;
 ep16-k2) run16 k2 k2_verify.json artifacts/repro_k2_8k.json ;;
 ep32-off) run32 off ;;
 ep32-eplb-static) cd "$ROOT" && LAYOUT="$ROOT/layouts/eplb32_static.json" ./scripts/serve_ep32.sh stop >/dev/null 2>&1 || true; LAYOUT="$ROOT/layouts/eplb32_static.json" ./scripts/serve_ep32.sh start eplb_static repro_eplb_static; wait_health; for s in 8192 16384 32768; do python3 scripts/request_agentx.py --url "http://127.0.0.1:$PORT" --input "inputs/agentx_${s}.json" --out "artifacts/repro_ep32_eplb_static_${s}.json" --max-new-tokens 2; done; ./scripts/serve_ep32.sh stop ;;
 ep32-k2) cd "$ROOT" && KCONFIG="$ROOT/control/k2_ep32.json" ./scripts/serve_ep32.sh start k2 repro_k2; wait_health; for s in 8192 16384 32768; do python3 scripts/request_agentx.py --url "http://127.0.0.1:$PORT" --input "inputs/agentx_${s}.json" --out "artifacts/repro_ep32_k2_${s}.json" --max-new-tokens 2 || true; done; ./scripts/serve_ep32.sh stop ;;
 validate) cd "$ROOT" && python3 scripts/validate_service.py ;;
 status) cd "$ROOT" && ./scripts/serve_ep32.sh status; sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep cyt-threeway-ep16 || true ;;
 stop) for m in off k1 k2 eplb_static eplb_online; do stop16 "$m"; done; cd "$ROOT" && ./scripts/serve_ep32.sh stop ;;
 *) echo "usage: $0 {inputs|ep16-k0|ep16-k1|ep16-k2|ep32-off|ep32-eplb-static|ep32-k2|validate|status|stop}" >&2; exit 2 ;;
esac
