#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-18023}"
WAIT_SECS="${WAIT_SECS:-900}"
stop_mode(){ sudo -n docker rm -f "cyt-threeway-ep16-$1" >/dev/null 2>&1 || true; }
wait_health(){
  local mode="$1" deadline=$((SECONDS+WAIT_SECS))
  while (( SECONDS < deadline )); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" || true)" == 200 ]]; then return 0; fi
    if ! sudo -n docker ps --filter "name=^/cyt-threeway-ep16-$mode$" --format '{{.Names}}' | grep -q .; then
      echo "service $mode exited before health" >&2; return 1
    fi
    sleep 5
  done
  echo "health timeout for $mode" >&2; return 1
}
observe(){
  stop_mode off
  (cd "$ROOT" && OBSERVE=1 MEM_FRACTION=0.88 CTX=65536 MAX_PREFILL=32768 CHUNKED=8192 ./scripts/serve_ep16.sh off observe)
  wait_health off
  for size in 8k 16k 32k; do (cd "$ROOT" && python3 scripts/request_ttft.py --url "http://127.0.0.1:$PORT" --prompt "prompts/prompt_${size}.txt" --out "artifacts/off_${size}_observe.json" --warmups 0 --reps 1 --max-tokens 32); done
  stop_mode off
  export OBS="$ROOT/logs/off_observe/observer"
  python3 "$ROOT/scripts/merge_calibration.py"
}
layout(){ (cd "$ROOT/scripts" && python3 make_layout.py --routes ../artifacts/calibration/layer49_8k_all_ranks.jsonl --out ../layouts/eplb16_static.json --summary ../artifacts/calibration/eplb16_layer49_summary.json --ep 16 --redundant 16 --layer 49); }
replay_all(){ for size in 8k 16k 32k; do (cd "$ROOT" && ./scripts/run_replay16.sh "$size" 10 "final_$size"); done; }
service_off(){
  stop_mode off
  (cd "$ROOT" && OBSERVE=0 MEM_FRACTION=0.88 CTX=65536 MAX_PREFILL=32768 CHUNKED=8192 ./scripts/serve_ep16.sh off formal_ttft)
  wait_health off
  for size in 8k 16k 32k; do (cd "$ROOT" && python3 scripts/request_ttft.py --url "http://127.0.0.1:$PORT" --prompt "prompts/prompt_${size}.txt" --out "artifacts/off_${size}_ttft_10.json" --warmups 1 --reps 10 --max-tokens 32); done
  stop_mode off
}
service_eplb(){
  stop_mode eplb_static
  (cd "$ROOT" && OBSERVE=0 MEM_FRACTION=0.88 CTX=65536 MAX_PREFILL=32768 CHUNKED=8192 LAYOUT="$ROOT/layouts/eplb16_static.json" ./scripts/serve_ep16.sh eplb_static formal_ttft)
  wait_health eplb_static
  for size in 8k 16k; do (cd "$ROOT" && python3 scripts/request_ttft.py --url "http://127.0.0.1:$PORT" --prompt "prompts/prompt_${size}.txt" --out "artifacts/eplb_static_${size}_ttft.json" --warmups 1 --reps 10 --max-tokens 32); done
  (cd "$ROOT" && python3 scripts/request_ttft.py --url "http://127.0.0.1:$PORT" --prompt prompts/prompt_32k.txt --out artifacts/eplb_static_32k_ttft.json --warmups 0 --reps 1 --max-tokens 32 || true)
  stop_mode eplb_static
}
k1_blocker(){
  stop_mode k1_invalid
  (cd "$ROOT" && ./scripts/serve_ep16.sh k1_invalid k1_invalid_probe)
  sleep 25
  if ! grep -q 'num_physical_experts % ep_size == 0' "$ROOT/logs/k1_invalid_k1_invalid_probe/server.log"; then echo 'expected K1 uniform-layout assertion missing' >&2; stop_mode k1_invalid; return 1; fi
  stop_mode k1_invalid
}
online_rebalance_failure(){
  stop_mode eplb_online
  (cd "$ROOT" && OBSERVE=0 MEM_FRACTION=0.88 CTX=65536 MAX_PREFILL=32768 CHUNKED=8192 ./scripts/serve_ep16.sh eplb_online online_failure)
  wait_health eplb_online
  (cd "$ROOT" && python3 scripts/request_ttft.py --url "http://127.0.0.1:$PORT" --prompt prompts/prompt_8k.txt --out artifacts/eplb_online_8k_attempt.json --warmups 0 --reps 1 --max-tokens 32 || true)
  sleep 20; stop_mode eplb_online
}
analyze(){ python3 "$ROOT/scripts/analyze.py" > "$ROOT/artifacts/analysis_stdout.json"; python3 "$ROOT/scripts/validate.py"; }
case "${1:-all}" in
  observe) observe ;;
  layout) layout ;;
  replay) replay_all ;;
  service-off) service_off ;;
  service-eplb) service_eplb ;;
  k1-blocker) k1_blocker ;;
  online-rebalance-failure) online_rebalance_failure ;;
  analyze) analyze ;;
  all) observe; layout; replay_all; service_off; service_eplb; k1_blocker; analyze ;;
  *) echo "usage: $0 {observe|layout|replay|service-off|service-eplb|k1-blocker|online-rebalance-failure|analyze|all}" >&2; exit 2 ;;
esac
