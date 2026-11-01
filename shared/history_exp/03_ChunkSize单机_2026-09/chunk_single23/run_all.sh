#!/usr/bin/env bash
set -uo pipefail

BASE=/home/chengyetai/workspace/current_exp/chunk_single23
HOST=10.120.72.23
PORT=30080
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
REPS="${REPS:-2}"
[[ "$RUN_ID" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo 'expected run id YYYYMMDD_HHMMSS' >&2; exit 2; }
RUN_DIR="$BASE/results/$RUN_ID"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/orchestrator.log") 2>&1
printf '[%s] run_id=%s host=%s reps=%s\n' "$(date -Iseconds)" "$RUN_ID" "$HOST" "$REPS"
sha256sum "$BASE/serve.sh" "$BASE/bench.py" "$BASE/run_all.sh" "$BASE/patches/load_model_utils.py.orig" "$BASE/patches/load_model_utils.py" > "$RUN_DIR/code_sha256.txt"

remote() { ssh -T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR "$HOST" "$@"; }
wait_health() {
  local chunk="$1" start now state
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 4 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] chunk=$chunk health=200"
      return 0
    fi
    state=$(remote "sudo -n docker inspect chunk-single23-$chunk -f '{{.State.Status}}'" 2>/dev/null || true)
    if [[ "$state" != running ]]; then
      echo "[$(date -Iseconds)] chunk=$chunk container=$state"
      remote "sudo -n docker logs --tail 50 chunk-single23-$chunk" 2>&1 | tail -50
      return 1
    fi
    now=$(date +%s)
    if (( now - start > 5400 )); then
      echo "[$(date -Iseconds)] chunk=$chunk health timeout after 5400 s"
      remote "sudo -n docker logs --tail 50 chunk-single23-$chunk" 2>&1 | tail -50
      return 1
    fi
    if (( (now - start) % 60 < 15 )); then echo "[$(date -Iseconds)] chunk=$chunk loading, elapsed=$((now-start)) s"; fi
    sleep 15
  done
}
smoke_ok() {
  python3 - "$1" <<'PY'
import json, sys
p = sys.argv[1]
try:
    rows = [json.loads(line) for line in open(p, encoding='utf-8')]
except (OSError, json.JSONDecodeError) as e:
    raise SystemExit(f'smoke results unavailable: {e}')
long = next((row for row in reversed(rows) if row.get('role') == 'long_warm'), {})
cached, prompt = long.get('cached_tokens'), long.get('prompt_tokens')
print('smoke main case:', {k: long.get(k) for k in ('output_tokens','prompt_tokens','cached_tokens','error','ttft_s','tpot_ms')}, flush=True)
if long.get('error') or long.get('output_tokens') != 300:
    raise SystemExit('smoke failed: 300 output tokens not completed')
if not isinstance(cached, int) or not isinstance(prompt, int) or not .55 <= cached / prompt <= .65:
    raise SystemExit('smoke failed: measured prefix hit ratio is missing or outside 55-65%')
PY
}

for chunk in 8192 1024; do
  result="$RUN_DIR/chunk_$chunk"
  mkdir -p "$result"
  if ! remote "sudo -n docker inspect chunk-single23-$chunk >/dev/null 2>&1"; then
    echo "[$(date -Iseconds)] starting chunk=$chunk"
    if ! bash "$BASE/serve.sh" start "$chunk" "$RUN_ID"; then
      echo "[$(date -Iseconds)] start failed chunk=$chunk" | tee "$result/failed.txt"
      continue
    fi
  else
    echo "[$(date -Iseconds)] using already created chunk=$chunk"
  fi
  if ! wait_health "$chunk"; then
    echo "[$(date -Iseconds)] startup failed chunk=$chunk" | tee "$result/failed.txt"
    bash "$BASE/serve.sh" stop
    continue
  fi
  echo "[$(date -Iseconds)] smoke chunk=$chunk"
  if ! python3 "$BASE/bench.py" --out-dir "$result" --mode smoke 2>&1 | tee "$result/smoke.log"; then
    echo "[$(date -Iseconds)] smoke command failed chunk=$chunk" | tee "$result/failed.txt"
  elif ! smoke_ok "$result/smoke.jsonl"; then
    echo "[$(date -Iseconds)] main case cannot be verified chunk=$chunk" | tee "$result/failed.txt"
  else
    echo "[$(date -Iseconds)] full benchmark chunk=$chunk"
    if ! python3 "$BASE/bench.py" --out-dir "$result" --mode full --reps "$REPS" 2>&1 | tee "$result/full.log"; then
      echo "[$(date -Iseconds)] full benchmark failed chunk=$chunk" | tee "$result/failed.txt"
    fi
  fi
  if [[ "$chunk" == 8192 ]]; then
    echo "[$(date -Iseconds)] switching 8192 -> 1024"
    bash "$BASE/serve.sh" stop
  fi
done
echo "[$(date -Iseconds)] experiment ended; results=$RUN_DIR"
