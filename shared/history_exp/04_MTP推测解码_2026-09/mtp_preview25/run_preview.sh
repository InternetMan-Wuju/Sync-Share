#!/usr/bin/env bash
set -euo pipefail
BASE=/home/chengyetai/workspace/current_exp/mtp_preview25
RUN_ID="${1:-20260926_181500}"
HOST=10.120.72.25
URL=http://$HOST:30081
wait_healthy() {
  local i mode="$1" state
  for i in $(seq 1 300); do
    if [[ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$URL/health" || true)" == 200 ]]; then echo "$mode healthy"; return 0; fi
    state=$(ssh -T -o LogLevel=ERROR "$HOST" "sudo -n docker inspect -f '{{.State.Running}}' mtp-preview25-$mode" 2>/dev/null || true)
    if [[ "$state" != true ]]; then echo "$mode container exited or not found" >&2; return 1; fi
    sleep 10
  done
  echo "$mode health timeout" >&2; return 1
}
sync_log() {
  local mode="$1"
  ssh -T -o LogLevel=ERROR "$HOST" "test -d '$BASE/results/$RUN_ID/$mode'" || return 0
  rsync -az -e 'ssh -T -o LogLevel=ERROR' "$HOST:$BASE/results/$RUN_ID/$mode/" "$BASE/results/$RUN_ID/$mode/"
}
trap 'sync_log on || true; sync_log off || true' EXIT
wait_healthy on
python3 "$BASE/bench.py" --mode on --plan preview --out-dir "$BASE/results/$RUN_ID/on" --url "$URL" --reps 2
sync_log on
"$BASE/serve.sh" stop
"$BASE/serve.sh" start off "$RUN_ID"
wait_healthy off
python3 "$BASE/bench.py" --mode off --plan preview --out-dir "$BASE/results/$RUN_ID/off" --url "$URL" --reps 2
sync_log off
python3 "$BASE/analyze.py" "$BASE/results/$RUN_ID"
