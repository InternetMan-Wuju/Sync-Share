#!/usr/bin/env bash
set -euo pipefail
BASE=/home/chengyetai/workspace/current_exp/chunk_manual_20260927
DEBUG="$BASE/semantic_debug_20260927/readonly_resume"
NODE="${1:-23}"
[[ "$NODE" == 23 || "$NODE" == 25 ]] || exit 2
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)
health_local() { curl -fsS --max-time 3 http://127.0.0.1:30122/health >/dev/null 2>&1; }
if ! ssh "${SSH[@]}" 10.120.72.31 'curl -fsS --max-time 3 http://127.0.0.1:30121/health >/dev/null 2>&1'; then
  ssh "${SSH[@]}" 10.120.72.31 "mkdir -p '$DEBUG/reference_snapshot'"
  scp -q "$DEBUG/reference_block_server.py" "10.120.72.31:$DEBUG/reference_block_server.py"
  rsync -az -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$BASE/semantic_debug_20260927/snapshot_node31/" "10.120.72.31:$DEBUG/reference_snapshot/"
  ssh "${SSH[@]}" 10.120.72.31 "nohup python3 '$DEBUG/reference_block_server.py' --snapshot '$DEBUG/reference_snapshot' --model /mnt/raid/user/data/models/glm-qq --port 30121 > '$DEBUG/reference_block_server.log' 2>&1 < /dev/null & echo \$! > '$DEBUG/reference_block_server.pid'"
fi
if ! health_local; then
  ssh -T -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o LogLevel=ERROR -L 127.0.0.1:30122:127.0.0.1:30121 10.120.72.31
fi
for attempt in $(seq 1 10); do
  if health_local; then break; fi
  sleep 1
done
health_local || { echo 'reference reader or SSH forwarding is unavailable' >&2; exit 1; }
if [[ "$NODE" == 25 ]]; then
  if ! ssh "${SSH[@]}" 10.120.72.25 'curl -fsS --max-time 3 http://127.0.0.1:30122/health >/dev/null 2>&1'; then
    ssh -T -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o LogLevel=ERROR -R 127.0.0.1:30122:127.0.0.1:30122 10.120.72.25
  fi
  ssh "${SSH[@]}" 10.120.72.25 'curl -fsS --max-time 10 http://127.0.0.1:30122/health'
else
  curl -fsS --max-time 10 http://127.0.0.1:30122/health
fi
