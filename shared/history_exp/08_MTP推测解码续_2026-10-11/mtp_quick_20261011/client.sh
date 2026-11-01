#!/usr/bin/env bash
set -euo pipefail
# Dedicated bench-client container (separate node from the served model).
DIR=/home/chengyetai/workspace/current_exp/mtp_quick_20261011
IMAGE=sglang-glm52-pd:fe694986
NODE="${NODE:-23}"
NAME="mtpclient-$NODE"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)

remote_up() {
  sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
  local dev=() i
  for i in $(seq 0 15); do dev+=(--device "/dev/davinci$i"); done
  dev+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  sudo -n docker run -d --name "$NAME" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=mtp_quick_client \
    "${dev[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /data_lib/data/models/glm-q:/models/glm-q:ro \
    -v "$DIR/client_logs:/exp-logs" \
    --entrypoint bash "$IMAGE" -lc 'sleep infinity'
  echo "client container up on $(hostname)"
}
case "${1:-}" in
  up)
    ssh "${SSH[@]}" "10.120.72.$NODE" 'mkdir -p '"$DIR"'/client_logs'
    ssh "${SSH[@]}" "10.120.72.$NODE" "bash -s -- __up" < "$0"
    ;;
  down) ssh "${SSH[@]}" "10.120.72.$NODE" "sudo -n docker rm -f $NAME" >/dev/null 2>&1 || true; echo down ;;
  __up) remote_up ;;
  *) echo "usage: NODE=23 client.sh {up|down}" >&2; exit 2 ;;
esac
