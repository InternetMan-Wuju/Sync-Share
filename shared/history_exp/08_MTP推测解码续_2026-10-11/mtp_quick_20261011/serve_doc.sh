#!/usr/bin/env bash
set -euo pipefail
# Strictly per docs/AboutModels.md: only the documented parameters.
# NOTHING is copied from the glm52-united container on npu1-25.
DIR=/home/chengyetai/workspace/current_exp/mtp_quick_20261011
IMAGE=quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16
MODEL="${MODEL:-/data_lib/data/models/glm-q}"     # 文档写 /mnt/raid/...；19 上没有，按实际
PORT="${PORT:-8000}"
SPEC="${SPEC:-on}"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)
spec_flags() {
  [[ "${SPEC:-on}" == on ]] || return 0
  echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
}
remote_start() {
  local node="$1" run_id="$2" devices=() i launch="" name="doc-glm52-$1"
  local logdir="$DIR/logs_doc/$run_id/node_$node"; mkdir -p "$logdir"
  sudo -n docker rm -f "$name" >/dev/null 2>&1 || true
  for i in $(seq 0 15); do devices+=(--device "/dev/davinci$i"); done
  devices+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export HCCL_IF_BASE_PORT=50000
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --model-path /models/glm-q --served-model-name glm-5 \\
      --host 0.0.0.0 --port $PORT \\
      --tp-size 16 --dp-size 2 --ep-size 16 \\
      --enable-dp-attention --enable-dp-lm-head \\
      --moe-a2a-backend deepep --deepep-mode auto \\
      --mem-fraction-static 0.90 --context-length 180000 \\
      --chunked-prefill-size 16384 --max-prefill-tokens 32768 \\
      --max-running-requests 64 --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32 \\
      $(spec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sudo -n docker run -d --name "$name" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=doc_aboutmodels --label spec="$SPEC" \
    "${devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    --entrypoint bash "$IMAGE" -lc "$launch"
  echo "started node=$node spec=$SPEC name=$name log=$logdir"
}
case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"; node="${3:-19}"
    ssh "${SSH[@]}" "10.120.72.$node" "mkdir -p '$DIR/logs_doc/$run_id/node_$node'"
    scp -q -o LogLevel=ERROR "$DIR/serve_doc.sh" "10.120.72.$node:$DIR/"
    ssh "${SSH[@]}" "10.120.72.$node" "SPEC='$SPEC' MODEL='$MODEL' PORT='$PORT' bash '$DIR/serve_doc.sh' __start '$node' '$run_id'"
    echo "run_id=$run_id" ;;
  status) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker ps -a --filter name=doc-glm52-${2:-19} --format '{{.Names}} {{.Status}}'; curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 30 http://127.0.0.1:${PORT}/health" ;;
  stop) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker rm -f doc-glm52-${2:-19}" >/dev/null 2>&1 || true ;;
  __start) remote_start "$2" "$3" ;;
  *) echo "usage: serve_doc.sh {start <run_id> <node>|status <node>|stop <node>}" >&2; exit 2 ;;
esac
