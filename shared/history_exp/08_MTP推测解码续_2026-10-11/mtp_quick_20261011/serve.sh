#!/usr/bin/env bash
set -euo pipefail
# One-off single-node MTP test service (SPEC=on/off) for the AL quick check.
DIR=/home/chengyetai/workspace/current_exp/mtp_quick_20261011
IMAGE=sglang-glm52-pd:fe694986
IMAGE_ID=sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d
PORT="${PORT:-30080}"; CHUNK="${CHUNK:-16384}"; MEM="${MEM:-0.88}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
SPEC="${SPEC:-on}"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)
spec_flags() {
  if [[ "$SPEC" == on ]]; then
    echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
  fi
}
remote_start() {
  local node="$1" run_id="$2" devices=() i launch=""
  local logdir="$DIR/logs/$run_id/node_$node"; mkdir -p "$logdir"
  [[ -d "$MODEL" ]] || { echo "missing model $MODEL" >&2; return 1; }
  [[ "$(sudo -n docker image inspect "$IMAGE" -f '{{.Id}}')" == "$IMAGE_ID" ]] || { echo "image mismatch" >&2; return 1; }
  sudo -n docker rm -f "mtpq-$node" >/dev/null 2>&1 || true
  for i in $(seq 0 15); do devices+=(--device "/dev/davinci$i"); done
  devices+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  local ip="10.120.72.$node"
  launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export SGLANG_HOST_IP=$ip HCCL_IF_IP=$ip
    export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-650}
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 STREAMS_PER_DEVICE=32
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --tp-size 16 --ep-size 16 --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode normal \\
      --context-length 131072 --chunked-prefill-size $CHUNK \\
      --max-prefill-tokens 65536 --max-running-requests 4 --mem-fraction-static $MEM \\
      --disable-shared-experts-fusion --disable-cuda-graph \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 $(spec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sudo -n docker run -d --name "mtpq-$node" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=mtp_quick_20261011 --label spec="$SPEC" \
    "${devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$DIR/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMAGE" -lc "$launch"
  echo "started node=$node spec=$SPEC log=$logdir"
}
case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"; node="${3:-19}"
    ssh "${SSH[@]}" "10.120.72.$node" "mkdir -p '$DIR/logs/$run_id/node_$node'"
    scp -q -o LogLevel=ERROR "$DIR/load_model_utils.py" "$DIR/serve.sh" "10.120.72.$node:$DIR/"
    ssh "${SSH[@]}" "10.120.72.$node" "SPEC='$SPEC' MODEL='$MODEL' PORT='$PORT' CHUNK='$CHUNK' MEM='$MEM' HCCL_BUFFSIZE='${HCCL_BUFFSIZE:-650}' bash '$DIR/serve.sh' __start '$node' '$run_id'"
    echo "run_id=$run_id"
    ;;
  status) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker ps -a --filter name=mtpq-${2:-19} --format '{{.Names}} {{.Status}}'; curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 4 http://127.0.0.1:$PORT/health" ;;
  stop) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker rm -f mtpq-${2:-19}" >/dev/null 2>&1 || true ;;
  __start) remote_start "$2" "$3" ;;
  *) echo "usage: serve.sh {start <run_id> <node>|status <node>|stop <node>}" >&2; exit 2 ;;
esac
