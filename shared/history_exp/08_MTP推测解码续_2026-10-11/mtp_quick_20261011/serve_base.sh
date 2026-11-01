#!/usr/bin/env bash
set -euo pipefail
# Faithful reproduction of the official 910C leaderboard default config
# (AboutModels.md / the live glm52-united container on npu1-25), plus an
# optional MTP (NEXTN) switch -- the ONLY added variable.
DIR=/home/chengyetai/workspace/current_exp/mtp_quick_20261011
IMAGE=quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-8000}"
SPEC="${SPEC:-on}"
NAME="base-glm52-$1"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)

spec_flags() {
  [[ "${SPEC:-on}" == on ]] || return 0
  echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
}

remote_start() {
  local node="$1" run_id="$2" devices=() i launch="" name="base-glm52-$1"
  local logdir="$DIR/logs_base/$run_id/node_$node"; mkdir -p "$logdir"
  [[ -d "$MODEL" ]] || { echo "missing model $MODEL" >&2; return 1; }
  [[ -f "$DIR/base_v0516/load_model_utils.py" ]] || { echo "missing patched loader" >&2; return 1; }
  sudo -n docker rm -f "$name" >/dev/null 2>&1 || true
  for i in $(seq 0 15); do devices+=(--device "/dev/davinci$i"); done
  devices+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export HCCL_IF_BASE_PORT=50000
    export HCCL_BUFFSIZE=650
    export DEEPEP_MODE=auto
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1
    export STREAMS_PER_DEVICE=32
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --model-path /models/glm-q --attention-backend ascend --device npu \\
      --tp-size 16 --dp-size 2 --ep-size 16 --nnodes 1 --node-rank 0 \\
      --chunked-prefill-size 16384 --max-prefill-tokens 32768 --context-length 180000 \\
      --trust-remote-code --quantization modelslim --dtype bfloat16 \\
      --host 0.0.0.0 --port $PORT --mem-fraction-static 0.9 \\
      --served-model-name glm-5.2 --max-running-requests 64 --watchdog-timeout 9000 \\
      --tokenizer-worker-num 4 --enable-metrics --enable-metrics-for-all-schedulers \\
      --enable-cache-report --disable-shared-experts-fusion --enable-dp-attention \\
      --moe-a2a-backend deepep --deepep-mode auto \\
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 32 \\
      $(spec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sudo -n docker run -d --name "$name" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=mtp_base_v0516 --label spec="$SPEC" \
    "${devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$DIR/base_v0516/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMAGE" -lc "$launch"
  echo "started node=$node spec=$SPEC image=$IMAGE name=$name log=$logdir"
}

case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"; node="${3:-19}"
    ssh "${SSH[@]}" "10.120.72.$node" "mkdir -p '$DIR/logs_base/$run_id/node_$node' '$DIR/base_v0516'"
    scp -q -o LogLevel=ERROR "$DIR/base_v0516/load_model_utils.py" "$DIR/serve_base.sh" "10.120.72.$node:$DIR/" 2>/dev/null || true
    ssh "${SSH[@]}" "10.120.72.$node" "mkdir -p '$DIR/base_v0516'"
    scp -q -o LogLevel=ERROR "$DIR/base_v0516/load_model_utils.py" "10.120.72.$node:$DIR/base_v0516/"
    ssh "${SSH[@]}" "10.120.72.$node" "SPEC='$SPEC' MODEL='$MODEL' PORT='$PORT' bash '$DIR/serve_base.sh' __start '$node' '$run_id'"
    echo "run_id=$run_id"
    ;;
  status) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker ps -a --filter name=base-glm52-${2:-19} --format '{{.Names}} {{.Status}}'; curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 4 http://127.0.0.1:${PORT}/health" ;;
  stop) ssh "${SSH[@]}" "10.120.72.${2:-19}" "sudo -n docker rm -f base-glm52-${2:-19}" >/dev/null 2>&1 || true ;;
  __start) remote_start "$2" "$3" ;;
  *) echo "usage: serve_base.sh {start <run_id> <node>|status <node>|stop <node>}" >&2; exit 2 ;;
esac
