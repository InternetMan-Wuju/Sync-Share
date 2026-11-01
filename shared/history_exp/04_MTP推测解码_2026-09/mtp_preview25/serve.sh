#!/usr/bin/env bash
set -euo pipefail
BASE=/home/chengyetai/workspace/current_exp/mtp_preview25
HOST=10.120.72.25
IMAGE=sglang-glm52-pd:fe694986
IMAGE_ID=sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d
MODEL=/mnt/raid/user/data/models/glm-qq
PATCH=/home/chengyetai/workspace/current_exp/chunk_single23/patches/load_model_utils.py
PORT=30081
SSH_OPTS=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)
start_remote() {
  local mode="$1" run_id="$2" name="mtp-preview25-$1" logdir="$BASE/results/$2/$1" devices=() i spec_args='' launch
  [[ "$mode" == on || "$mode" == off ]] || exit 2
  [[ "$run_id" =~ ^[0-9]{8}_[0-9]{6}$ ]] || exit 2
  mkdir -p "$logdir"
  [[ -f "$PATCH" && -d "$MODEL" ]] || { echo 'missing patch or model'; exit 1; }
  [[ "$(sudo -n docker image inspect "$IMAGE" -f '{{.Id}}')" == "$IMAGE_ID" ]] || exit 1
  if sudo -n docker inspect "$name" >/dev/null 2>&1; then echo "container $name already exists" >&2; exit 1; fi
  if ss -lnt | grep -Eq ":${PORT}[[:space:]]"; then echo "port $PORT in use" >&2; exit 1; fi
  if [[ "$mode" == on ]]; then
    spec_args='--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4'
  fi
  for i in $(seq 0 15); do devices+=(--device "/dev/davinci$i"); done
  devices+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1
    export SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export SGLANG_HOST_IP=$HOST HCCL_IF_IP=$HOST
    export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=650
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 STREAMS_PER_DEVICE=32
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --tp-size 16 --ep-size 16 --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode normal \\
      --context-length 131072 --chunked-prefill-size 8192 \\
      --max-prefill-tokens 65536 --max-running-requests 4 --mem-fraction-static 0.88 \\
      --disable-shared-experts-fusion --disable-cuda-graph \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      $spec_args 2>&1 | tee /exp-logs/server.log"
  sudo -n docker run -d --name "$name" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label experiment=mtp_preview25 --label mode="$mode" \
    "${devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$PATCH:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMAGE" -lc "$launch"
}
case "${1:-}" in
  _start) start_remote "${2:-}" "${3:-}" ;;
  _status)
    sudo -n docker ps -a --filter label=experiment=mtp_preview25 --format '{{.Names}} {{.Status}}'
    curl -s -o /dev/null --max-time 3 -w 'health_http=%{http_code}\n' "http://127.0.0.1:$PORT/health" || true ;;
  _stop)
    for mode in on off; do
      if sudo -n docker inspect "mtp-preview25-$mode" >/dev/null 2>&1; then sudo -n docker rm -f "mtp-preview25-$mode"; fi
    done ;;
  start)
    mode="${2:-on}"; [[ "$mode" == on || "$mode" == off ]] || exit 2
    run_id="${3:-$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$BASE/results/$run_id/$mode"
    printf 'start=%s\nhost=%s\nimage=%s\nimage_id=%s\nmodel=%s\nmode=%s\nport=%s\nchunk=8192\nmem_fraction=0.88\n' "$(date -Iseconds)" "$HOST" "$IMAGE" "$IMAGE_ID" "$MODEL" "$mode" "$PORT" > "$BASE/results/$run_id/$mode/manifest.txt"
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$BASE/results/$run_id/$mode' "$(dirname "$PATCH")""
    scp -q -o LogLevel=ERROR "$BASE/serve.sh" "$HOST:$BASE/serve.sh"
    scp -q -o LogLevel=ERROR "$PATCH" "$HOST:$PATCH"
    scp -q -o LogLevel=ERROR "$BASE/results/$run_id/$mode/manifest.txt" "$HOST:$BASE/results/$run_id/$mode/manifest.txt"
    ssh "${SSH_OPTS[@]}" "$HOST" "bash '$BASE/serve.sh' _start '$mode' '$run_id'"
    echo "run_id=$run_id endpoint=http://$HOST:$PORT log=$BASE/results/$run_id/$mode/server.log" ;;
  status) ssh "${SSH_OPTS[@]}" "$HOST" "bash '$BASE/serve.sh' _status" ;;
  stop) ssh "${SSH_OPTS[@]}" "$HOST" "bash '$BASE/serve.sh' _stop" ;;
  *) echo 'usage: serve.sh start on|off [run_id] | status | stop' >&2; exit 2 ;;
esac
