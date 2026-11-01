#!/usr/bin/env bash
set -euo pipefail
# Bring-up: one independent single-node TP16/EP16 service per node (19/23/31/32).
DIR=/home/chengyetai/workspace/history_exp/06_四节点服务与权重核查_2026-10-11
IMAGE=sglang-glm52-pd:fe694986
IMAGE_ID=sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d
PORT="${PORT:-30080}"
CHUNK="${CHUNK:-8192}"
MEM="${MEM:-0.88}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-650}"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-0}"
WEIGHT_REFERENCE_URL="${WEIGHT_REFERENCE_URL:-}"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)
NODES=(19 23 31 32)

# /mnt/raid is a DEGRADED raid5 ([4/3]); use the nvme copy, identical across nodes.
model_for() { echo "${MODEL_ALL:-/data_lib/data/models/glm-q}"; }
name_for() { echo "bringup-glm52-$1"; }

remote_start() {
  local node="$1" run_id="$2" model devices=() extra=() i launch="" overlay=""
  model=$(model_for "$node")
  local logdir="$DIR/logs/$run_id/node_$node"
  mkdir -p "$logdir"
  [[ -d "$model" ]] || { echo "missing model $model" >&2; return 1; }
  [[ -f "$DIR/load_model_utils.py" ]] || { echo "missing load timeout patch" >&2; return 1; }
  [[ "$(sudo -n docker image inspect "$IMAGE" -f '{{.Id}}')" == "$IMAGE_ID" ]] || { echo "image mismatch on $(hostname)" >&2; return 1; }
  if sudo -n docker inspect "$(name_for "$node")" >/dev/null 2>&1; then echo "container exists" >&2; return 1; fi
  if ss -lnt | grep -Eq ":${PORT}[[:space:]]"; then echo "port $PORT busy" >&2; return 1; fi
  if [[ "$WEIGHT_SOURCE" == 1 ]]; then
    local wp="$DIR/readonly_weight_source" wd="$DIR/readonly_source$node"
    [[ -f "$wp/external_weight_source.py" && -f "$wd/manifest.json" ]] || { echo "weight source files missing" >&2; return 1; }
    extra+=(-v "$wp:/readonly-loader:ro" -v "$wd:/readonly-weight-source:ro")
    overlay="export SGLANG_READONLY_WEIGHT_MANIFEST=/readonly-weight-source/manifest.json
      export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$wd/summary.json")
      export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q
      export SGLANG_READONLY_WEIGHT_REFERENCE_URL=$WEIGHT_REFERENCE_URL
      export PYTHONPATH=/readonly-loader:\${PYTHONPATH:-}"
  fi
  for i in $(seq 0 15); do devices+=(--device "/dev/davinci$i"); done
  devices+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  local host_ip="10.120.72.$node"
  launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export SGLANG_HOST_IP=$host_ip HCCL_IF_IP=$host_ip
    export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=$HCCL_BUFFSIZE
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 STREAMS_PER_DEVICE=32
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    $overlay
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
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sudo -n docker run -d --name "$(name_for "$node")" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=bringup_4node_20261011 --label node="$node" \
    "${devices[@]}" "${extra[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$model:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$DIR/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMAGE" -lc "$launch"
  echo "started node=$node model=$model weight_source=$WEIGHT_SOURCE log=$logdir"
}

remote_status() {
  local node="$1"
  echo "== node=$node host=$(hostname)"
  sudo -n docker ps -a --filter "name=^/$(name_for "$node")$" --format '{{.Names}} {{.Status}} {{.Image}}'
  curl -sS -o /dev/null --max-time 4 -w 'health=%{http_code}\n' "http://127.0.0.1:$PORT/health" || true
  sudo -n docker logs --tail 4 "$(name_for "$node")" 2>&1 | tail -4 || true
}

remote_stop() { sudo -n docker rm -f "$(name_for "$1")" >/dev/null 2>&1 || true; }

invoke() { ssh "${SSH[@]}" "10.120.72.$1" "bash -s -- $2 ${*:3}" < "$0"; }

case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"
    targets=("${@:3}"); [[ ${#targets[@]} -gt 0 ]] || targets=("${NODES[@]}")
    for n in "${targets[@]}"; do
      ssh "${SSH[@]}" "10.120.72.$n" "mkdir -p '$DIR/logs/$run_id/node_$n'" || true
      scp -q -o LogLevel=ERROR "$DIR/load_model_utils.py" "$DIR/serve.sh" "10.120.72.$n:$DIR/" || true
      if [[ "$WEIGHT_SOURCE" == 1 ]]; then
        rsync -a -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$DIR/readonly_weight_source/" "10.120.72.$n:$DIR/readonly_weight_source/" 
        if [[ -d "$DIR/readonly_source$n" ]]; then rsync -a -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$DIR/readonly_source$n/" "10.120.72.$n:$DIR/readonly_source$n/"; fi
      fi
    done
    for n in "${targets[@]}"; do
      echo "--> starting node $n"
      ssh "${SSH[@]}" "10.120.72.$n" "WEIGHT_SOURCE='$WEIGHT_SOURCE' WEIGHT_REFERENCE_URL='$WEIGHT_REFERENCE_URL' PORT='$PORT' CHUNK='$CHUNK' MEM='$MEM' HCCL_BUFFSIZE='$HCCL_BUFFSIZE' bash '$DIR/serve.sh' __remote_start '$n' '$run_id'" || echo "start failed node=$n"
    done
    echo "run_id=$run_id"
    ;;
  status) for n in "${@:2}"; do invoke "$n" __remote_status "$n"; echo; done ;;
  stop)   for n in "${@:2}"; do invoke "$n" __remote_stop "$n"; done ;;
  __remote_start) remote_start "${@:2}" ;;
  __remote_status) remote_status "$2" ;;
  __remote_stop) remote_stop "$2" ;;
  *) echo "usage: serve.sh start <run_id> [nodes...] | status <nodes...> | stop <nodes...>" >&2; exit 2 ;;
esac
