#!/usr/bin/env bash
set -euo pipefail

# Manual Chunk Size workflow. Each host is an independent TP16/EP16 server.
BASE=/home/chengyetai/workspace/current_exp/chunk_manual_20260927
RUN_BASE=/home/chengyetai/workspace/current_exp/chunk_redo_20260928
GRAPH_MODE="${GRAPH_MODE:-full}"
[[ "$GRAPH_MODE" == full || "$GRAPH_MODE" == disabled ]] || { echo "invalid GRAPH_MODE=$GRAPH_MODE" >&2; exit 2; }
NODE="${2:-23}"
[[ "$NODE" == 23 || "$NODE" == 25 ]] || { echo 'node must be 23 or 25' >&2; exit 2; }
HOST="10.120.72.$NODE"
IMAGE=sglang-glm52-pd:fe694986
IMAGE_ID=sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d
MODEL="${MODEL:-/mnt/raid/user/data/models/glm-qq}"
[[ "$MODEL" == /mnt/raid/user/data/models/glm-q || "$MODEL" == /mnt/raid/user/data/models/glm-qq ]] || { echo "unsupported read-only checkpoint path" >&2; exit 2; }
PORT=30080
DEEPEP_FIX="${DEEPEP_FIX:-0}"
[[ "$DEEPEP_FIX" == 0 || "$DEEPEP_FIX" == 1 ]] || { echo "DEEPEP_FIX must be 0 or 1" >&2; exit 2; }
DEEPEP_PATCH="$BASE/patches/deepep_fix_20260927"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-0}"
[[ "$WEIGHT_SOURCE" == 0 || "$WEIGHT_SOURCE" == 1 ]] || { echo "WEIGHT_SOURCE must be 0 or 1" >&2; exit 2; }
WEIGHT_PATCH="$BASE/patches/readonly_weight_source"
WEIGHT_DATA="$BASE/patches/readonly_source$NODE"
WEIGHT_REFERENCE_URL="${WEIGHT_REFERENCE_URL:-}"
[[ -z "$WEIGHT_REFERENCE_URL" || "$WEIGHT_REFERENCE_URL" == http://127.0.0.1:30122 ]] || { echo "unsupported reference URL" >&2; exit 2; }
SSH_OPTS=(-T -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)

valid_chunk() { [[ "${1:-}" == 1024 || "${1:-}" == 8192 ]]; }
name_for() { echo "chunk-redo${NODE}-$1"; }
remote_start() {
  local chunk="$1" run_id="$2" name logdir devices=() extra_mounts=() i launch overlay_launch="" manifest_sha=""
  valid_chunk "$chunk" || { echo 'chunk must be 1024 or 8192' >&2; return 2; }
  [[ "$run_id" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo 'invalid run id' >&2; return 2; }
  name=$(name_for "$chunk")
  logdir="$RUN_BASE/results/$run_id/node_$NODE/chunk_$chunk"
  mkdir -p "$logdir"
  [[ -d "$MODEL" ]] || { echo "missing model $MODEL" >&2; return 1; }
  [[ -f "$BASE/patches/load_model_utils.py" ]] || { echo "missing loading timeout patch" >&2; return 1; }
  [[ "$(sudo -n docker image inspect "$IMAGE" -f '{{.Id}}')" == "$IMAGE_ID" ]] || { echo 'image ID differs' >&2; return 1; }
  if [[ "$DEEPEP_FIX" == 1 ]]; then
    python3 "$DEEPEP_PATCH/verify_copy.py"
    extra_mounts+=(-v "$DEEPEP_PATCH/deep_ep:/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep:ro")
    cp "$DEEPEP_PATCH/manifest.json" "$logdir/deepep_copy_manifest.json"
  fi
  if [[ "$WEIGHT_SOURCE" == 1 ]]; then
    python3 "$WEIGHT_PATCH/verify_payloads.py" "$WEIGHT_DATA" > "$logdir/weight_source_preflight.json"
    if [[ -n "$WEIGHT_REFERENCE_URL" ]]; then
      curl -fsS --max-time 10 "$WEIGHT_REFERENCE_URL/health" > "$logdir/reference_health.json"
    fi
    manifest_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$WEIGHT_DATA/summary.json")
    extra_mounts+=(-v "$WEIGHT_PATCH:/readonly-loader:ro" -v "$WEIGHT_DATA:/readonly-weight-source:ro")
    cp "$WEIGHT_DATA/summary.json" "$logdir/weight_source_summary.json"
    mkdir -p "$logdir/weight_loader"
    cp "$WEIGHT_PATCH/sitecustomize.py" "$WEIGHT_PATCH/external_weight_source.py" "$WEIGHT_PATCH/verify_payloads.py" "$logdir/weight_loader/"
    sha256sum "$WEIGHT_PATCH/sitecustomize.py" "$WEIGHT_PATCH/external_weight_source.py" > "$logdir/weight_loader_sha256.txt"
    overlay_launch="export SGLANG_READONLY_WEIGHT_MANIFEST=/readonly-weight-source/manifest.json
      export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=$manifest_sha
      export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q
      export SGLANG_READONLY_WEIGHT_AUDIT_DIR=/exp-logs
      export SGLANG_READONLY_WEIGHT_REFERENCE_URL=$WEIGHT_REFERENCE_URL
      export PYTHONPATH=/readonly-loader:\${PYTHONPATH:-}"
  fi
  if [[ "$GRAPH_MODE" == full ]]; then
    graph_flags="--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32"
  else
    graph_flags="--disable-cuda-graph"
  fi
  for i in 1024 8192; do
    if sudo -n docker inspect "$(name_for "$i")" >/dev/null 2>&1; then echo "container $(name_for "$i") already exists" >&2; return 1; fi
  done
  if ss -lnt | grep -Eq ":${PORT}[[:space:]]"; then echo "port $PORT already in use" >&2; return 1; fi
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
    $overlay_launch
    python3 -m sglang.launch_server \\
      --tp-size 16 --ep-size 16 --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode normal \\
      --context-length 131072 --chunked-prefill-size $chunk \\
      --max-prefill-tokens 65536 --max-running-requests 4 --mem-fraction-static 0.88 \\
      --disable-shared-experts-fusion $graph_flags \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sha256sum "$MODEL/config.json" "$MODEL/quant_model_description.json" > "$logdir/model_config_sha256.txt"
  sudo -n docker run -d --name "$name" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label experiment=chunk_redo_20260928 --label node="$NODE" --label chunk_size="$chunk" \
    "${devices[@]}" "${extra_mounts[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$BASE/patches/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMAGE" -lc "$launch"
}
remote_status() {
  sudo -n docker ps -a --filter label=experiment=chunk_manual_20260927 --format '{{.Names}} {{.Status}} {{.Image}}'
  for i in 1024 8192; do
    if sudo -n docker inspect "$(name_for "$i")" >/dev/null 2>&1; then
      sudo -n docker logs --tail 10 "$(name_for "$i")" 2>&1 || true
    fi
  done
  curl -sS -o /dev/null --max-time 3 -w 'health_http=%{http_code}\n' "http://127.0.0.1:$PORT/health" || true
  npu-smi info | grep '/ 65536' | head -16 || true
}
remote_stop() {
  local i
  for i in 1024 8192; do
    if sudo -n docker inspect "$(name_for "$i")" >/dev/null 2>&1; then sudo -n docker rm -f "$(name_for "$i")"; fi
  done
}
case "${1:-}" in
  _start) remote_start "${3:-}" "${4:-}" ;;
  _status) remote_status ;;
  _stop) remote_stop ;;
  start)
    chunk="${3:-8192}"
    valid_chunk "$chunk" || { echo 'chunk must be 1024 or 8192' >&2; exit 2; }
    run_id="${4:-$(date +%Y%m%d_%H%M%S)}"
    [[ "$run_id" =~ ^[0-9]{8}_[0-9]{6}$ ]] || exit 2
    logdir="$RUN_BASE/results/$run_id/node_$NODE/chunk_$chunk"
    mkdir -p "$logdir"
    printf 'start_time=%s\nhost=%s\nimage=%s\nimage_id=%s\nmodel=%s\nchunk_size=%s\nport=%s\nmem_fraction=0.88\nmtp=off\nmixed_chunk=default_false\ngraph_mode=%s\ndeepep_fix=%s\nweight_source=%s\n' \
      "$(date -Iseconds)" "$HOST" "$IMAGE" "$IMAGE_ID" "$MODEL" "$chunk" "$PORT" "$GRAPH_MODE" "$DEEPEP_FIX" "$WEIGHT_SOURCE" > "$logdir/manifest.txt"
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$BASE/patches' '$logdir'"
    # 23 is the current host; avoid overwriting a file with itself.
    if [[ "$NODE" == 25 ]]; then
      scp -q -o LogLevel=ERROR "$RUN_BASE/serve.sh" "$HOST:$BASE/serve.sh"
      scp -q -o LogLevel=ERROR "$BASE/patches/load_model_utils.py" "$HOST:$BASE/patches/load_model_utils.py"
      scp -q -o LogLevel=ERROR "$logdir/manifest.txt" "$HOST:$logdir/manifest.txt"
    fi
    if [[ "$DEEPEP_FIX" == 1 && "$NODE" == 25 ]]; then
      rsync -az -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$DEEPEP_PATCH/" "$HOST:$DEEPEP_PATCH/"
    fi
    if [[ "$WEIGHT_SOURCE" == 1 && "$NODE" == 25 ]]; then
      rsync -az -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$WEIGHT_PATCH/" "$HOST:$WEIGHT_PATCH/"
      if [[ -d "$WEIGHT_DATA" ]]; then
        rsync -az -e 'ssh -T -o BatchMode=yes -o LogLevel=ERROR' "$WEIGHT_DATA/" "$HOST:$WEIGHT_DATA/"
      fi
    fi
    ssh "${SSH_OPTS[@]}" "$HOST" "GRAPH_MODE='$GRAPH_MODE' WEIGHT_REFERENCE_URL='$WEIGHT_REFERENCE_URL' WEIGHT_SOURCE='$WEIGHT_SOURCE' DEEPEP_FIX='$DEEPEP_FIX' MODEL='$MODEL' bash '$RUN_BASE/serve.sh' _start '$NODE' '$chunk' '$run_id'"
    echo "run_id=$run_id endpoint=http://$HOST:$PORT log=$logdir/server.log"
    ;;
  status) ssh "${SSH_OPTS[@]}" "$HOST" "bash '$RUN_BASE/serve.sh' _status '$NODE'" ;;
  stop) ssh "${SSH_OPTS[@]}" "$HOST" "bash '$RUN_BASE/serve.sh' _stop '$NODE'" ;;
  sync)
    run_id="${3:-}"
    [[ "$run_id" =~ ^[0-9]{8}_[0-9]{6}$ ]] || exit 2
    if [[ "$NODE" == 25 ]]; then
      mkdir -p "$BASE/results/$run_id/node_$NODE"
      rsync -az -e 'ssh -T -o LogLevel=ERROR' "$HOST:$BASE/results/$run_id/node_$NODE/" "$BASE/results/$run_id/node_$NODE/"
    fi
    ;;
  *) echo 'usage: serve.sh start 23|25 [8192|1024] [run_id] | status 23|25 | stop 23|25 | sync 23|25 run_id' >&2; exit 2 ;;
esac
