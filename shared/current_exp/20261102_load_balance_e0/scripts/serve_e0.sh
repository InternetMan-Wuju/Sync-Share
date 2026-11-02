#!/usr/bin/env bash
set -euo pipefail

DIR=/home/chengyetai/workspace/current_exp/20261102_load_balance_e0
HIST=/home/chengyetai/workspace/history_exp/09_自主探索与ChunkSize_2026-10-12/autonomous_20261012
WEIGHTS="$HIST/weights"
EPLB_PATCH=/home/chengyetai/workspace/history_exp/07_PD共置与EPLB负优化_2026-10-11/pd_colloc_20261011/patches/expert_distribution_phase_window.py
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
IMG="${IMG:-quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3}"
NAME="${NAME:-cxy-e0-baseline}"
PORT="${PORT:-30080}"
MEM="${MEM:-0.88}"
CTX="${CTX:-131072}"
MAXPREF="${MAXPREF:-32768}"
MRR="${MRR:-16}"
CHUNK="${CHUNK:-16384}"
TP="${TP:-16}"
EP="${EP:-16}"
DP="${DP:-1}"
HCCL_BUF="${HCCL_BUF:-650}"
NET_IFNAME="${NET_IFNAME:-tap0}"
DISPATCH_TOKENS="${DISPATCH_TOKENS:-64}"
RECD_BUF="${RECD_BUF:-128}"
CACHE_ROOT="${CACHE_ROOT:-/data_lib/cyt/sglang-cache-e0}"
GRAPH_MAX_BS="${GRAPH_MAX_BS:-16}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16}"
MANIFEST_SHA="${MANIFEST_SHA:-7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5}"
DOCKER_HOST="${DOCKER_HOST:-unix:///tmp/chengyetai-rootless-docker/docker.sock}"
export DOCKER_HOST

docker_cmd() { docker "$@"; }

preflight() {
  [[ -d "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }
  [[ -f "$WEIGHTS/loader/sitecustomize.py" ]] || { echo "missing weight loader" >&2; exit 1; }
  [[ -f "$WEIGHTS/publisher/manifest.json" ]] || { echo "missing publisher manifest" >&2; exit 1; }
  local sha
  sha=$(sha256sum "$WEIGHTS/publisher/manifest.json" | awk '{print $1}')
  [[ "$sha" == "$MANIFEST_SHA" ]] || { echo "manifest SHA mismatch: $sha" >&2; exit 1; }
  if ss -lnt | grep -Eq ":${PORT}[[:space:]]"; then echo "port $PORT in use" >&2; exit 1; fi
  if docker_cmd inspect "$NAME" >/dev/null 2>&1; then echo "container $NAME exists" >&2; exit 1; fi
}

start() {
  local run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
  local logdir="$DIR/logs/$run_id"
  mkdir -p "$logdir/weight_audit" "$logdir/hf-cache" "$logdir/tmp" "$logdir/eplb_windows"
  mkdir -p "$CACHE_ROOT/triton" "$CACHE_ROOT/inductor"
  preflight
  local devs=()
  local i
  for i in $(seq 0 $((TP-1))); do devs+=(--device "/dev/davinci$i"); done
  devs+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)

  local launch="set -eo pipefail
    exec > >(tee /exp-logs/server.log) 2>&1
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=$NET_IFNAME HCCL_SOCKET_IFNAME=$NET_IFNAME
    export HCCL_IF_IP=\$(hostname -I | awk '{print \$1}')
    export SGLANG_HOST_IP=\$HCCL_IF_IP
    export HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=$HCCL_BUF HCCL_OP_EXPANSION_MODE=AIV
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$DISPATCH_TOKENS
    export SGLANG_ENABLE_OVERPLAN_PLAN_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 STREAMS_PER_DEVICE=32
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/exp-logs/hf-cache TMPDIR=/exp-logs/tmp
    export TRITON_CACHE_DIR=/root/.cache/sglang/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang/inductor
    export SGLANG_ENABLE_FAST_INPUT_LOGPROBS=0
    export PYTHONPATH=/study/weights/loader:\${PYTHONPATH:-}
    export SGLANG_READONLY_WEIGHT_MANIFEST=/study/weights/publisher/manifest.json
    export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=$MANIFEST_SHA
    export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q
    export SGLANG_READONLY_WEIGHT_AUDIT_DIR=/exp-logs/weight_audit
    export SGLANG_DUMP_EPLB_WINDOWS=1
    export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/exp-logs/eplb_windows
    python3 -c 'import tbe; print(\"[preflight] CANN tbe import OK\", flush=True)'
    python3 -m sglang.launch_server \\
      --tp-size $TP --dp-size $DP --ep-size $EP --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode auto \\
      --context-length $CTX --chunked-prefill-size $CHUNK --max-prefill-tokens $MAXPREF \\
      --max-running-requests $MRR --mem-fraction-static $MEM \\
      --disable-shared-experts-fusion \\
      --cuda-graph-backend-decode full --cuda-graph-max-bs-decode $GRAPH_MAX_BS --cuda-graph-bs-decode $GRAPH_BS \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size $RECD_BUF \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  {
    printf 'run_id=%s\n' "$run_id"
    printf 'started_at=%s\n' "$(date -Iseconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'image=%s\n' "$IMG"
    printf 'model=%s\n' "$MODEL"
    printf 'tp=%s ep=%s dp=%s\n' "$TP" "$EP" "$DP"
    printf 'port=%s mem=%s ctx=%s max_prefill=%s mrr=%s chunk=%s\n' "$PORT" "$MEM" "$CTX" "$MAXPREF" "$MRR" "$CHUNK"
    printf 'cache_root=%s\n' "$CACHE_ROOT"
    printf 'hccl_buf=%s dispatch_tokens=%s recd_buf=%s graph_max_bs=%s graph_bs=%s net_ifname=%s\n' "$HCCL_BUF" "$DISPATCH_TOKENS" "$RECD_BUF" "$GRAPH_MAX_BS" "$GRAPH_BS" "$NET_IFNAME"
    printf 'weight_manifest_sha256=%s\n' "$MANIFEST_SHA"
  } > "$logdir/manifest.txt"
  sha256sum "$MODEL/config.json" "$MODEL/quant_model_description.json" "$WEIGHTS/publisher/manifest.json" > "$logdir/input_sha256.txt"
  docker_cmd image inspect "$IMG" --format '{{.Id}} {{.RepoDigests}}' > "$logdir/image_digest.txt"

  docker_cmd run -d --name "$NAME" --rm \
    --net host --shm-size 16g --security-opt seccomp=unconfined \
    --label owner=chengyetai --label experiment=load_balance_e0 --label run_id="$run_id" \
    "${devs[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" \
    -v "$WEIGHTS/loader:/study/weights/loader:ro" \
    -v "$WEIGHTS/publisher:/study/weights/publisher:ro" \
    -v "$EPLB_PATCH:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro" \
    -v "$logdir:/exp-logs" \
    -v "$CACHE_ROOT:/root/.cache/sglang" \
    --entrypoint bash "$IMG" -lc "$launch" >/dev/null
  echo "started run_id=$run_id name=$NAME endpoint=http://127.0.0.1:$PORT log=$logdir/server.log"
}

status() {
  docker_cmd ps -a --filter "name=$NAME" --format '{{.Names}} {{.Status}} {{.Image}}'
  curl -sS -o /dev/null -w 'health_http=%{http_code}\n' --max-time 20 "http://127.0.0.1:$PORT/health" || true
  npu-smi info | grep '/ 65536' | head -n 16 || true
}

stop() {
  docker_cmd stop "$NAME" >/dev/null 2>&1 || true
  echo stopped
}

case "${1:-}" in
  start) start "${2:-}" ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start [run_id]|status|stop}" >&2; exit 2 ;;
esac
