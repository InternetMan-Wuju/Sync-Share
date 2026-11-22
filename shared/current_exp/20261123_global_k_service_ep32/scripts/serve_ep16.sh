#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?off|eplb_online|eplb_static|k1_invalid|status|stop}"
RUN_ID="${2:-$(date +%Y%m%d_%H%M%S)}"
IMG="${IMG:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-18023}"
NAME="cyt-threeway-ep16-${MODE}"
LOG="$ROOT/logs/${MODE}_${RUN_ID}"
CTX="${CTX:-65536}"
MAX_PREFILL="${MAX_PREFILL:-32768}"
CHUNKED="${CHUNKED:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.78}"
OBSERVE="${OBSERVE:-0}"
INSPECT="${INSPECT:-0}"
KCONFIG="${KCONFIG:-$ROOT/control/k1.json}"
LAYOUT="${LAYOUT:-$ROOT/layouts/eplb16_static.json}"

if [[ "$MODE" == status ]]; then
  sudo -n docker ps --filter "name=^/${NAME}$" --format '{{.Names}} {{.Status}}'
  curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 10 "http://127.0.0.1:$PORT/health" || true
  exit 0
fi
if [[ "$MODE" == stop ]]; then
  sudo -n docker stop "$NAME" >/dev/null 2>&1 || true
  sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo stopped
  exit 0
fi

mkdir -p "$LOG/observer"
DEV=(); for i in $(seq 0 15); do DEV+=(--device "/dev/davinci$i"); done
DEV+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
EPLB_FLAGS=(); EXTRA_MOUNT=(); OBS_ENV=(); WARMUP_FLAGS=()
case "$MODE" in
  off)
    EPLB_FLAGS=(--ep-num-redundant-experts 0)
    ;;
  eplb_online)
    EPLB_FLAGS=(--enable-eplb --ep-num-redundant-experts 16 --eplb-algorithm deepseek_hierarchical --eplb-rebalance-num-iterations 2 --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 1)
    ;;
  eplb_static)
    [[ -f "$LAYOUT" ]] || { echo "missing layout: $LAYOUT" >&2; exit 2; }
    EPLB_FLAGS=(--ep-num-redundant-experts 16 --ep-dispatch-algorithm static --init-expert-location "/runexp/layouts/$(basename "$LAYOUT")")
    EXTRA_MOUNT=(-v "$LAYOUT:/runexp/layouts/$(basename "$LAYOUT"):ro")
    ;;
  k1_invalid)
    EPLB_FLAGS=(--ep-num-redundant-experts 1)
    ;;
  k1|k2)
    [[ -f "$KCONFIG" ]] || { echo "missing K config: $KCONFIG" >&2; exit 2; }
    EPLB_FLAGS=(--ep-num-redundant-experts 0)
    EXTRA_MOUNT+=(-v "$ROOT/plugin:/runexp/plugin:ro" -v "$KCONFIG:/runexp/control/global_k.json:ro")
    OBS_ENV=(-e PYTHONPATH=/runexp/plugin:/sgl-workspace/sglang/python -e CYT_GLOBAL_K_CONFIG=/runexp/control/global_k.json -e CYT_K_LOG_DIR=/exp-logs/k_service)
    WARMUP_FLAGS=(--skip-server-warmup)
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2;;
esac
if [[ "$OBSERVE" == 1 || "$INSPECT" == 1 ]]; then
  EXTRA_MOUNT+=(-v "$ROOT/plugin:/runexp/plugin:ro")
  OBS_ENV=(-e PYTHONPATH=/runexp/plugin:/sgl-workspace/sglang/python)
  if [[ "$OBSERVE" == 1 ]]; then OBS_ENV+=(-e CYT_LAYER49_OBSERVER=1 -e CYT_OBSERVER_DIR=/exp-logs/observer -e CYT_OBSERVER_CAPTURE_HIDDEN=1); fi
  if [[ "$INSPECT" == 1 ]]; then OBS_ENV+=(-e CYT_INSPECT_EXPERT=1 -e CYT_INSPECT_DIR=/exp-logs/inspect); fi
fi

LAUNCH="set -eo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SGLANG_HOST_IP=10.120.72.23 HCCL_IF_IP=10.120.72.23
export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
export HCCL_BUFFSIZE=1000 HCCL_OP_EXPANSION_MODE=AIV
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export SGLANG_NPU_USE_MULTI_STREAM=1 STREAMS_PER_DEVICE=32 SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error PYTHONDONTWRITEBYTECODE=1 TE_PARALLEL_COMPILER=1
export HF_ENDPOINT=https://hf-mirror.com
python3 -m sglang.launch_server \\
  --model-path /models/glm-q --served-model-name glm-5.2 \\
  --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
  --dtype bfloat16 --quantization modelslim --trust-remote-code \\
  --tp-size 16 --ep-size 16 --dp-size 1 --nnodes 1 --node-rank 0 --dist-timeout 1800 \\
  --moe-a2a-backend deepep --deepep-mode normal --disable-shared-experts-fusion \\
  ${EPLB_FLAGS[*]} \\
  --context-length $CTX --chunked-prefill-size $CHUNKED --max-prefill-tokens $MAX_PREFILL --max-total-tokens $CTX \\
  --max-running-requests 1 --mem-fraction-static $MEM_FRACTION \\
  --disable-prefill-cuda-graph --disable-decode-cuda-graph --disable-overlap-schedule --disable-radix-cache \\
  --enable-cache-report --enable-metrics --watchdog-timeout 3600 \\
  2>&1 | tee /exp-logs/server.log"
printf '%s\n' "$LAUNCH" > "$LOG/launch.sh"
sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo -n docker run -d --name "$NAME" --net host --ipc host --shm-size 16g --pids-limit -1 --ulimit nofile=65535:65535 --security-opt seccomp=unconfined \
  "${DEV[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v "$MODEL:/models/glm-q:ro" -v "$LOG:/exp-logs" \
  "${EXTRA_MOUNT[@]}" "${OBS_ENV[@]}" \
  --entrypoint bash "$IMG" -lc "$LAUNCH" >/dev/null
echo "started mode=$MODE name=$NAME port=$PORT log=$LOG"
