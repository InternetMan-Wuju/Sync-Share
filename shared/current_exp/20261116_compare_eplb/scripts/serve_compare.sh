#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?off|eplb0|eplb16|ours}
RUN_ID=${2:-$(date +%Y%m%d_%H%M%S)}
ROOT=/home/chengyetai/workspace/current_exp/20261116_compare_eplb
LOG=$ROOT/logs/${MODE}_${RUN_ID}
mkdir -p "$LOG"
MODEL=/data_lib/data/models/glm-q
IMG=quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3
NAME=cyt-compare-${MODE}
PORT=18000
REBAL=${REBAL:-2}
MEM_FRACTION=${MEM_FRACTION:-0.80}
CONTEXT=${CONTEXT:-2048}
MAX_PREFILL=${MAX_PREFILL:-2048}
MAX_TOTAL=${MAX_TOTAL:-2048}
DEV=(); for i in $(seq 0 15); do DEV+=(--device "/dev/davinci$i"); done
DEV+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
EPLB_FLAGS=""
REC_FLAGS=""
EXTRA_MOUNT=""
PYENV=""
case "$MODE" in
  off)
    EPLB_FLAGS="--ep-num-redundant-experts 0"
    ;;
  eplb0)
    EPLB_FLAGS="--enable-eplb --ep-num-redundant-experts 0 --eplb-rebalance-num-iterations $REBAL"
    REC_FLAGS="--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 1"
    ;;
  eplb16)
    EPLB_FLAGS="--enable-eplb --ep-num-redundant-experts 16 --eplb-rebalance-num-iterations $REBAL"
    REC_FLAGS="--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 1"
    ;;
  ours)
    EPLB_FLAGS="--ep-num-redundant-experts 16 --ep-dispatch-algorithm static --init-expert-location /runexp/layout.json"
    PYENV="export CYT_EP16_EXPERIMENT=1; export PYTHONPATH=/runexp/plugin:/sgl-workspace/sglang/python:\$PYTHONPATH"
    EXTRA_MOUNT="-v $ROOT:/runexp:rw"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2;;
esac
# EPLB needs the NPU-safe updater patch; keep it read-only.
if [[ "$MODE" == eplb0 || "$MODE" == eplb16 ]]; then
  EXTRA_MOUNT="-v /home/chengyetai/workspace/current_exp/pd_colloc_20261011/patches/expert_location_updater_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro -v /home/chengyetai/workspace/current_exp/pd_colloc_20261011/patches/expert_distribution_phase_window.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro"
fi
LAUNCH="set -eo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SGLANG_HOST_IP=10.120.72.19 HCCL_IF_IP=10.120.72.19
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
$PYENV
python3 -m sglang.launch_server \
  --model-path /models/glm-q --served-model-name glm-5.2 \
  --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \
  --dtype bfloat16 --quantization modelslim --trust-remote-code \
  --tp-size 16 --ep-size 16 --dp-size 1 --nnodes 1 --node-rank 0 --dist-timeout 1800 \
  --moe-a2a-backend deepep --deepep-mode auto --disable-shared-experts-fusion \
  $EPLB_FLAGS $REC_FLAGS \
  --context-length $CONTEXT --chunked-prefill-size 8192 --max-prefill-tokens $MAX_PREFILL --max-total-tokens $MAX_TOTAL \
  --max-running-requests 1 --mem-fraction-static $MEM_FRACTION \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph --disable-overlap-schedule --disable-radix-cache \
  --enable-cache-report --enable-metrics --watchdog-timeout 3600 \
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
  $EXTRA_MOUNT \
  -v /home/chengyetai/workspace/current_exp/pd_colloc_20261011/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro \
  --entrypoint bash "$IMG" -lc "$LAUNCH"
echo "started mode=$MODE name=$NAME port=$PORT log=$LOG"
