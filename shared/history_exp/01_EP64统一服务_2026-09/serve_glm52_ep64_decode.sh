#!/bin/bash
set -euo pipefail

# 参数化版：EP / EPLB / 补丁 / NZ 开关都由环境变量控制。
# 用法：bash serve_glm52_ep64_decode.sh <NODE_RANK>

DOCKER="${DOCKER:-sudo docker}"
IMG=quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16
MODEL=/mnt/raid/user/data/models/glm-q
PORT="${PORT:-30000}"
NODE0_IP="${NODE0_IP:-10.120.72.23}"
NNODES="${NNODES:-4}"
EP_SIZE="${EP_SIZE:-64}"
ENABLE_EPLB="${ENABLE_EPLB:-1}"
PATCH12="${PATCH12:-1}"
PATCH2="${PATCH2:-1}"
NZ_OFF="${NZ_OFF:-0}"
NODE_RANK="${1:?usage: $0 <NODE_RANK>}"

N=$(hostname | sed 's/npu1-//')
LOCAL_IP="10.120.72.${N}"

DEVS=""
for i in $(seq 0 15); do
  DEVS="${DEVS} --device /dev/davinci${i}"
done
DEVS="${DEVS} --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"

MOUNTS="
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /var/queue_schedule:/var/queue_schedule
  -v ${MODEL}:/models/glm-q:ro
"
if [ "${PATCH12}" = "1" ]; then
  MOUNTS="${MOUNTS}
  -v /tmp/expert_location_updater_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro"
fi
if [ "${PATCH2}" = "1" ]; then
  MOUNTS="${MOUNTS}
  -v /tmp/torch_npu_distributed_c10d_patched.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro"
fi
if [ "${INSTRUMENT}" = "1" ]; then
  MOUNTS="${MOUNTS}
  -v /tmp/tokenizer_manager_instrumented.py:/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py:ro"
fi

LOG=/tmp/glm52_ep_${EP_SIZE}_$(hostname).log

if [ "${ENABLE_EPLB}" = "1" ]; then
  EPLB_ARGS="--enable-eplb"
else
  EPLB_ARGS=""
fi
EPLB_ARGS="${EPLB_ARGS} --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size -1"

CACHE_SPEC_ARGS=""
if [ "${MTP}" = "1" ]; then
  CACHE_SPEC_ARGS="${CACHE_SPEC_ARGS} --speculative-algorithm NEXTN"
fi

if [ "${NZ_OFF}" = "1" ]; then
  NZ_EXPORT="export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1"
else
  NZ_EXPORT=""
fi

RUN_CMD="
  unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
  export HOME=/tmp/glm52-home
  mkdir -p /tmp/glm52-home
  export PYTHONDONTWRITEBYTECODE=1
  export SGLANG_SET_CPU_AFFINITY=1
  export GLOO_SOCKET_IFNAME=bond4
  export HCCL_SOCKET_IFNAME=bond4
  export STREAMS_PER_DEVICE=32
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
  export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
  export SGLANG_LOG_EXPERT_LOCATION_METADATA=1
  ${NZ_EXPORT}
  export SGLANG_HOST_IP=${LOCAL_IP}
  export HCCL_IF_IP=${LOCAL_IP}
  export HCCL_BUFFSIZE=650
  export TASK_QUEUE_ENABLE=0
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
  export SGLANG_NPU_USE_MULTI_STREAM=1
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  export HCCL_IF_BASE_PORT=50000
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export TRANSFORMERS_VERBOSITY=error
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  python3 -m sglang.launch_server --tp-size ${EP_SIZE} --ep-size ${EP_SIZE} --nnodes ${NNODES} --node-rank ${NODE_RANK} --dist-init-addr ${NODE0_IP}:5000 --dist-timeout 1800 --model-path /models/glm-q --attention-backend ascend --device npu --quantization modelslim --trust-remote-code --served-model-name glm-5.2 --moe-a2a-backend deepep --deepep-mode low_latency --cuda-graph-max-bs 128 --max-running-requests 128 --mem-fraction-static 0.70 --context-length 131072 --chunked-prefill-size 128 --tokenizer-worker-num 4 --disable-shared-experts-fusion --dtype bfloat16 --host 0.0.0.0 --port ${PORT} --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers ${EPLB_ARGS} ${CACHE_SPEC_ARGS} --watchdog-timeout 9000
  2>&1 | tee ${LOG}
"

${DOCKER} rm -f glm52-ep64-decode-chengyetai 2>/dev/null || true

${DOCKER} run -d \
  --name glm52-ep64-decode-chengyetai \
  --net=host \
  --ipc=host \
  --shm-size=16g \
  ${DEVS} \
  ${MOUNTS} \
  --entrypoint bash \
  ${IMG} \
  -c "${RUN_CMD}"

echo "[started] $(hostname) rank=${NODE_RANK} ip=${LOCAL_IP} log=${LOG} ep=${EP_SIZE} eplb=${ENABLE_EPLB} patch12=${PATCH12} patch2=${PATCH2} nz_off=${NZ_OFF}"
