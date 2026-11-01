#!/bin/bash
set -euo pipefail

# PD 角色启动脚本（2P2D 已验证配置）。
# prefill: 每机 TP16 + DP4，deepep normal，禁 graph。
# decode : 两机 TP32 + DP32 + EP32，deepep low_latency，decode graph 开。
# 用法: bash serve_pd.sh <prefill|decode> <NODE_RANK>

ROLE="${1:?usage: serve_pd.sh <prefill|decode> <NODE_RANK>}"
NODE_RANK="${2:?usage: serve_pd.sh <prefill|decode> <NODE_RANK>}"

# 固定 2026-08-30 / fe694986 镜像，避免 main 标签漂移破坏实验对照。
IMG="${IMG:-sglang-glm52-pd:fe694986}"
MODEL="${MODEL:-/mnt/raid/user/data/models/glm-qq}"

PORT="${PORT:-18423}"
CTX="${CTX:-131072}"
MF_STORE_URL="${MF_STORE_URL:-tcp://10.120.72.23:24727}"
NANFIX="${NANFIX:-0}"
LOAD_BALANCE="${LOAD_BALANCE:-round_robin}"
DC_GRAPH="${DC_GRAPH:-on}"
SPEC="${SPEC:-off}"
DECODE_NNODES="${DECODE_NNODES:-2}"
DECODE_TP="${DECODE_TP:-$((DECODE_NNODES*16))}"
DECODE_EP="${DECODE_EP:-${DECODE_TP}}"
DECODE_DP="${DECODE_DP:-${DECODE_TP}}"
MASTER_IP="${MASTER_IP:-10.120.72.31}"
DEEPEP_MODE="${DEEPEP_MODE:-low_latency}"
PF_MOE_DENSE_TP="${PF_MOE_DENSE_TP:-0}"
PR32209="${PR32209:-1}"
PR32209_ROOT="${PR32209_ROOT:-/home/chengyetai/workspace/current_exp/patches/pr32209}"
EPLB="${EPLB:-1}"
EPLB_PATCHES="${EPLB_PATCHES:-${EPLB}}"
EPLB_PATCH_ROOT="${EPLB_PATCH_ROOT:-/home/chengyetai/workspace/current_exp/patches/eplb}"
EP_NUM_REDUNDANT="${EP_NUM_REDUNDANT:-0}"
EPLB_REBALANCE_ITERATIONS="${EPLB_REBALANCE_ITERATIONS:-1000}"
EXPERT_RECORDER="${EXPERT_RECORDER:-1}"
FULL_WINDOW_DUMP="${FULL_WINDOW_DUMP:-0}"
FULL_WINDOW_PATCH_ROOT="${FULL_WINDOW_PATCH_ROOT:-/home/chengyetai/workspace/current_exp/patches/full_window}"

N=$(hostname | sed 's/npu1-//')
LOCAL_IP="10.120.72.${N}"

DEVS=""
for i in $(seq 0 15); do DEVS="${DEVS} --device /dev/davinci${i}"; done
DEVS="${DEVS} --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"

if [ "${ROLE}" = "prefill" ]; then
  MEM="${MEM:-0.90}"
  MAX_RUNNING="${MAX_RUNNING:-4}"
  PF_CPS="${PF_CPS:-16384}"
  DISAGG_ARGS="--disaggregation-mode prefill --disaggregation-transfer-backend ascend --disaggregation-bootstrap-port 18998"
  DIST_ARGS="--tp-size 16 --dp-size 4 --nnodes 1 --node-rank ${NODE_RANK}"
  if [ "${PF_MOE_DENSE_TP}" = "1" ]; then
    MOE_ARGS="--moe-a2a-backend deepep --deepep-mode normal --moe-dense-tp 1"
  else
    MOE_ARGS="--moe-a2a-backend deepep --deepep-mode normal"
  fi
  DP_ARGS="--enable-dp-attention --enable-dp-lm-head"
  GRAPH_ARGS="--disable-cuda-graph"
  if [ "${SPEC}" = "off" ]; then
    SPEC_ARGS=""
  else
    SPEC_ARGS="--speculative-draft-model-quantization unquant --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2"
  fi
  CPS=${PF_CPS}
  MAX_PREFILL=180000
else
  MEM="${MEM:-0.82}"
  MAX_RUNNING="${MAX_RUNNING:-128}"
  DISAGG_ARGS="--disaggregation-mode decode --disaggregation-transfer-backend ascend"
  if [ "${DECODE_NNODES}" -gt 1 ]; then
    DIST_ARGS="--tp-size ${DECODE_TP} --dp-size ${DECODE_DP} --ep-size ${DECODE_EP} --nnodes ${DECODE_NNODES} --node-rank ${NODE_RANK} --dist-init-addr ${MASTER_IP}:5000 --dist-timeout 1800"
  else
    DIST_ARGS="--tp-size ${DECODE_TP} --dp-size ${DECODE_DP} --ep-size ${DECODE_EP} --nnodes 1 --node-rank ${NODE_RANK}"
  fi
  MOE_ARGS="--moe-a2a-backend deepep --deepep-mode ${DEEPEP_MODE}"
  if [ "${DP_ATTN:-on}" = "off" ]; then
    DP_ARGS=""
  else
    DP_ARGS="--enable-dp-attention"
  fi
  if [ "${DC_GRAPH}" = "off" ]; then
    GRAPH_ARGS="--disable-cuda-graph"
  else
    GRAPH_ARGS="--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32"
  fi
  if [ "${SPEC}" = "off" ]; then
    SPEC_ARGS=""
  else
    SPEC_ARGS="--speculative-draft-model-quantization unquant --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
  fi
  CPS=16384
  MAX_PREFILL=131072
fi

LOG=/tmp/pd_${ROLE}_$(hostname).log

NANFIX_MOUNT=""
if [ "${NANFIX}" = "1" ]; then
  NANFIX_MOUNT="
  -v /mnt/raid/user/data/tools/bench3/patch/g4m.patched:/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py:ro
  -v /mnt/raid/user/data/tools/bench3/patch/dsv2.patched:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:ro"
fi

PR32209_MOUNT=""
if [ "${PR32209}" = "1" ]; then
  PR32209_MOUNT="
  -v ${PR32209_ROOT}/python/sglang/srt/disaggregation/decode.py:/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/managers/schedule_batch.py:/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/managers/scheduler_components/dp_attn.py:/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/speculative/base_spec_worker.py:/sgl-workspace/sglang/python/sglang/srt/speculative/base_spec_worker.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/speculative/eagle_worker_v2.py:/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py:/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/model_executor/forward_batch_info.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:ro
  -v ${PR32209_ROOT}/python/sglang/srt/layers/attention/dsa_backend.py:/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:ro"
fi

EPLB_MOUNT=""
if [ "${EPLB_PATCHES}" = "1" ]; then
  EPLB_MOUNT="
  -v ${EPLB_PATCH_ROOT}/expert_location_updater_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro
  -v ${EPLB_PATCH_ROOT}/torch_npu_distributed_c10d_patched.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro"
fi

FULL_WINDOW_MOUNT=""
if [ "${FULL_WINDOW_DUMP}" = "1" ]; then
  FULL_WINDOW_MOUNT="
  -v ${FULL_WINDOW_PATCH_ROOT}/expert_distribution_full_window_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro"
fi

EPLB_ARGS=""
if [ "${EPLB}" = "1" ]; then
  EPLB_ARGS="--enable-eplb"
fi

EPLB_ITER_ARGS="--eplb-rebalance-num-iterations ${EPLB_REBALANCE_ITERATIONS}"

RECORDER_ARGS=""
if [ "${EXPERT_RECORDER}" = "1" ]; then
  RECORDER_ARGS="--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size -1"
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
  export HCCL_OP_EXPANSION_MODE=AIV
  export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
  export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
  export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
  export TASK_QUEUE_ENABLE=0
  export SGLANG_NPU_USE_MULTI_STREAM=1
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  export SGLANG_SCHEDULER_SKIP_ALL_GATHER=${SKIP_ALL_GATHER:-0}
  export SGLANG_LOG_EXPERT_LOCATION_METADATA=1
  export SGLANG_DUMP_EPLB_WINDOWS=${FULL_WINDOW_DUMP}
  export ASCEND_MF_STORE_URL=${MF_STORE_URL}
  export ASCEND_MF_TRANSFER_PROTOCOL=sdma
  export SGLANG_HOST_IP=${LOCAL_IP}
  export HCCL_IF_IP=${LOCAL_IP}
  export HCCL_BUFFSIZE=650
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export TRANSFORMERS_VERBOSITY=error
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  python3 -m sglang.launch_server \
    ${DIST_ARGS} \
    --model-path /models/glm-q \
    --served-model-name glm-5.2 \
    --host 0.0.0.0 --port ${PORT} \
    --device npu --attention-backend ascend \
    --dtype bfloat16 \
    --quantization modelslim \
    --trust-remote-code \
    ${MOE_ARGS} \
    --context-length ${CTX} \
    --chunked-prefill-size ${CPS} \
    --max-prefill-tokens ${MAX_PREFILL} \
    --mem-fraction-static ${MEM} \
    --max-running-requests ${MAX_RUNNING} \
    --disable-shared-experts-fusion \
    --load-balance-method ${LOAD_BALANCE} \
    --ep-num-redundant-experts ${EP_NUM_REDUNDANT} \
    ${EPLB_ARGS} \
    ${EPLB_ITER_ARGS} \
    ${RECORDER_ARGS} \
    --tokenizer-worker-num 4 \
    ${DP_ARGS} \
    --enable-metrics --enable-metrics-for-all-schedulers \
    ${SPEC_ARGS} \
    ${GRAPH_ARGS} \
    ${DISAGG_ARGS} \
    --watchdog-timeout 9000 \
    2>&1 | tee ${LOG}
"

sudo docker rm -f pd-${ROLE}-glm52 2>/dev/null || true

sudo docker run -d \
  --name pd-${ROLE}-glm52 \
  --net=host \
  --ipc=host \
  --shm-size=16g \
  --pids-limit=-1 \
  ${DEVS} \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/queue_schedule:/var/queue_schedule \
  -v ${MODEL}:/models/glm-q:ro \
  ${NANFIX_MOUNT} \
  ${PR32209_MOUNT} \
  ${EPLB_MOUNT} \
  ${FULL_WINDOW_MOUNT} \
  --entrypoint bash \
  ${IMG} \
  -c "${RUN_CMD}"

echo "[started] ${ROLE} ip=${LOCAL_IP} port=${PORT} log=${LOG}"
