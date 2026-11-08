#!/usr/bin/env bash
set -eo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export GLOO_SOCKET_IFNAME=tap0 HCCL_SOCKET_IFNAME=tap0
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export SGLANG_HOST_IP=$HCCL_IF_IP HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=650 HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export SGLANG_NPU_USE_MULTI_STREAM=1 STREAMS_PER_DEVICE=32 SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/root/.cache/sglang/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang/inductor TMPDIR=/runexp/local/tmp
export PYTHONPATH=/runexp/plugin:/study/weights/loader:/sgl-workspace/sglang/python:${PYTHONPATH:-}
export SGLANG_READONLY_WEIGHT_MANIFEST=/study/weights/publisher/manifest.json
export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5
export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q SGLANG_READONLY_WEIGHT_AUDIT_DIR=/runexp/local/weight_audit
python3 -c 'import torch,torch_npu,tbe; torch.npu.set_device(0); x=torch.ones((8,8),device="npu"); y=torch_npu.npu_format_cast(x+1,2); torch.npu.synchronize(); assert y.sum().item()==128; print("NPU_PREFLIGHT_PASS",flush=True)'
exec python3 -m sglang.launch_server \
 --tp-size 16 --ep-size 16 --dp-size 1 --nnodes 1 --node-rank 0 --dist-timeout 1800 \
 --model-path /models/glm-q --served-model-name glm-5.2 --trust-remote-code \
 --host 0.0.0.0 --port 30080 --device npu --attention-backend ascend --dtype bfloat16 --quantization modelslim \
 --moe-a2a-backend deepep --deepep-mode auto --disable-shared-experts-fusion \
 --ep-num-redundant-experts 16 --ep-dispatch-algorithm static --init-expert-location /runexp/layout.json \
 --context-length 16384 --chunked-prefill-size 8192 --max-prefill-tokens 8192 --max-total-tokens 12288 --max-running-requests 1 --mem-fraction-static 0.86 \
 --disable-prefill-cuda-graph --disable-decode-cuda-graph --disable-overlap-schedule --disable-radix-cache \
 --enable-cache-report --enable-metrics --watchdog-timeout 1800
