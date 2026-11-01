#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chengyetai/workspace/current_exp/eplb_pd_agentx_20260928
ROLE=${1:?prefill/decode/router} NODE=${2:?node-number} RUN=${3:?run-id} EPLB=${4:?0/1} REDUNDANT=${5:?integer}
IMAGE=sglang-glm52-pd:fe694986
EXPECTED=sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d
IP=10.120.72.$NODE
NAME=eplb26-${ROLE}-${NODE}
LOG=$ROOT/results/$RUN/node_$NODE
mkdir -p "$LOG/eplb_windows" "$LOG/weight_audit"
MODEL=/mnt/raid/user/data/models/glm-qq

[[ -d $MODEL ]] || { echo "model missing $MODEL" >&2; exit 1; }
[[ $(sudo -n docker image inspect "$IMAGE" --format '{{.Id}}') == "$EXPECTED" ]] || { echo "image ID differs at $NODE" >&2; exit 1; }
[[ ! $(sudo -n docker ps -a --filter name="^/$NAME$" --format '{{.Names}}') ]] || { echo "container exists: $NAME" >&2; exit 1; }
if [[ $ROLE != router ]]; then
  if [[ $NODE == 25 ]]; then SOURCE=$ROOT/patches/readonly_source25; else SOURCE=$ROOT/patches/readonly_source23; fi
  python3 "$ROOT/patches/readonly_weight_source/verify_payloads.py" "$SOURCE" > "$LOG/weight_source_preflight.json"
  curl -fsS --max-time 10 http://127.0.0.1:30122/health > "$LOG/reference_health.json"
  MANIFEST_SHA=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$SOURCE/summary.json")
fi
DEVS=(); for i in $(seq 0 15); do DEVS+=(--device "/dev/davinci$i"); done
DEVS+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
MOUNTS=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/queue_schedule:/var/queue_schedule -v "$MODEL:/models/glm-q:ro" -v "$LOG:/exp-logs")
if [[ $ROLE == router ]]; then
  printf '%s\n' 'router: 23:18423 prefill -> 25:18423 decode' > "$LOG/router.txt"
  sudo -n docker run -d --name "$NAME" --label experiment=eplb_pd_agentx_20260928 --net host --ipc host "${MOUNTS[@]}" --entrypoint bash "$IMAGE" -lc 'export TORCH_DEVICE_BACKEND_AUTOLOAD=0; exec python3 -m sglang_router.launch_router --pd-disaggregation --policy round_robin --prefill http://10.120.72.23:18423 18998 --decode http://10.120.72.25:18423 --host 0.0.0.0 --port 18623 --prometheus-port 18723'
  exit
fi
MOUNTS+=(-v "$ROOT/patches/readonly_weight_source:/readonly-loader:ro" -v "$SOURCE:/readonly-weight-source:ro")
for f in "$ROOT"/patches/pr32209/python/sglang/srt/{disaggregation/decode.py,managers/schedule_batch.py,managers/scheduler_components/dp_attn.py,speculative/base_spec_worker.py,speculative/eagle_worker_v2.py,speculative/eagle_draft_cuda_graph_runner.py,model_executor/forward_batch_info.py,layers/attention/dsa/dsa_indexer.py,layers/attention/dsa_backend.py}; do
  [[ -f $f ]] || { echo "missing $f" >&2; exit 1; }
  MOUNTS+=(-v "$f:/sgl-workspace/sglang/${f#"$ROOT/patches/pr32209/"}:ro")
done
# Versioned overlays only; container image and model files remain read-only.
MOUNTS+=(-v "$ROOT/patches/eplb/expert_location_updater_lazy_send.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro")
MOUNTS+=(-v "$ROOT/patches/eplb/torch_npu_distributed_c10d_patched.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro")
MOUNTS+=(-v "$ROOT/patches/full_window/expert_distribution_full_window_cpu_gloo.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro")
if [[ $ROLE == prefill ]]; then
  DIST='--tp-size 16 --dp-size 4 --nnodes 1 --node-rank 0'
  MOE='--moe-a2a-backend deepep --deepep-mode normal'
  OTHER='--enable-dp-attention --enable-dp-lm-head --disaggregation-mode prefill --disaggregation-transfer-backend ascend --disaggregation-bootstrap-port 18998 --disable-cuda-graph --max-prefill-tokens 180000 --max-running-requests 4 --mem-fraction-static ${PF_MEM:-0.92}'
  REBALANCE=${PREFILL_REBALANCE:-32}
  RECORDER_BUFFER=${PREFILL_RECORDER_BUFFER:-32}
  HCCL_BUF=${PREFILL_HCCL_BUFFSIZE:-1024}
else
  DIST="--tp-size 16 --dp-size 4 --ep-size 16 --nnodes 1 --node-rank 0 --dist-timeout 3600"
  MOE='--moe-a2a-backend deepep --deepep-mode low_latency'
  OTHER='--enable-dp-attention --disaggregation-mode decode --disaggregation-transfer-backend ascend --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32 --max-prefill-tokens 131072 --max-running-requests 128 --mem-fraction-static ${DECODE_MEM:-0.92}'
  REBALANCE=${DECODE_REBALANCE:-1000}
  RECORDER_BUFFER=${DECODE_RECORDER_BUFFER:-128}
  HCCL_BUF=${DECODE_HCCL_BUFFSIZE:-1024}
fi
EPLB_ARGS=''; [[ $EPLB == 1 ]] && EPLB_ARGS='--enable-eplb'
LAUNCH="set -eo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
export HOME=/tmp/glm52-home; mkdir -p \"\$HOME\"
export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4 SGLANG_HOST_IP=$IP HCCL_IF_IP=$IP
export HCCL_BUFFSIZE=$HCCL_BUF HCCL_OP_EXPANSION_MODE=AIV
export STREAMS_PER_DEVICE=32 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024 DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32 TASK_QUEUE_ENABLE=0
export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0 SGLANG_LOG_EXPERT_LOCATION_METADATA=1
export SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE=1
export SGLANG_DUMP_EPLB_WINDOWS=1 SGLANG_EPLB_STAT_CPU_ALLREDUCE=1 SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/exp-logs/eplb_windows
export ASCEND_MF_STORE_URL=tcp://10.120.72.23:24727 ASCEND_MF_TRANSFER_PROTOCOL=sdma
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
export SGLANG_READONLY_WEIGHT_MANIFEST=/readonly-weight-source/manifest.json
export SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=$MANIFEST_SHA
export SGLANG_READONLY_WEIGHT_BASE=/models/glm-q SGLANG_READONLY_WEIGHT_AUDIT_DIR=/exp-logs/weight_audit
export SGLANG_READONLY_WEIGHT_REFERENCE_URL=http://127.0.0.1:30122
export PYTHONPATH=/readonly-loader:\${PYTHONPATH:-}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
exec python3 -m sglang.launch_server \
  $DIST --model-path /models/glm-q --served-model-name glm-5.2 \
  --host 0.0.0.0 --port 18423 --device npu --attention-backend ascend \
  --dtype bfloat16 --quantization modelslim --trust-remote-code \
  $MOE --context-length 131072 --chunked-prefill-size 16384 \
  --disable-shared-experts-fusion --load-balance-method round_robin \
  --ep-num-redundant-experts $REDUNDANT $EPLB_ARGS --eplb-rebalance-num-iterations $REBALANCE \
  --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size $RECORDER_BUFFER \
  --tokenizer-worker-num 4 --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \
  $OTHER --watchdog-timeout 9000"
printf '%s\n' "$LAUNCH" > "$LOG/launch.sh"
printf 'node=%s\nrole=%s\nimage=%s\nmodel=%s\neplb=%s\nredundant=%s\nrebalance=%s\nrecorder_buffer=%s\nhccl_buffsize=%s\n' "$NODE" "$ROLE" "$EXPECTED" "$MODEL" "$EPLB" "$REDUNDANT" "$REBALANCE" "$RECORDER_BUFFER" "$HCCL_BUF" > "$LOG/config.txt"
sudo -n docker run -d --name "$NAME" --label experiment=eplb_pd_agentx_20260928 --label group="$RUN" --label role="$ROLE" --net host --ipc host --shm-size 16g --pids-limit -1 "${DEVS[@]}" "${MOUNTS[@]}" --entrypoint bash "$IMAGE" -c "$LAUNCH"
