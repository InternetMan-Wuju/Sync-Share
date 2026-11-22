#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZE="${1:-8k}"
REPS="${2:-10}"
RUN_ID="${3:-$(date +%Y%m%d_%H%M%S)}"
COPY_REPS="${4:-5}"
IMG="${IMG:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
REUSE="${REUSE:-/home/chengyetai/workspace/Share/shared/current_exp/20261123_global_k_adapter_repair/scripts}"
OBS="${OBS:-$ROOT/logs/off_off_observe_all_v4/observer}"
OUTDIR="$ROOT/artifacts/replay16_${SIZE}_${RUN_ID}"
mkdir -p "$OUTDIR"
DEV=(); for i in $(seq 0 15); do DEV+=(--device "/dev/davinci$i"); done
DEV+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
sudo -n docker rm -f cyt-threeway-replay16 >/dev/null 2>&1 || true
sudo -n docker run --rm --name cyt-threeway-replay16 --net host --ipc host --shm-size 16g --pids-limit -1 --security-opt seccomp=unconfined \
  "${DEV[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v "$MODEL:/models/glm-q:ro" -v "$ROOT:/runexp:rw" -v "$REUSE:/runexp/reuse:ro" \
  -e PYTHONPATH=/runexp/scripts:/runexp/reuse -e GLM_MODEL=/models/glm-q \
  -e DEEP_USE_MODE=alltoall -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 \
  -e DEEP_NORMAL_MODE_USE_INT8_QUANT=1 -e DEEPEP_NORMAL_LONG_SEQ_ROUND=72 -e DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024 \
  -e DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e GLOO_SOCKET_IFNAME=bond4 -e HCCL_SOCKET_IFNAME=bond4 -e HCCL_BUFFSIZE=1000 \
  --entrypoint bash "$IMG" -lc \
  "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && \
   torchrun --standalone --nproc_per_node=16 /runexp/scripts/replay_ep16.py \
     --observer-dir ${OBS/#$ROOT//runexp} \
     --layout /runexp/layouts/eplb16_static.json \
     --layout-summary /runexp/artifacts/calibration/eplb16_layer49_summary.json \
     --size $SIZE --reps $REPS --copy-reps $COPY_REPS --out /runexp/artifacts/replay16_${SIZE}_${RUN_ID}" \
  2>&1 | tee "$OUTDIR/replay.log"
