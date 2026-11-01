#!/usr/bin/env bash
set -euo pipefail

# Start one 2-node GLM-5.2 W8A8 service across npu1-23 and npu1-25.
# This reserves both nodes for the service. It deliberately uses a distinct
# container name and never removes or replaces any pre-existing container.

HOST0="10.120.72.23"
HOST1="10.120.72.25"
IMAGE="sglang-glm52-pd:fe694986"
IMAGE_ID="sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d"
MODEL="/mnt/raid/user/data/models/glm-qq"
CONTAINER="chunk-mtp-glm52-tp32"
PORT="30080"
CHUNK_SIZE="${CHUNK_SIZE:-8192}"

remote_launch() {
  local rank="$1" node_ip="$2"
  if sudo -n docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "container already exists on $(hostname): $CONTAINER" >&2
    return 20
  fi
  if ! sudo -n docker image inspect "$IMAGE" --format '{{.Id}}' | grep -Fxq "$IMAGE_ID"; then
    echo "image ID mismatch on $(hostname)" >&2
    return 21
  fi
  if [ ! -d "$MODEL" ]; then
    echo "model directory not found on $(hostname): $MODEL" >&2
    return 22
  fi
  if ss -lnt | grep -Eq ":${PORT}[[:space:]]"; then
    echo "HTTP port ${PORT} already has a listener on $(hostname)" >&2
    return 23
  fi

  local devices=""
  for i in $(seq 0 15); do devices+=" --device /dev/davinci${i}"; done
  devices+=" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"

  local run_cmd
  run_cmd="set -eo pipefail
    unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
    export HOME=/tmp/chunk-mtp-glm52-home
    mkdir -p \"\$HOME\"
    export PYTHONDONTWRITEBYTECODE=1
    export SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4
    export HCCL_SOCKET_IFNAME=bond4
    export STREAMS_PER_DEVICE=32
    export HCCL_OP_EXPANSION_MODE=AIV
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_NPU_USE_MULTI_STREAM=1
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_HOST_IP=${node_ip}
    export HCCL_IF_IP=${node_ip}
    export HCCL_BUFFSIZE=650
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TRANSFORMERS_VERBOSITY=error
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --tp-size 32 --nnodes 2 --node-rank ${rank} \\
      --dist-init-addr ${HOST0}:5000 --dist-timeout 1800 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port ${PORT} \\
      --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode auto \\
      --context-length 131072 \\
      --chunked-prefill-size ${CHUNK_SIZE} --max-prefill-tokens 131072 \\
      --max-running-requests 32 --mem-fraction-static 0.80 \\
      --disable-shared-experts-fusion --disable-cuda-graph \\
      --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      2>&1 | tee /tmp/${CONTAINER}_\$(hostname).log"

  # Mount the model read-only; use a unique container name so existing jobs stay intact.
  local cid
  cid=$(sudo -n docker run -d \
    --name "$CONTAINER" --net=host --ipc=host --shm-size=16g --pids-limit=-1 \
    $devices \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" \
    --entrypoint bash "$IMAGE" -lc "$run_cmd")
  echo "$cid"
}

remote_remove() {
  local cid="$1"
  sudo -n docker rm -f "$cid"
}

remote_status() {
  echo "node=$(hostname)"
  sudo -n docker ps -a --filter "name=^/${CONTAINER}$" --no-trunc --format '{{.ID}} {{.Names}} {{.Status}} {{.Image}}'
  sudo -n docker logs --tail 25 "$CONTAINER" 2>&1 || true
  npu-smi info | grep -E 'HBM-Usage|/ 65536|No running processes' || true
}

if [[ "${1:-start}" == "__remote_start" ]]; then
  remote_launch "$2" "$3"
  exit
elif [[ "${1:-}" == "__remote_remove" ]]; then
  remote_remove "$2"
  exit
elif [[ "${1:-}" == "__remote_status" ]]; then
  remote_status
  exit
fi

action="${1:-start}"
case "$action" in
  start)
    started_host=""
    started_id=""
    for spec in "${HOST0}:0" "${HOST1}:1"; do
      host="${spec%:*}"
      rank="${spec##*:}"
      ip="${host##*.}"
      node_ip="10.120.72.${ip}"
      echo "[start] host=${host} rank=${rank} image=${IMAGE_ID} chunk=${CHUNK_SIZE}"
      if ! cid=$(ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$host" \
        "bash -s -- __remote_start ${rank} ${node_ip}" < "$0"); then
        echo "[error] start failed on ${host}; removing only the container created by this invocation, if any" >&2
        if [[ -n "$started_id" ]]; then
          ssh -T -o BatchMode=yes "$started_host" \
            "bash -s -- __remote_remove ${started_id}" < "$0" || true
        fi
        exit 1
      fi
      started_host="$host"
      started_id="$cid"
      echo "[started] host=${host} container_id=${cid}"
    done
    echo "[submitted] endpoint=http://${HOST0}:${PORT} model=glm-5.2"
    ;;
  status)
    for host in "$HOST0" "$HOST1"; do
      echo "===== ${host} ====="
      ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$host" \
        "bash -s -- __remote_status" < "$0" || true
    done
    ;;
  stop)
    for host in "$HOST1" "$HOST0"; do
      echo "[stop] ${host} container=${CONTAINER}"
      ssh -T -o BatchMode=yes -o ConnectTimeout=8 "$host" \
        "sudo -n docker rm -f ${CONTAINER}" || true
    done
    ;;
  *)
    echo "usage: $0 [start|status|stop]" >&2
    exit 2
    ;;
esac
