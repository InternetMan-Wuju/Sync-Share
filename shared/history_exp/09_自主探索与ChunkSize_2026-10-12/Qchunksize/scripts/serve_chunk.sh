#!/usr/bin/env bash
# 任务2：Chunk Size 单机服务起停（MTP 关；DeepEP auto + Decode Graph 开 = 快路径）
# 用法: CHUNK=4096 ./serve_chunk.sh {start <run_id>|status|stop}
# 只在"目标机"本地执行（用 sudo -n docker）。除 CHUNK 外所有开关固定。
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Qchunksize
IMG="${IMG:-quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-30080}"
MEM="${MEM:-0.88}"
CTX="${CTX:-131072}"
MAXPREF="${MAXPREF:-32768}"
MRR="${MRR:-16}"
CHUNK="${CHUNK:-16384}"          # <<< 唯一自变量（必须 <= MAXPREF）
TP="${TP:-16}"; EP="${EP:-16}"; DP="${DP:-1}"
DEEPEP="${DEEPEP:-auto}"         # auto: decode 走 low-latency 且 Graph 可开
GRAPH="${GRAPH:-on}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8}"
MAXBS_DECODE="${MAXBS_DECODE:-8}"
HCCL_BUF="${HCCL_BUF:-1000}"
DISPATCH_TOKENS="${DISPATCH_TOKENS:-64}"   # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
NAME="${NAME:-qchunk}"
LOGDIR_BASE="${LOGDIR_BASE:-/data_lib/cyt/qchunk}"
LOADER=""                        # 新镜像不要挂 v0.5.16 loader 补丁

graph_flags(){ if [ "$GRAPH" = on ]; then echo "--cuda-graph-backend-decode full --cuda-graph-max-bs-decode $MAXBS_DECODE --cuda-graph-bs-decode $GRAPH_BS"; else echo "--disable-cuda-graph"; fi; }

start(){
  local run_id="$1"; local logdir="$LOGDIR_BASE/$run_id"; mkdir -p "$logdir"
  local devs=""; for i in $(seq 0 $((TP-1))); do devs="$devs --device /dev/davinci$i"; done
  devs="$devs --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
  local launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=$HCCL_BUF
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$DISPATCH_TOKENS
    export SGLANG_ENABLE_OVERPLAN_PLAN_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 STREAMS_PER_DEVICE=32
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --tp-size $TP --dp-size $DP --ep-size $EP --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --moe-a2a-backend deepep --deepep-mode $DEEPEP \\
      --context-length $CTX --chunked-prefill-size $CHUNK --max-prefill-tokens $MAXPREF \\
      --max-running-requests $MRR --mem-fraction-static $MEM --disable-shared-experts-fusion \\
      $(graph_flags) \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  printf 'CHUNK=%s MEM=%s MAXPREF=%s GRAPH=%s GRAPH_BS=%s DISPATCH=%s HCCL_BUF=%s\n' "$CHUNK" "$MEM" "$MAXPREF" "$GRAPH" "$GRAPH_BS" "$DISPATCH_TOKENS" "$HCCL_BUF" > "$logdir/config.txt"
  if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
    echo "容器 $NAME 已存在，保留它；请指定新的 NAME。" >&2
    return 1
  fi
  sudo -n docker run -d --name "$NAME" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=qchunk --label chunk="$CHUNK" \
    $devs \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    --entrypoint bash "$IMG" -lc "$launch" >/dev/null
  echo "started chunk=$CHUNK img=$IMG port=$PORT log=$logdir/server.log run_id=$run_id"
}
case "${1:-}" in
  start) start "${2:-$(date +%Y%m%d_%H%M%S)}" ;;
  status) sudo -n docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -sS -o /dev/null -w "health=%{http_code}\n" --max-time 20 "http://127.0.0.1:$PORT/health" ;;
  stop) sudo -n docker stop "$NAME" >/dev/null 2>&1 || true; echo stopped ;;
  *) echo "usage: CHUNK=4096 $0 {start <run_id>|status|stop}" >&2; exit 2 ;;
esac
