#!/usr/bin/env bash
# 单机(node23) MTP 服务起停 —— 配置照 2026-09-28 那轮"已验证可行"的 MTP 实验
# 用法: MTP=on|off IMG=<镜像> ./serve_mtp_q1.sh {start <run_id>|status|stop}
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
LOG_ROOT="${LOG_ROOT:-$DIR/logs}"
IMG="${IMG:-quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-30080}"
MTP="${MTP:-on}"
GRAPH="${GRAPH:-off}"
DEEPEP="${DEEPEP:-normal}"
BACKEND="${BACKEND:-deepep}"   # deepep | none（none = 关 DeepEP，走路线甲）
EP="${EP:-16}"
DP="${DP:-1}"
DPATTN="${DPATTN:-0}"   # 1 = 开 dp-attention/dp-lm-head
HCCL_BUF="${HCCL_BUF:-650}"
MEM="${MEM:-0.88}"
MAXPREF="${MAXPREF:-32768}"   # 文档默认 32768（9-28 用的 65536 会抬高 prefill 激活峰值）
CTX="${CTX:-131072}"
MRR="${MRR:-8}"
NAME="${NAME:-q1-mtp-23}"
LOADER="${LOADER-$DIR/scripts/load_model_utils_v0516.py}"   # 显式给空串=不挂 loader 补丁

graph_flags(){ if [ "$GRAPH" = on ]; then echo "--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 8 16 24 32"; else echo "--disable-cuda-graph"; fi; }

attn_flags(){ [ "$DPATTN" = 1 ] && echo "--enable-dp-attention --enable-dp-lm-head"; }
backend_flags(){ if [ "$BACKEND" = none ]; then echo "--moe-a2a-backend none --ep-dispatch-algorithm dynamic"; else echo "--moe-a2a-backend deepep --deepep-mode $DEEPEP"; fi; }

spec_flags(){ [ "$MTP" = on ] && echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"; }

start(){
  local run_id="$1"; local logdir="$LOG_ROOT/$run_id"; mkdir -p "$logdir"
  local LOADER_MOUNT=""
  [ -n "$LOADER" ] && LOADER_MOUNT="-v $LOADER:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro"
  local devs=""; for i in $(seq 0 15); do devs="$devs --device /dev/davinci$i"; done
  devs="$devs --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
  local launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=$HCCL_BUF
    export SGLANG_ENABLE_OVERPLAN_PLAN_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1 STREAMS_PER_DEVICE=32
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --tp-size 16 --dp-size $DP --ep-size $EP --nnodes 1 --node-rank 0 --dist-timeout 3600 \\
      $(attn_flags) \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --host 0.0.0.0 --port $PORT --device npu --attention-backend ascend \\
      --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      $(backend_flags) \\
      --context-length $CTX --chunked-prefill-size 16384 --max-prefill-tokens $MAXPREF \\
      --max-running-requests $MRR --mem-fraction-static $MEM --disable-shared-experts-fusion \\
      $(graph_flags) \\
      --enable-cache-report --enable-metrics --enable-metrics-for-all-schedulers \\
      --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      $(spec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
    echo "容器 $NAME 已存在，保留它；请用 NAME=新名称启动新配置。" >&2
    return 1
  fi
  sudo -n docker run -d --name "$NAME" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=q1mtp --label mtp="$MTP" \
    $devs \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" $LOADER_MOUNT \
    --entrypoint bash "$IMG" -lc "$launch" >/dev/null
  echo "started mtp=$MTP img=$IMG port=$PORT log=$logdir/server.log run_id=$run_id"
}
case "${1:-}" in
  start) start "${2:-$(date +%Y%m%d_%H%M%S)}" ;;
  status) sudo -n docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -sS -o /dev/null -w "health=%{http_code}\n" --max-time 20 "http://127.0.0.1:$PORT/health" ;;
  stop) sudo -n docker stop "$NAME" >/dev/null 2>&1 || true; echo stopped ;;
  *) echo "usage: MTP=on|off IMG=... $0 {start <run_id>|status|stop}" >&2; exit 2 ;;
esac
