#!/usr/bin/env bash
# 2 机共置 MTP 服务（node23=rank0 master, node20=rank1）
# 用法: MTP=on ./serve_mtp_2n.sh {start <run_id>|status|stop}
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
IMG="${IMG:-quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-30080}"
MTP="${MTP:-on}"
MEM="${MEM:-0.88}"
CTX="${CTX:-65536}"
MAXPREF="${MAXPREF:-32768}"
MRR="${MRR:-8}"
HCCL_BUF="${HCCL_BUF:-1024}"
MASTER_IP=10.120.72.23
SSH=(-T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
name(){ echo "q1mtp2n-$1"; }
spec_flags(){ [ "$MTP" = on ] && echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"; }
remote_up(){
  local node="$1" rank="$2" run_id="$3"
  local logdir="/data_lib/cyt/q1mtp2n/$run_id/node_$node"; mkdir -p "$logdir"
  local devs=""; for i in $(seq 0 15); do devs="$devs --device /dev/davinci$i"; done
  devs="$devs --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
  local launch="set -eo pipefail
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
    export HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=$HCCL_BUF
    export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 STREAMS_PER_DEVICE=32
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
    export HF_ENDPOINT=https://hf-mirror.com
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    python3 -m sglang.launch_server \\
      --model-path /models/glm-q --served-model-name glm-5.2 --host 0.0.0.0 --port $PORT \\
      --device npu --attention-backend ascend --dtype bfloat16 --quantization modelslim --trust-remote-code \\
      --tp-size 32 --dp-size 1 --ep-size 32 --nnodes 2 --node-rank $rank \\
      --dist-init-addr $MASTER_IP:5000 --dist-timeout 1800 \\
      --moe-a2a-backend deepep --deepep-mode normal \\
      --context-length $CTX --chunked-prefill-size 16384 --max-prefill-tokens $MAXPREF \\
      --max-running-requests $MRR --tokenizer-worker-num 4 --watchdog-timeout 9000 \\
      --mem-fraction-static $MEM --disable-shared-experts-fusion --disable-cuda-graph \\
      --enable-cache-report --enable-metrics \\
      $(spec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  if sudo -n docker inspect "$(name "$node")" >/dev/null 2>&1; then
    echo "容器 $(name "$node") 已存在，保留它；未启动新服务。" >&2
    return 1
  fi
  sudo -n docker run -d --name "$(name "$node")" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=q1mtp2n --label rank="$rank" \
    $devs \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    --entrypoint bash "$IMG" -lc "$launch" >/dev/null
  echo "started node=$node rank=$rank"
}
case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"; echo "$run_id" > "$DIR/run2n.txt"
    for n in 20 23; do timeout 30 ssh "${SSH[@]}" "10.120.72.$n" "mkdir -p '$DIR/scripts'" 2>/dev/null; done
    scp -q -o LogLevel=ERROR "$DIR/scripts/serve_mtp_2n.sh" "10.120.72.20:$DIR/scripts/" 2>/dev/null
    for r in 1 0; do
      n=$([ "$r" = 1 ] && echo 20 || echo 23)
      echo "--> up node=$n rank=$r"
      ssh "${SSH[@]}" "10.120.72.$n" "MTP='$MTP' IMG='$IMG' MODEL='$MODEL' PORT='$PORT' MEM='$MEM' CTX='$CTX' MAXPREF='$MAXPREF' MRR='$MRR' HCCL_BUF='$HCCL_BUF' bash '$DIR/scripts/serve_mtp_2n.sh' __up '$n' '$r' '$run_id'"
    done
    echo "run_id=$run_id  (master 日志: /data_lib/cyt/q1mtp2n/$run_id/node_23/server.log)" ;;
  status) for n in 23 20; do echo -n "npu$n: "; timeout 20 ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker ps --filter name=$(name $n) --format '{{.Names}} {{.Status}}'" 2>/dev/null; echo; done
          curl -sS -o /dev/null -w "health=%{http_code}\n" --max-time 15 "http://127.0.0.1:$PORT/health" ;;
  stop) for n in 23 20; do timeout 20 ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker stop $(name $n)" >/dev/null 2>&1 || true; done; echo stopped ;;
  __up) shift; remote_up "$1" "$2" "$3" ;;
  *) echo "usage: MTP=on|off $0 {start <run_id>|status|stop}"; exit 2 ;;
esac
