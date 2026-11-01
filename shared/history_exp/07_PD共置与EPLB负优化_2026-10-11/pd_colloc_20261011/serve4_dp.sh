#!/usr/bin/env bash
set -euo pipefail
# 4-node PD-collocated (united) service WITH dp-attention.
# Recipe borrowed from bench3 SOP (v0.5.18 + deepep auto + decode graph list + MEM0.65),
# ONLY added variable vs. the proven TP64 baseline: --dp-size 4 --enable-dp-attention.
DIR=/home/chengyetai/workspace/current_exp/pd_colloc_20261011
IMG=quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-30000}"
MEM="${MEM:-0.65}"
CPS="${CPS:-65536}"        # 注意: 开 dp-attn 后源码会 //dp_size → 有效 16384
MRR="${MRR:-32}"
LL_CAP="${LL_CAP:-64}"
HCCL_BUF="${HCCL_BUF:-1000}"
CGBS="${CGBS:-1 2 4 8 16 32}"
SPEC="${SPEC:-off}"
EPLB="${EPLB:-0}"
LOGDIR_ROOT="${LOGDIR_ROOT:-$DIR/logs}"   # 落盘根目录（可指到 /data_lib/cyt）
REC="${REC:-$EPLB}"        # 采集开关：默认跟 EPLB 保持一致（REC=1 才会挂采集补丁 + recorder-mode）
REBAL="${REBAL:-2000}"
RECD_BUF="${RECD_BUF:-1024}"
REDUNDANT="${REDUNDANT:-0}"
NODES=(19 20 23 25)        # index = node_rank; rank0 = 19 (master, holds the recorder dumps)
MASTER_IP=10.120.72.19
SSH=(-T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
eplb_flags(){ [[ "$EPLB" == 1 ]] || return 0; echo "--enable-eplb --ep-num-redundant-experts $REDUNDANT --eplb-rebalance-num-iterations $REBAL"; }
rec_flags(){ [[ "$REC" == 1 ]] || return 0; echo "--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size $RECD_BUF"; }
spec_flags(){ [[ "$SPEC" == on ]] || return 0; echo "--speculative-algorithm NEXTN --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"; }
name(){ echo "cxy-pd4-$1"; }

remote_up() {
  local node="$1" rank="$2" run_id="$3"
  local logdir="$LOGDIR_ROOT/$run_id/node_$node"; mkdir -p "$logdir/eplb_windows"
  local ip="10.120.72.$node"
  local devs=""; for i in $(seq 0 15); do devs="$devs --device /dev/davinci$i"; done
  devs="$devs --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
  local EPLBEXTRA=""
  [[ "$REC" == 1 ]] && EPLBEXTRA="-v $DIR/patches/expert_distribution_phase_window.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py:ro"
  sudo -n docker rm -f "$(name "$node")" >/dev/null 2>&1 || true
  local launch="set -eo pipefail
    unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
    export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
    export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4 STREAMS_PER_DEVICE=32
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export SGLANG_HOST_IP=$ip HCCL_IF_IP=$ip HCCL_BUFFSIZE=$HCCL_BUF HCCL_OP_EXPANSION_MODE=AIV
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$LL_CAP
    export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
    export HF_ENDPOINT=https://hf-mirror.com
    export SGLANG_DUMP_EPLB_WINDOWS=${SGLANG_DUMP_EPLB_WINDOWS:-1}
    export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/exp-logs/eplb_windows
    python3 -m sglang.launch_server \\
      --model-path /models/glm-q --served-model-name glm-5.2 \\
      --attention-backend ascend --device npu --quantization modelslim --trust-remote-code \\
      --tp-size 64 --dp-size 4 --ep-size 64 --enable-dp-attention \\
      --nnodes 4 --node-rank $rank --dist-init-addr $MASTER_IP:5000 --dist-timeout 1800 \\
      --moe-a2a-backend deepep --deepep-mode auto \\
      --context-length 131072 --chunked-prefill-size $CPS --max-prefill-tokens 131072 \\
      --cuda-graph-bs $CGBS --max-running-requests $MRR --mem-fraction-static $MEM \\
      --host 0.0.0.0 --port $PORT --disable-shared-experts-fusion \\
      --enable-cache-report --enable-metrics --watchdog-timeout 9000 \\
      $(spec_flags) \\
      $(eplb_flags) \\
      $(rec_flags) \\
      2>&1 | tee /exp-logs/server.log"
  printf '%s\n' "$launch" > "$logdir/launch.sh"
  sudo -n docker run -d --name "$(name "$node")" --net host --ipc host --shm-size 16g --pids-limit -1 \
    --label owner=chengyetai --label experiment=pd_colloc_dp --label rank="$rank" \
    $devs \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" \
    -v "$DIR/patches/deepep_patched:/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:ro" \
    -v "$DIR/patches/expert_location_updater_patched.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro" \
    -v "$DIR/patches/torch_npu_distributed_c10d_patched.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro" \
    $EPLBEXTRA \
    -v "$DIR/load_model_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro" \
    --entrypoint bash "$IMG" -lc "$launch"
  echo "started node=$node rank=$rank"
}

case "${1:-}" in
  start)
    run_id="${2:-$(date +%Y%m%d_%H%M%S)}"
    for i in 0 1 2 3; do n=${NODES[$i]}
      ssh "${SSH[@]}" "10.120.72.$n" "mkdir -p '$LOGDIR_ROOT/$run_id/node_$n' '$DIR/patches'"
      scp -q -o LogLevel=ERROR "$DIR/serve4_dp.sh" "$DIR/load_model_utils.py" "10.120.72.$n:$DIR/" 2>/dev/null || true
      scp -q -o LogLevel=ERROR "$DIR/patches/deepep_patched" "$DIR/patches/expert_location_updater_patched.py" "$DIR/patches/torch_npu_distributed_c10d_patched.py" "$DIR/patches/expert_distribution_phase_window.py" "10.120.72.$n:$DIR/patches/"
    done
    # rank1..3 first, rank0 last (bench3 order)
    for i in 3 2 1 0; do n=${NODES[$i]}
      echo "--> up node=$n rank=$i"
      ssh "${SSH[@]}" "10.120.72.$n" "SPEC='$SPEC' EPLB='$EPLB' REC='$REC' LOGDIR_ROOT='$LOGDIR_ROOT' REBAL='$REBAL' RECD_BUF='$RECD_BUF' REDUNDANT='$REDUNDANT' MODEL='$MODEL' PORT='$PORT' MEM='$MEM' CPS='$CPS' MRR='$MRR' LL_CAP='$LL_CAP' HCCL_BUF='$HCCL_BUF' CGBS='$CGBS' bash '$DIR/serve4_dp.sh' __up '$n' '$i' '$run_id'"
    done
    echo "run_id=$run_id" ;;
  status) for n in "${NODES[@]}"; do echo -n "npu$n: "; ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker ps -a --filter name=$(name $n) --format '{{.Names}} {{.Status}}'" 2>/dev/null; done
          ssh "${SSH[@]}" "10.120.72.23" "curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 20 http://127.0.0.1:$PORT/health" 2>/dev/null || true ;;
  stop) for n in "${NODES[@]}"; do ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker rm -f $(name $n)" >/dev/null 2>&1 || true; done; echo "stopped" ;;
  __up) remote_up "$2" "$3" "$4" ;;
  *) echo "usage: serve4_dp.sh {start <run_id>|status|stop}" >&2; exit 2 ;;
esac
