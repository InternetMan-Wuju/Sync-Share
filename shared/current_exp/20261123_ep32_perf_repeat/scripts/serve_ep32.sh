#!/usr/bin/env bash
set -euo pipefail
ROOT="${DEPLOY_ROOT:-/data_lib/cyt/global_k_ep32}"
IMG="${IMG:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
PORT="${PORT:-18023}"
MODE="${MODE:-off}"
OBSERVE="${OBSERVE:-0}"
KCONFIG="${KCONFIG:-$ROOT/control/k1_ep32.json}"
LAYOUT="${LAYOUT:-$ROOT/layouts/eplb32_static.json}"
PATCHDIR="${PATCHDIR:-$ROOT/patches}"
MEM="${MEM:-0.85}"
CTX="${CTX:-65536}"
MAXPREF="${MAXPREF:-32768}"
CHUNKED="${CHUNKED:-8192}"
SSH=(-T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
name(){ echo "cyt-global-k-ep32-$1"; }
up_one(){
 local node="$1" rank="$2" run="$3"; local ip="10.120.72.$node"; local logdir="$ROOT/logs/ep32_${run}/node$node"; mkdir -p "$logdir"
 local devs=""; for i in $(seq 0 15); do devs="$devs --device /dev/davinci$i"; done; devs="$devs --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
 local eplb_flags="" extra_mount="" pyenv="" patch_env="" mode="$4"
 case "$mode" in
  off) eplb_flags="--ep-num-redundant-experts 0" ;;
  eplb_static) eplb_flags="--ep-num-redundant-experts 32 --ep-dispatch-algorithm static --init-expert-location /runexp/layouts/$(basename "$LAYOUT")"; extra_mount="-v $LAYOUT:/runexp/layouts/$(basename "$LAYOUT"):ro" ;;
  eplb_online) eplb_flags="--enable-eplb --ep-num-redundant-experts 32 --eplb-algorithm deepseek --eplb-rebalance-num-iterations 2 --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 1"; extra_mount="-v $PATCHDIR/expert_location_updater_v0520.py:/sgl-workspace/sglang/python/sglang/srt/eplb/expert_location_updater.py:ro -v $PATCHDIR/torch_npu_distributed_c10d_v0520.py:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/distributed/distributed_c10d.py:ro"; patch_env="export SGLANG_LOG_EXPERT_LOCATION_METADATA=1" ;;
  k1|k2) eplb_flags="--ep-num-redundant-experts 0"; extra_mount="-v $ROOT/plugin:/runexp/plugin:ro -v $KCONFIG:/runexp/control/global_k.json:ro"; pyenv="export PYTHONPATH=/runexp/plugin:/sgl-workspace/sglang/python; export CYT_GLOBAL_K_CONFIG=/runexp/control/global_k.json; export CYT_K_LOG_DIR=/exp-logs/k_service" ;;
  *) echo "unknown mode $mode" >&2; return 2 ;;
 esac
 if [[ "$OBSERVE" == 1 ]]; then extra_mount="$extra_mount -v $ROOT/plugin:/runexp/plugin:ro"; pyenv="$pyenv export PYTHONPATH=/runexp/plugin:/sgl-workspace/sglang/python; export CYT_LAYER49_OBSERVER=1; export CYT_OBSERVER_DIR=/exp-logs/observer; export CYT_OBSERVER_CAPTURE_HIDDEN=0"; fi
 local launch="set -eo pipefail
 unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
 export PYTHONDONTWRITEBYTECODE=1 SGLANG_SET_CPU_AFFINITY=1
 export GLOO_SOCKET_IFNAME=bond4 HCCL_SOCKET_IFNAME=bond4
 export HCCL_IF_BASE_PORT=50000 HCCL_BUFFSIZE=1000
 $pyenv
 $patch_env
 export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
 export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
 export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
 export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
 export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 STREAMS_PER_DEVICE=32
 export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /usr/local/Ascend/nnal/atb/set_env.sh
 python3 -m sglang.launch_server \
   --model-path /models/glm-q --served-model-name glm-5.2 --host 0.0.0.0 --port $PORT \
   --device npu --attention-backend ascend --dtype bfloat16 --quantization modelslim --trust-remote-code \
   --tp-size 32 --ep-size 32 --dp-size 1 --nnodes 2 --node-rank $rank \
   --dist-init-addr 10.120.72.23:5000 --dist-timeout 1800 \
   --moe-a2a-backend deepep --deepep-mode normal --disable-shared-experts-fusion \
   $eplb_flags \
   --context-length $CTX --chunked-prefill-size $CHUNKED --max-prefill-tokens $MAXPREF --max-total-tokens $CTX \
   --max-running-requests 1 --mem-fraction-static $MEM \
   --disable-prefill-cuda-graph --disable-decode-cuda-graph --disable-overlap-schedule --disable-radix-cache \
   --enable-cache-report --enable-metrics --watchdog-timeout 9000 \
   2>&1 | tee /exp-logs/server.log"
 printf '%s\n' "$launch" > "$logdir/launch.sh"
 sudo -n docker rm -f "$(name "$node")" >/dev/null 2>&1 || true
 sudo -n docker run -d --name "$(name "$node")" --net host --ipc host --shm-size 16g --pids-limit -1 \
   --label owner=chengyetai --label experiment=global-k-ep32 --label rank="$rank" \
   $devs \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
   -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /var/queue_schedule:/var/queue_schedule \
   -v "$MODEL:/models/glm-q:ro" -v "$logdir:/exp-logs" $extra_mount \
   --entrypoint bash "$IMG" -lc "$launch" >/dev/null
 echo "started node=$node rank=$rank"
}
case "${1:-}" in
 start)
  mode="${2:-off}"; run="${3:-$(date +%Y%m%d_%H%M%S)}"
  for r in 1 0; do n=$([ "$r" = 1 ] && echo 25 || echo 23); echo "--> node=$n rank=$r"; ssh "${SSH[@]}" "10.120.72.$n" "mkdir -p '$ROOT/scripts' '$ROOT/plugin' '$ROOT/control' '$ROOT/layouts' '$ROOT/patches'"; scp -q -o LogLevel=ERROR "$0" "10.120.72.$n:$ROOT/scripts/serve_ep32.sh"; scp -q -r -o LogLevel=ERROR plugin/* "10.120.72.$n:$ROOT/plugin/"; scp -q -r -o LogLevel=ERROR control/* "10.120.72.$n:$ROOT/control/"; scp -q -r -o LogLevel=ERROR layouts/* "10.120.72.$n:$ROOT/layouts/"; scp -q -r -o LogLevel=ERROR patches/* "10.120.72.$n:$ROOT/patches/"; ssh "${SSH[@]}" "10.120.72.$n" "DEPLOY_ROOT='$ROOT' IMG='$IMG' MODEL='$MODEL' PORT='$PORT' MEM='$MEM' CTX='$CTX' MAXPREF='$MAXPREF' CHUNKED='$CHUNKED' MODE='$mode' OBSERVE='$OBSERVE' KCONFIG='$ROOT/control/$(basename "$KCONFIG")' LAYOUT='$ROOT/layouts/$(basename "$LAYOUT")' bash '$ROOT/scripts/serve_ep32.sh' __up '$n' '$r' '$run' '$mode'"; done ;;
 status)
  for n in 23 25; do echo -n "npu$n: "; ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker ps --filter name=^/$(name $n)$ --format '{{.Status}}'"; done
  curl -sS -o /dev/null -w 'health=%{http_code}\n' --max-time 10 "http://127.0.0.1:$PORT/health" || true ;;
 stop)
  for n in 23 25; do ssh "${SSH[@]}" "10.120.72.$n" "sudo -n docker rm -f $(name $n)" >/dev/null 2>&1 || true; done; echo stopped ;;
 __up) shift; up_one "$1" "$2" "$3" "$4" ;;
 *) echo "usage: $0 {start [run]|status|stop}" >&2; exit 2 ;;
esac
