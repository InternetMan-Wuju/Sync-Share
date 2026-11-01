#!/bin/bash
set -euo pipefail

# 下一次实验：复现 2026-09-14 的 2P2D 16K/1024 基线，关闭 EPLB 做严格对照。
# 默认只做 preflight；start/matrix/stop 必须显式调用。

DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_SCRIPT="${DIR}/serve_pd.sh"
BENCH_SCRIPT="${DIR}/bench_pd_prefix.py"
BREAKDOWN_SCRIPT="${DIR}/breakdown_ttft.py"
HEAT_SCRIPT="${DIR}/../glm52_ep64_chengyetai/ep_lab/analyze_expert_heat.py"
COMPARE_SCRIPT="${DIR}/compare_eplb_ab.py"

PREFILL0="${PREFILL0:-10.120.72.23}"
PREFILL1="${PREFILL1:-10.120.72.25}"
DECODE0="${DECODE0:-10.120.72.31}"
DECODE1="${DECODE1:-10.120.72.32}"
HOSTS=("${PREFILL0}" "${PREFILL1}" "${DECODE0}" "${DECODE1}")

IMAGE="${IMAGE:-sglang-glm52-pd:fe694986}"
IMAGE_ID="sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d"
MODEL="${MODEL:-/mnt/raid/user/data/models/glm-qq}"
ROUTER_URL="${ROUTER_URL:-http://${PREFILL0}:18623}"
EPLB="${EPLB:-0}"
EXPERT_RECORDER="${EXPERT_RECORDER:-1}"
FULL_WINDOW_DUMP="${FULL_WINDOW_DUMP:-0}"
PF_MEM="${PF_MEM:-0.90}"
DECODE_MEM="${DECODE_MEM:-0.82}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
MODE="eplb_on"
if [ "${EPLB}" = "0" ]; then
  MODE="eplb_off"
fi
RESULTS_ROOT="${RESULTS_ROOT:-${DIR}/results/${RUN_DATE}_pd_ep32_${MODE}_16k}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/home/chengyetai/workspace/current_exp}"
REMOTE_TMP="${REMOTE_TMP:-${REMOTE_WORKDIR}/tmp}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${host}" "$@"
}

require_file() {
  [ -f "$1" ] || die "missing file: $1"
}

wait_health() {
  local name="$1"
  local url="$2"
  local start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1; then
      log "ready: ${name} (${url})"
      return 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "${STARTUP_TIMEOUT}" ]; then
      die "timeout waiting for ${name}: ${url}"
    fi
    sleep 10
  done
}

write_manifest() {
  mkdir -p "${RESULTS_ROOT}"
  {
    echo "created_at=$(date -Iseconds)"
    echo "experiment=pd_ep32_${MODE}_16k"
    echo "baseline=/home/chengyetai/workspace/current_exp/results_final"
    echo "image=${IMAGE}"
    echo "expected_image_id=${IMAGE_ID}"
    echo "model=${MODEL}"
    echo "eplb=${EPLB}"
    echo "expert_recorder=${EXPERT_RECORDER}"
    echo "topology=prefill:${PREFILL0},${PREFILL1};decode:${DECODE0},${DECODE1};router:${PREFILL0}"
    echo "workload=input=16384,output=1024,prefix_ratio=0.9,concurrency=10/20/30,requests=concurrency*4,seed=1024"
    echo "serve_sha256=$(sha256sum "${SERVE_SCRIPT}" | awk '{print $1}')"
    echo "bench_sha256=$(sha256sum "${BENCH_SCRIPT}" | awk '{print $1}')"
    echo "breakdown_sha256=$(sha256sum "${BREAKDOWN_SCRIPT}" | awk '{print $1}')"
    echo "orchestrator_sha256=$(sha256sum "${DIR}/next_experiment.sh" | awk '{print $1}')"
    echo "compare_sha256=$(sha256sum "${COMPARE_SCRIPT}" | awk '{print $1}')"
    for host in "${HOSTS[@]}"; do
      remote "${host}" "printf 'host=%s image_id=%s\n' \"\$(hostname)\" \"\$(sudo docker image inspect '${IMAGE}' --format '{{.Id}}')\""
    done
  } >"${RESULTS_ROOT}/manifest.txt"
}

preflight() {
  require_file "${SERVE_SCRIPT}"
  require_file "${BENCH_SCRIPT}"
  require_file "${BREAKDOWN_SCRIPT}"
  require_file "${HEAT_SCRIPT}"
  require_file "${COMPARE_SCRIPT}"
  python3 -m py_compile "${BENCH_SCRIPT}" "${BREAKDOWN_SCRIPT}" "${COMPARE_SCRIPT}"
  bash -n "${SERVE_SCRIPT}"

  local host actual patch local_patch_sha remote_patch_sha
  local patch_files=(
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/disaggregation/decode.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/managers/schedule_batch.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/managers/scheduler_components/dp_attn.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/speculative/base_spec_worker.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/speculative/eagle_worker_v2.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/model_executor/forward_batch_info.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
    /home/chengyetai/workspace/current_exp/patches/pr32209/python/sglang/srt/layers/attention/dsa_backend.py
  )
  for host in "${HOSTS[@]}"; do
    log "preflight ${host}"
    remote "${host}" "mkdir -p '${REMOTE_WORKDIR}' '${REMOTE_TMP}'"
    remote "${host}" "test -d '${MODEL}'" || die "model missing on ${host}: ${MODEL}"
    actual=$(remote "${host}" "sudo docker image inspect '${IMAGE}' --format '{{.Id}}'")
    [ "${actual}" = "${IMAGE_ID}" ] || die "image mismatch on ${host}: ${actual}"
    for patch in "${patch_files[@]}"; do
      require_file "${patch}"
      local_patch_sha=$(sha256sum "${patch}" | cut -d ' ' -f 1)
      remote "${host}" "test -f '${patch}'" || die "patch missing on ${host}: ${patch}"
      remote_patch_sha=$(remote "${host}" "sha256sum '${patch}' | cut -d ' ' -f 1")
      [ "${remote_patch_sha}" = "${local_patch_sha}" ] \
        || die "patch mismatch on ${host}: ${patch}"
    done
    remote "${host}" "sudo docker ps --format '{{.Names}} {{.Status}}' | grep -E '^pd-' || true"
  done
  write_manifest
  log "preflight passed; manifest=${RESULTS_ROOT}/manifest.txt"
}

sync_scripts() {
  local host
  for host in "${HOSTS[@]}"; do
    remote "${host}" "mkdir -p '${REMOTE_WORKDIR}' '${REMOTE_TMP}'"
    scp "${SSH_OPTS[@]}" "${SERVE_SCRIPT}" \
      "${host}:${REMOTE_WORKDIR}/serve_pd.sh" >/dev/null
  done
}

start_services() {
  preflight
  sync_scripts

  local common
  common="IMG=${IMAGE} MODEL=${MODEL} CTX=131072 LOAD_BALANCE=round_robin SPEC=off PR32209=1 EPLB=${EPLB} EPLB_PATCHES=${EPLB} EXPERT_RECORDER=${EXPERT_RECORDER} FULL_WINDOW_DUMP=${FULL_WINDOW_DUMP}"

  log "starting prefill workers"
  remote "${PREFILL0}" "${common} EP_NUM_REDUNDANT=${PREFILL_EP_NUM_REDUNDANT:-0} EPLB_REBALANCE_ITERATIONS=${PREFILL_EPLB_REBALANCE_ITERATIONS:-1000} MEM=${PF_MEM} MAX_RUNNING=4 PF_CPS=${PF_CPS:-16384} PF_MOE_DENSE_TP=0 bash '${REMOTE_WORKDIR}/serve_pd.sh' prefill 0"
  remote "${PREFILL1}" "${common} EP_NUM_REDUNDANT=${PREFILL_EP_NUM_REDUNDANT:-0} EPLB_REBALANCE_ITERATIONS=${PREFILL_EPLB_REBALANCE_ITERATIONS:-1000} MEM=${PF_MEM} MAX_RUNNING=4 PF_CPS=${PF_CPS:-16384} PF_MOE_DENSE_TP=0 bash '${REMOTE_WORKDIR}/serve_pd.sh' prefill 0"

  log "starting two-node decode"
  remote "${DECODE0}" "${common} EP_NUM_REDUNDANT=${DECODE_EP_NUM_REDUNDANT:-${EP_NUM_REDUNDANT:-0}} EPLB_REBALANCE_ITERATIONS=${DECODE_EPLB_REBALANCE_ITERATIONS:-1000} MEM=${DECODE_MEM} MAX_RUNNING=128 DECODE_NNODES=2 DECODE_TP=32 DECODE_EP=32 DECODE_DP=32 MASTER_IP=${DECODE0} DEEPEP_MODE=low_latency DP_ATTN=on DC_GRAPH=on bash '${REMOTE_WORKDIR}/serve_pd.sh' decode 0"
  remote "${DECODE1}" "${common} EP_NUM_REDUNDANT=${DECODE_EP_NUM_REDUNDANT:-${EP_NUM_REDUNDANT:-0}} EPLB_REBALANCE_ITERATIONS=${DECODE_EPLB_REBALANCE_ITERATIONS:-1000} MEM=${DECODE_MEM} MAX_RUNNING=128 DECODE_NNODES=2 DECODE_TP=32 DECODE_EP=32 DECODE_DP=32 MASTER_IP=${DECODE0} DEEPEP_MODE=low_latency DP_ATTN=on DC_GRAPH=on bash '${REMOTE_WORKDIR}/serve_pd.sh' decode 1"

  wait_health "prefill0" "http://${PREFILL0}:18423"
  wait_health "prefill1" "http://${PREFILL1}:18423"
  wait_health "decode0" "http://${DECODE0}:18423"

  log "starting router"
  remote "${PREFILL0}" "sudo docker rm -f pd-router-glm52 >/dev/null 2>&1 || true; \
    sudo docker run -d --name pd-router-glm52 --net=host --ipc=host \
    -v '${MODEL}:/models/glm-q:ro' --entrypoint bash '${IMAGE}' -lc \
    'export TORCH_DEVICE_BACKEND_AUTOLOAD=0; python3 -m sglang_router.launch_router \
      --pd-disaggregation --policy round_robin \
      --prefill http://${PREFILL0}:18423 18998 \
      --prefill http://${PREFILL1}:18423 18998 \
      --decode http://${DECODE0}:18423 \
      --host 0.0.0.0 --port 18623 --prometheus-port 18723'"
  wait_health "router" "${ROUTER_URL}"
  status_services
}

status_services() {
  local host
  for host in "${HOSTS[@]}"; do
    echo "===== ${host} ====="
    remote "${host}" "sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep -E '^pd-' || true"
  done
  curl -sS --max-time 5 -o /dev/null -w 'router_http=%{http_code}\n' "${ROUTER_URL}/health" || true
}

stop_services() {
  local host
  for host in "${HOSTS[@]}"; do
    log "stopping PD containers on ${host}"
    remote "${host}" "for c in pd-prefill-glm52 pd-decode-glm52 pd-router-glm52; do sudo docker rm -f \"\${c}\" >/dev/null 2>&1 || true; done"
  done
}

smoke() {
  wait_health "router" "${ROUTER_URL}"
  mkdir -p "${RESULTS_ROOT}/smoke"
  python3 "${BENCH_SCRIPT}" \
    --url "${ROUTER_URL}" \
    --input-tokens 16384 \
    --output-tokens 16 \
    --concurrency 1 \
    --requests 1 \
    --prefix-ratio 0.9 \
    --prefix-num 1 \
    --seed 1024 \
    --out "${RESULTS_ROOT}/smoke/bench.json"
  python3 - "${RESULTS_ROOT}/smoke/bench.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
if s["completed"] != 1 or s["errors"] != 0:
    raise SystemExit(f"smoke failed: {s}")
print("smoke passed")
PY
}

capture_metrics() {
  local cell="$1"
  local stage="$2"
  curl -fsS "http://${PREFILL0}:18423/metrics" -o "${cell}/prefill0_${stage}.prom"
  curl -fsS "http://${PREFILL1}:18423/metrics" -o "${cell}/prefill1_${stage}.prom"
  cat "${cell}/prefill0_${stage}.prom" "${cell}/prefill1_${stage}.prom" >"${cell}/prefill_${stage}.prom"
  curl -fsS "http://${DECODE0}:18423/metrics" -o "${cell}/decode_${stage}.prom"
}

dump_expert_heat() {
  local cell="$1"
  local conc="$2"
  local newest remote_pt remote_json

  newest=$(remote "${DECODE0}" "sudo docker exec pd-decode-glm52 bash -lc 'ls -t /tmp/expert_distribution_recorder_*.pt 2>/dev/null | head -1'")
  [ -n "${newest}" ] || die "no expert distribution dump for c${conc}"
  remote_pt="${REMOTE_TMP}/pd_expert_c${conc}.pt"
  remote_json="${REMOTE_TMP}/pd_expert_heat_c${conc}.json"

  remote "${DECODE0}" "mkdir -p '${REMOTE_TMP}'"
  remote "${DECODE0}" "sudo docker cp 'pd-decode-glm52:${newest}' '${remote_pt}'"
  scp "${SSH_OPTS[@]}" "${DECODE0}:${remote_pt}" "${cell}/expert_distribution.pt" >/dev/null

  scp "${SSH_OPTS[@]}" "${HEAT_SCRIPT}" "${DECODE0}:${REMOTE_TMP}/analyze_expert_heat.py" >/dev/null
  remote "${DECODE0}" "sudo docker cp '${REMOTE_TMP}/analyze_expert_heat.py' pd-decode-glm52:/tmp/analyze_expert_heat.py && \
    sudo docker exec pd-decode-glm52 python3 /tmp/analyze_expert_heat.py \
      --pt '${newest}' --out /tmp/expert_heat.json --ep-size 32 >/tmp/analyze_expert_heat.log && \
    sudo docker cp pd-decode-glm52:/tmp/expert_heat.json '${remote_json}'"
  scp "${SSH_OPTS[@]}" "${DECODE0}:${remote_json}" "${cell}/expert_heat.json" >/dev/null
  remote "${DECODE0}" "rm -f '${remote_pt}' '${remote_json}' '${REMOTE_TMP}/analyze_expert_heat.py'"
}

validate_cell() {
  local cell="$1"
  local conc="$2"
  local f
  for f in bench.json prefill_before.prom prefill_after.prom decode_before.prom decode_after.prom breakdown.json expert_distribution.pt expert_heat.json decode_rank0.log decode_rank1.log; do
    [ -s "${cell}/${f}" ] || die "c${conc} missing or empty: ${f}"
  done
  python3 - "${cell}/bench.json" "${conc}" <<'PY'
import json, sys
path, conc = sys.argv[1], int(sys.argv[2])
s = json.load(open(path, encoding="utf-8"))["summary"]
expected = conc * 4
if s["completed"] != expected or s["errors"] != 0:
    raise SystemExit(f"cell validation failed: {s}")
if s["input_tokens"] != 16384 or s["output_tokens"] != 1024:
    raise SystemExit(f"unexpected workload: {s}")
print(f"c{conc} validated: {expected} requests")
PY
  if [ "${EPLB}" = "0" ] && [ -s "${cell}/rebalance.log" ]; then
    die "c${conc} unexpectedly contains EPLB rebalance events"
  fi
}

run_cell() {
  local conc="$1"
  local requests=$((conc * 4))
  local prefix_num=$((conc / 2))
  local cell="${RESULTS_ROOT}/c${conc}"
  mkdir -p "${cell}"

  log "c${conc}: capture before metrics and start recorder"
  capture_metrics "${cell}" before
  remote "${DECODE0}" "sudo docker exec pd-decode-glm52 bash -lc 'rm -f /tmp/expert_distribution_recorder_*.pt 2>/dev/null || true'"
  curl -fsS -X POST "http://${DECODE0}:18423/start_expert_distribution_record" >"${cell}/recorder_start.txt"
  date -Iseconds >"${cell}/start_ts.txt"

  log "c${conc}: run ${requests} requests"
  if ! python3 "${BENCH_SCRIPT}" \
    --url "${ROUTER_URL}" \
    --input-tokens 16384 \
    --output-tokens 1024 \
    --concurrency "${conc}" \
    --requests "${requests}" \
    --prefix-ratio 0.9 \
    --prefix-num "${prefix_num}" \
    --seed 1024 \
    --timeout 3600 \
    --out "${cell}/bench.json"; then
    curl -sS -X POST "http://${DECODE0}:18423/stop_expert_distribution_record" >"${cell}/recorder_stop.txt" || true
    die "benchmark failed for c${conc}"
  fi

  date -Iseconds >"${cell}/end_ts.txt"
  curl -fsS -X POST "http://${DECODE0}:18423/stop_expert_distribution_record" >"${cell}/recorder_stop.txt"
  curl -fsS -X POST "http://${DECODE0}:18423/dump_expert_distribution_record" >"${cell}/recorder_dump.txt"
  sleep 3
  capture_metrics "${cell}" after

  remote "${DECODE0}" "sudo docker logs pd-decode-glm52" >"${cell}/decode_rank0.log" 2>&1
  remote "${DECODE1}" "sudo docker logs pd-decode-glm52" >"${cell}/decode_rank1.log" 2>&1
  grep -hE 'rebalance (start|end)' "${cell}/decode_rank0.log" "${cell}/decode_rank1.log" >"${cell}/rebalance.log" || true

  python3 "${BREAKDOWN_SCRIPT}" \
    --prefill-before "${cell}/prefill_before.prom" \
    --prefill-metrics "${cell}/prefill_after.prom" \
    --decode-before "${cell}/decode_before.prom" \
    --decode-metrics "${cell}/decode_after.prom" \
    --out "${cell}/breakdown.json" >/dev/null
  dump_expert_heat "${cell}" "${conc}"
  validate_cell "${cell}" "${conc}"
  log "c${conc}: complete"
}

matrix() {
  preflight
  wait_health "router" "${ROUTER_URL}"
  local conc
  for conc in 10 20 30; do
    run_cell "${conc}"
  done
  log "matrix complete: ${RESULTS_ROOT}"
}

compare() {
  python3 "${COMPARE_SCRIPT}" \
    --eplb-on "${DIR}/results_final" \
    --eplb-off "${RESULTS_ROOT}" \
    --out "${RESULTS_ROOT}/eplb_ab.md"
}

usage() {
  cat <<EOF
Usage: $0 <preflight|start|status|smoke|matrix|compare|stop>

Default experiment:
  EPLB=0, EP32/DP32, 2P2D, 16K input, 1024 output, concurrency 10/20/30.

Typical sequence:
  $0 preflight
  $0 start
  $0 smoke
  $0 matrix
  $0 compare
  $0 stop
EOF
}

case "${1:-preflight}" in
  preflight) preflight ;;
  start) start_services ;;
  status) status_services ;;
  smoke) smoke ;;
  matrix) matrix ;;
  compare) compare ;;
  stop) stop_services ;;
  *) usage; exit 2 ;;
esac
