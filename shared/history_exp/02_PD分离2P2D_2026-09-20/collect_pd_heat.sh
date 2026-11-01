#!/bin/bash
set -euo pipefail

# Re-run the proven 2P2D EPLB-on workload and collect separate P0/P1/D heat.

DIR="$(cd "$(dirname "$0")" && pwd)"
ORCHESTRATOR="${DIR}/next_experiment.sh"
BENCH="${DIR}/bench_pd_prefix.py"
ANALYZER="${DIR}/analyze_pd_rank_heat.py"
PLOTTER="${DIR}/plot_pd_heat.py"
PARSE_MAP="${DIR}/../glm52_ep64_chengyetai/ep_lab/parse_layout_from_log.py"

P0="${P0:-10.120.72.23}"
P1="${P1:-10.120.72.25}"
D0="${D0:-10.120.72.31}"
D1="${D1:-10.120.72.32}"
ROUTER_URL="${ROUTER_URL:-http://${P0}:18623}"
IMAGE="${IMAGE:-sglang-glm52-pd:fe694986}"
FULL_WINDOW_PATCH="${DIR}/patches/full_window/expert_distribution_full_window_patched.py"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RESULTS_ROOT="${RESULTS_ROOT:-${DIR}/results/${RUN_DATE}_pd_separated_heat_eplb_on_16k}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/home/chengyetai/workspace/current_exp}"
REMOTE_TMP="${REMOTE_TMP:-${REMOTE_WORKDIR}/tmp}"
CONCURRENCIES="${CONCURRENCIES:-10 20 30}"
INPUT_TOKENS="${INPUT_TOKENS:-16384}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
EPLB="${EPLB:-1}"
EP_NUM_REDUNDANT="${EP_NUM_REDUNDANT:-0}"
OUTPUT_UID="${OUTPUT_UID:-$(id -u)}"
OUTPUT_GID="${OUTPUT_GID:-$(id -g)}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${host}" "$@"
}

endpoint() {
  case "$1" in
    prefill0) echo "http://${P0}:18423" ;;
    prefill1) echo "http://${P1}:18423" ;;
    decode) echo "http://${D0}:18423" ;;
    *) return 2 ;;
  esac
}

role_host() {
  case "$1" in
    prefill0) echo "${P0}" ;;
    prefill1) echo "${P1}" ;;
    decode) echo "${D0}" ;;
    *) return 2 ;;
  esac
}

role_container() {
  case "$1" in
    prefill0|prefill1) echo "pd-prefill-glm52" ;;
    decode) echo "pd-decode-glm52" ;;
    *) return 2 ;;
  esac
}

role_ep_size() {
  case "$1" in
    prefill0|prefill1) echo 16 ;;
    decode) echo 32 ;;
    *) return 2 ;;
  esac
}

preflight() {
  bash -n "$0" "${ORCHESTRATOR}"
  python3 -m py_compile "${BENCH}" "${ANALYZER}" "${PLOTTER}" "${PARSE_MAP}"
  local f host local_sha remote_sha
  [ -f "${FULL_WINDOW_PATCH}" ] || {
    echo "missing full-window patch: ${FULL_WINDOW_PATCH}" >&2
    exit 1
  }
  for host in "${P0}" "${P1}" "${D0}" "${D1}"; do
    remote "${host}" "mkdir -p '${REMOTE_WORKDIR}/patches/full_window' '${REMOTE_TMP}'"
    scp "${SSH_OPTS[@]}" "${FULL_WINDOW_PATCH}" \
      "${host}:${REMOTE_WORKDIR}/patches/full_window/expert_distribution_full_window_patched.py" \
      >/dev/null
  done
  for f in \
    /home/chengyetai/workspace/current_exp/patches/eplb/expert_location_updater_patched.py \
    /home/chengyetai/workspace/current_exp/patches/eplb/torch_npu_distributed_c10d_patched.py \
    /home/chengyetai/workspace/current_exp/patches/full_window/expert_distribution_full_window_patched.py; do
    for host in "${P0}" "${P1}" "${D0}" "${D1}"; do
      local_sha=$(sha256sum "${f}" | cut -d ' ' -f 1)
      remote_sha=$(remote "${host}" "sha256sum '${f}' | cut -d ' ' -f 1")
      [ "${local_sha}" = "${remote_sha}" ] || {
        echo "patch mismatch: ${host}:${f}" >&2
        exit 1
      }
    done
  done
  mkdir -p "${RESULTS_ROOT}"
  log "preflight passed; results=${RESULTS_ROOT}"
}

start_services() {
  preflight
  EPLB="${EPLB}" EP_NUM_REDUNDANT="${EP_NUM_REDUNDANT}" EXPERT_RECORDER=1 FULL_WINDOW_DUMP=1 RESULTS_ROOT="${RESULTS_ROOT}" \
    bash "${ORCHESTRATOR}" start
}

stop_services() {
  bash "${ORCHESTRATOR}" stop
}

check_health() {
  for url in "$(endpoint prefill0)" "$(endpoint prefill1)" "$(endpoint decode)" "${ROUTER_URL}"; do
    curl -fsS --max-time 5 "${url}/health" >/dev/null
  done
}

recorder_call() {
  local action="$1"
  local role url
  for role in prefill0 prefill1 decode; do
    url=$(endpoint "${role}")
    curl -fsS --connect-timeout 5 --max-time 30 -X POST \
      "${url}/${action}_expert_distribution_record" \
      >"${CURRENT_CELL}/${role}_recorder_${action}.txt"
  done
}

clear_recorders() {
  local role host container
  for role in prefill0 prefill1 decode; do
    host=$(role_host "${role}")
    container=$(role_container "${role}")
    remote "${host}" "sudo docker exec '${container}' bash -lc 'rm -f /tmp/expert_distribution_recorder_*.pt /tmp/expert_distribution_window_*.pt 2>/dev/null || true'"
  done
}

collect_window_pts() {
  local role="$1"
  local dest_dir="$2"
  local host container archive container_archive host_archive count
  host=$(role_host "${role}")
  container=$(role_container "${role}")
  mkdir -p "${dest_dir}"
  archive="${role}_expert_windows_$$.tar"
  container_archive="/tmp/${archive}"
  host_archive="${REMOTE_TMP}/${archive}"
  remote "${host}" "mkdir -p '${REMOTE_TMP}'"
  remote "${host}" "sudo docker exec '${container}' bash -lc 'cd /tmp; files=\$(ls -1 expert_distribution_window_*.pt expert_distribution_recorder_*.pt 2>/dev/null); test -n \"\${files}\"; tar -cf ${container_archive} \${files}' && \
    sudo docker cp '${container}:${container_archive}' '${host_archive}'"
  scp "${SSH_OPTS[@]}" "${host}:${host_archive}" "${dest_dir}/${archive}" >/dev/null
  tar -xf "${dest_dir}/${archive}" -C "${dest_dir}"
  unlink "${dest_dir}/${archive}"
  remote "${host}" "sudo docker exec '${container}' rm -f '${container_archive}'; rm -f '${host_archive}'"
  count=$(find "${dest_dir}" -maxdepth 1 -type f -name 'expert_distribution_*.pt' | wc -l)
  [ "${count}" -gt 0 ] || {
    echo "no recorder windows collected for ${role}" >&2
    return 1
  }
  log "${role}: collected ${count} recorder windows"
}

capture_logs() {
  remote "${P0}" "sudo docker logs pd-prefill-glm52" >"${CURRENT_CELL}/prefill0.log" 2>&1
  remote "${P1}" "sudo docker logs pd-prefill-glm52" >"${CURRENT_CELL}/prefill1.log" 2>&1
  remote "${D0}" "sudo docker logs pd-decode-glm52" >"${CURRENT_CELL}/decode_rank0.log" 2>&1
  remote "${D1}" "sudo docker logs pd-decode-glm52" >"${CURRENT_CELL}/decode_rank1.log" 2>&1
}

analyze_role() {
  local role="$1"
  local ep_size role_dir rel_file
  local pt_args=()
  ep_size=$(role_ep_size "${role}")
  role_dir="${CURRENT_CELL}/windows/${role}"
  while IFS= read -r file; do
    rel_file="${file#${DIR}/}"
    pt_args+=(--pt "/workspace/current_exp/${rel_file}")
  done < <(find "${role_dir}" -maxdepth 1 -type f -name 'expert_distribution_*.pt' | sort)
  [ "${#pt_args[@]}" -gt 0 ] || {
    echo "no window files for ${role}" >&2
    return 1
  }

  remote "${P0}" sudo docker run --rm \
      --user "${OUTPUT_UID}:${OUTPUT_GID}" \
      -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      -v /home/chengyetai/workspace:/workspace \
      --entrypoint python3 "${IMAGE}" \
      /workspace/current_exp/analyze_pd_rank_heat.py \
      "${pt_args[@]}" \
      --ep-size "${ep_size}" \
      --role "${role}" \
      --out-dir "/workspace/current_exp/${CURRENT_CELL#${DIR}/}/heat"
}

run_cell() {
  local conc="$1"
  local requests=$((conc * 4))
  local prefix_num=$((conc / 2))
  CURRENT_CELL="${RESULTS_ROOT}/c${conc}"
  export CURRENT_CELL
  mkdir -p "${CURRENT_CELL}/heat"

  check_health
  clear_recorders
  recorder_call start
  date -Iseconds >"${CURRENT_CELL}/start_ts.txt"
  log "c${conc}: collecting separate P0/P1/D heat"
  if ! python3 "${BENCH}" \
    --url "${ROUTER_URL}" \
    --input-tokens "${INPUT_TOKENS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --concurrency "${conc}" \
    --requests "${requests}" \
    --prefix-ratio 0.9 \
    --prefix-num "${prefix_num}" \
    --seed 1024 \
    --timeout 3600 \
    --out "${CURRENT_CELL}/bench.json"; then
    recorder_call stop || true
    exit 1
  fi
  date -Iseconds >"${CURRENT_CELL}/end_ts.txt"
  recorder_call stop
  recorder_call dump
  sleep 3

  capture_logs
  collect_window_pts prefill0 "${CURRENT_CELL}/windows/prefill0"
  collect_window_pts prefill1 "${CURRENT_CELL}/windows/prefill1"
  collect_window_pts decode "${CURRENT_CELL}/windows/decode"

  analyze_role prefill0
  analyze_role prefill1
  analyze_role decode
  python3 "${PLOTTER}" \
    --prefill0 "${CURRENT_CELL}/heat/prefill0_heat.json" \
    --prefill1 "${CURRENT_CELL}/heat/prefill1_heat.json" \
    --decode "${CURRENT_CELL}/heat/decode_heat.json" \
    --out-dir "${CURRENT_CELL}/heat" \
    --title "PD separated heat, concurrency ${conc}"

  python3 - "${CURRENT_CELL}/bench.json" "${CURRENT_CELL}/heat/pd_heat_summary.json" "${requests}" <<'PY'
import json, sys
bench = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
summary = json.load(open(sys.argv[2], encoding="utf-8"))
expected = int(sys.argv[3])
if bench["completed"] != expected or bench["errors"] != 0:
    raise SystemExit(f"benchmark incomplete: {bench}")
if summary["prefill"]["total_expert_token_assignments"] <= 0:
    raise SystemExit("prefill heat is empty")
if summary["decode"]["total_expert_token_assignments"] <= 0:
    raise SystemExit("decode heat is empty")
for role in ("prefill", "decode"):
    if not summary[role]["coverage_validated"]:
        raise SystemExit(f"{role} does not cover all rebalance windows")
print(f"validated {expected} requests and non-empty P/D heat")
PY
}

matrix() {
  preflight
  check_health
  local conc
  for conc in ${CONCURRENCIES}; do
    case "${conc}" in
      10|20|30) ;;
      *) echo "unsupported concurrency: ${conc}" >&2; return 2 ;;
    esac
    run_cell "${conc}"
  done
}

usage() {
  cat <<EOF
Usage: $0 <preflight|start|matrix|stop|all>

  start   Start the proven EPLB-on 2P2D service.
  matrix  Run c10/c20/c30 and collect separate P0/P1/D heat.
  all     Start, run matrix, and stop services.

Set CONCURRENCIES="20 30" to run only selected cells.
EOF
}

case "${1:-preflight}" in
  preflight) preflight ;;
  start) start_services ;;
  matrix) matrix ;;
  stop) stop_services ;;
  all)
    trap stop_services EXIT
    start_services
    matrix
    ;;
  *) usage; exit 2 ;;
esac
