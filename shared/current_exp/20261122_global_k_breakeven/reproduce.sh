#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$ROOT/artifacts"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"

cpu() {
  mkdir -p "$ART"
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$ART/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$ART/cpu_expert_search_tests.txt"
  python3 "$ROOT/scripts/run_expert_search_validation.py"
}

run_two_rank() {
  local name="$1" script="$2"
  shift 2
  sudo -n docker run --rm --name "$name" --net host --ipc host --shm-size 16g \
    --security-opt seccomp=unconfined \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" \
    -v "$ROOT:/runexp:rw" \
    -e PYTHONPATH=/runexp/scripts -e GLM_MODEL=/models/glm-q \
    -e OUT_DIR=/runexp/artifacts \
    --entrypoint bash "$IMAGE" -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/$script $*"
}

curves() {
  run_two_rank cyt-global-k-breakeven-curves compute_curves.py \
    2>&1 | tee "$ART/compute_curves.log"
  python3 "$ROOT/scripts/build_compute_curves.py" >/dev/null
}

smoke() {
  run_two_rank cyt-global-k-breakeven-smoke prefill_pair_timing.py \
    --mode formal --tag smoke --points 6:3 --seeds 809000,809001 \
    2>&1 | tee "$ART/prefill_pair_smoke.log"
  python3 "$ROOT/scripts/analyze_pair_results.py" --tag smoke >/dev/null
}

diagnostic() {
  run_two_rank cyt-global-k-breakeven-diag prefill_pair_timing.py \
    --mode diagnostic --tag diagnostic --seeds 810000,810001 \
    2>&1 | tee "$ART/prefill_pair_diagnostic.log"
  python3 "$ROOT/scripts/analyze_pair_results.py" --diagnostic >/dev/null
}

formal() {
  run_two_rank cyt-global-k-breakeven-formal prefill_pair_timing.py \
    --mode formal --tag standard --seeds 810000,810001 \
    2>&1 | tee "$ART/prefill_pair_formal_standard.log"
  python3 "$ROOT/scripts/analyze_pair_results.py" --tag standard >/dev/null
}

overlap() {
  run_two_rank cyt-global-k-breakeven-overlap overlap_timeline.py \
    2>&1 | tee "$ART/overlap_timeline.log"
  python3 "$ROOT/scripts/build_overlap_summary.py" >/dev/null
}

summary() {
  python3 "$ROOT/scripts/build_summary.py"
}

case "${1:-help}" in
  cpu) cpu ;;
  curves) curves ;;
  smoke) smoke ;;
  diagnostic) diagnostic ;;
  formal) formal ;;
  overlap) overlap ;;
  summary) summary ;;
  all)
    cpu
    curves
    smoke
    diagnostic
    formal
    overlap
    summary
    ;;
  *)
    echo "usage: $0 {cpu|curves|smoke|diagnostic|formal|overlap|summary|all}" >&2
    exit 2
    ;;
esac
