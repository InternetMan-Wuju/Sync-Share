#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$ROOT/artifacts"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"

run2() {
  local name="$1" script="$2"; shift 2
  sudo -n docker run --rm --name "$name" --net host --ipc host --shm-size 16g \
    --security-opt seccomp=unconfined \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" -v "$ROOT:/runexp:rw" \
    -e PYTHONPATH=/runexp/scripts -e GLM_MODEL=/models/glm-q -e OUT_DIR=/runexp/artifacts \
    -e DEEP_USE_MODE=alltoall -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64 \
    --entrypoint bash "$IMAGE" -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/$script $*"
}

functional() {
  run2 cyt-unified-functional run_functional.py --scenario matrix --tag unified 2>&1 | tee "$ART/functional.log"
}

format_memory() {
  functional
  run2 cyt-unified-mem-baseline run_functional.py --scenario baseline --tag mem_baseline 2>&1 | tee "$ART/memory_baseline.log"
  run2 cyt-unified-mem-candidate run_functional.py --scenario candidate --tag mem_candidate 2>&1 | tee "$ART/memory_candidate.log"
  python3 "$ROOT/scripts/build_validation.py"
}

reuse() { functional; python3 "$ROOT/scripts/build_validation.py"; }
validate() { python3 "$ROOT/scripts/build_validation.py"; }

cpu() {
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$ART/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$ART/cpu_expert_search.txt"
}

case "${1:-help}" in
  functional) functional ;;
  format-memory) format_memory ;;
  reuse) reuse ;;
  validate) validate ;;
  cpu) cpu ;;
  all) cpu; format_memory ;;
  *) echo "usage: $0 {functional|format-memory|reuse|validate|cpu|all}" >&2; exit 2 ;;
esac
