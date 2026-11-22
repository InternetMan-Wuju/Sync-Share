#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$ROOT/artifacts"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"

run2() {
  local name="$1" script="$2"; shift 2
  local deep_mode="${DEEP_MODE:-alltoall}"
  local env_args=(-e PYTHONPATH=/runexp/scripts -e GLM_MODEL=/models/glm-q -e OUT_DIR=/runexp/artifacts -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64)
  if [ "$deep_mode" != "none" ]; then env_args+=(-e "DEEP_USE_MODE=$deep_mode"); fi
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
    "${env_args[@]}" \
    --entrypoint bash "$IMAGE" -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/$script $*"
}

cpu() {
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$ART/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$ART/cpu_expert_search.txt"
}

interface() {
  if DEEP_MODE=none run2 cyt-placeholder-default native_interface_harness.py --profile 256 2>&1 | tee "$ART/default_normal_256_failure.log"; then
    echo "expected default-normal failure did not occur" >&2; exit 1
  fi
  grep -q 'aclnnDispatchLayout' "$ART/default_normal_256_failure.log"
  run2 cyt-placeholder-256 native_interface_harness.py --profile 256 2>&1 | tee "$ART/interface_256.log"
  run2 cyt-placeholder-257 native_interface_harness.py --profile 257 2>&1 | tee "$ART/interface_257.log"
  run2 cyt-placeholder-258 native_interface_harness.py --profile 258 2>&1 | tee "$ART/interface_258.log"
  run2 cyt-placeholder-zero zero_rank_protocol.py 2>&1 | tee "$ART/zero_rank_protocol.log"
  python3 "$ROOT/scripts/build_interface_artifact.py"
}

real() {
  run2 cyt-placeholder-real real_native_placeholder.py --tag real258 2>&1 | tee "$ART/real_native.log"
  python3 "$ROOT/scripts/build_artifacts.py"
}

case "${1:-help}" in
  cpu) cpu ;;
  interface|protocol) interface ;;
  real|memory|reuse) real ;;
  all) interface; real ;;
  *) echo "usage: $0 {cpu|interface|protocol|real|memory|reuse|all}" >&2; exit 2 ;;
esac
