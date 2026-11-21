#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
RUN_DIR="$ROOT"
ART="$ROOT/artifacts"

cpu() {
  mkdir -p "$ART"
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$ART/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$ART/cpu_expert_search_tests.txt"
  python3 "$ROOT/tests/test_global_k.py" 2>&1 | tee "$ART/cpu_global_k.txt"
  python3 "$ROOT/scripts/run_expert_search_validation.py"
}

docker_common() {
  local name="$1"
  local command="$2"
  shift 2
  local devices=("$@")
  sudo -n docker run --rm --name "$name" --net host --ipc host --shm-size 16g \
    --security-opt seccomp=unconfined \
    "${devices[@]}" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /var/queue_schedule:/var/queue_schedule \
    -v "$MODEL:/models/glm-q:ro" \
    -v "$RUN_DIR:/runexp:rw" \
    -e PYTHONPATH=/runexp/scripts \
    -e GLM_MODEL=/models/glm-q \
    -e OUT_DIR=/runexp/artifacts \
    --entrypoint bash "$IMAGE" -lc "$command"
}

single() {
  docker_common cyt-global-k-rework-single \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && python3 /runexp/scripts/real_expert_chain_rework.py' \
    --device /dev/davinci0 \
    2>&1 | tee "$ART/single_card_rework.log"
}

router() {
  docker_common cyt-global-k-rework-router \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && python3 /runexp/scripts/single_layer_manual_router_probe.py' \
    --device /dev/davinci0 \
    2>&1 | tee "$ART/single_layer_manual_router.log"
}

two_rank() {
  docker_common cyt-global-k-rework-two-rank \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/two_rank_paired_timing.py' \
    --device /dev/davinci0 --device /dev/davinci1 \
    2>&1 | tee "$ART/two_rank_paired.log"
  python3 "$ROOT/scripts/combine_two_rank_results.py"
}

case "${1:-help}" in
  cpu) cpu ;;
  single) single ;;
  router) router ;;
  two-rank) two_rank ;;
  all-npu) single; router; two_rank ;;
  *) echo "usage: $0 {cpu|single|router|two-rank|all-npu}" >&2; exit 2 ;;
esac
