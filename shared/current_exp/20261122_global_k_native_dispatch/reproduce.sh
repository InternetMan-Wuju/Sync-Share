#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$ROOT/artifacts"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"

interface() {
  python3 "$ROOT/scripts/interface_contract_probe.py"
}

cpu() {
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$ART/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$ART/cpu_expert_search.txt"
}

run2() {
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
    -v "$MODEL:/models/glm-q:ro" -v "$ROOT:/runexp:rw" \
    -e PYTHONPATH=/runexp/scripts -e GLM_MODEL=/models/glm-q -e OUT_DIR=/runexp/artifacts \
    --entrypoint bash "$IMAGE" -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/$script $*"
}

old() {
  run2 cyt-native-old pair_lifecycle_timing.py --lifecycle old --round 1 --seeds 920000,920001 --points 8192:2048 2>&1 | tee "$ART/old_round1.log"
}

resident() {
  run2 cyt-native-res1 pair_lifecycle_timing.py --lifecycle resident --round 1 --seeds 930000,930001 --points 8192:2048,4096:1024 2>&1 | tee "$ART/resident_round1.log"
  run2 cyt-native-res2 pair_lifecycle_timing.py --lifecycle resident --round 2 --seeds 940000,940001 --points 8192:2048,4096:1024 2>&1 | tee "$ART/resident_round2.log"
}

profile() {
  run2 cyt-native-profile profile_device.py 2>&1 | tee "$ART/device_profiler.log"
  python3 "$ROOT/scripts/build_device_summary.py" >/dev/null
}

analyze() {
  python3 "$ROOT/scripts/analyze_lifecycle.py" >/dev/null
  python3 "$ROOT/scripts/build_device_summary.py" >/dev/null
  python3 "$ROOT/scripts/build_summary.py"
}

case "${1:-help}" in
  interface) interface ;;
  cpu) cpu ;;
  old) old ;;
  resident) resident ;;
  profile) profile ;;
  analyze) analyze ;;
  all) interface; cpu; old; resident; profile; analyze ;;
  *) echo "usage: $0 {interface|cpu|old|resident|profile|analyze|all}" >&2; exit 2 ;;
esac
