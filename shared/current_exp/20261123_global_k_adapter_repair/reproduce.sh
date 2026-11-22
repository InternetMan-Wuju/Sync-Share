#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3}"
MODEL="${MODEL:-/data_lib/data/models/glm-q}"
MAX_DISPATCH="${MAX_DISPATCH:-512}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
CAMPAIGN_DIR=""

new_uuid() { python3 -c 'import uuid; print(uuid.uuid4().hex)'; }
code_hash() { (cd "$ROOT" && find scripts tests -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'); }

start_campaign() {
  if [ -z "$CAMPAIGN_ID" ]; then CAMPAIGN_ID="$(new_uuid)"; fi
  CAMPAIGN_DIR="$ROOT/artifacts/campaigns/$CAMPAIGN_ID"
  mkdir -p "$CAMPAIGN_DIR"/{functional,memory-baseline,memory-candidate,logs}
  CODE_HASH="$(code_hash)"
  SUB_FUNCTIONAL="$(new_uuid)"; SUB_MEM_BASE="$(new_uuid)"; SUB_MEM_CAND="$(new_uuid)"
  export CAMPAIGN_ID CODE_HASH SUB_FUNCTIONAL SUB_MEM_BASE SUB_MEM_CAND CAMPAIGN_DIR IMAGE MODEL MAX_DISPATCH
  python3 - "$CAMPAIGN_DIR/campaign_manifest.json" <<'PY'
import json,os,sys,datetime,hashlib
from pathlib import Path
p=Path(sys.argv[1])
manifest={
  'campaign_id':os.environ['CAMPAIGN_ID'],
  'code_hash':os.environ['CODE_HASH'],
  'subruns':{'functional':os.environ['SUB_FUNCTIONAL'],'memory-baseline':os.environ['SUB_MEM_BASE'],'memory-candidate':os.environ['SUB_MEM_CAND']},
  'expected_cases':['both_mixed_q0_noslot','no_input_receives_tasks','input_but_zero_expert_tasks','both_empty','invalid_rank0','valid_after_invalid_rank0','invalid_rank1','valid_after_invalid_rank1','reserved_but_claims_ready','valid_after_reserved','partial_slot','all_slot','large_259_partial','one_rank_bad_slot','valid_after_one_rank_bad','replay_old_49_after_install50','current_50_after_reinstall','replay_old_generation_49','current_generation_49','fixed_lease_overwrite_blocked'],
  'expected_install_tags':['repaired_install49','repaired_transition50','repaired_reinstall49','repaired_reuse0_l50','repaired_reuse0_l49','repaired_reuse1_l50','repaired_reuse1_l49','repaired_reuse2_l50','repaired_reuse2_l49'],
  'image':os.environ.get('IMAGE','quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3'),
  'devices':['davinci0','davinci1'],'host':'npu1-23','model_mount':'/models/glm-q',
  'dispatch_capacity':int(os.environ.get('MAX_DISPATCH','512')),'deep_ep_mode':'alltoall',
  'started_at':datetime.datetime.now().astimezone().isoformat(),
  'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','performance_status':'NOT_RUN'
}
model=Path(os.environ.get('MODEL','/data_lib/data/models/glm-q'))
for name in ('config.json','quant_model_weights.safetensors.index.json'):
    q=model/name
    if q.exists(): manifest['model_'+name.replace('.','_')+'_sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
p.write_text(json.dumps(manifest,indent=2)+'\n')
PY
}

run2() {
  local name="$1" subdir="$2" script="$3"; shift 3
  local subrun
  case "$subdir" in
    functional) subrun="$SUB_FUNCTIONAL" ;;
    memory-baseline) subrun="$SUB_MEM_BASE" ;;
    memory-candidate) subrun="$SUB_MEM_CAND" ;;
    *) echo "bad subdir $subdir" >&2; return 2 ;;
  esac
  local container_out="/runexp/artifacts/campaigns/$CAMPAIGN_ID/$subdir"
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
    -e PYTHONPATH=/runexp/scripts -e GLM_MODEL=/models/glm-q -e OUT_DIR="$container_out" \
    -e CAMPAIGN_ID="$CAMPAIGN_ID" -e SUBRUN_ID="$subrun" -e CODE_HASH="$CODE_HASH" \
    -e DEEP_USE_MODE=alltoall -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$MAX_DISPATCH" \
    -e IMAGE="$IMAGE" -e MODEL="$MODEL" -e MAX_DISPATCH="$MAX_DISPATCH" \
    --entrypoint bash "$IMAGE" -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && torchrun --standalone --nproc_per_node=2 /runexp/scripts/$script $*"
}

functional_run() {
  run2 "cyt-repair-functional" functional run_functional.py --scenario matrix --tag repaired --out-dir "/runexp/artifacts/campaigns/$CAMPAIGN_ID/functional" 2>&1 | tee "$CAMPAIGN_DIR/logs/functional.log"
}
memory_run() {
  run2 "cyt-repair-mem-base" memory-baseline run_functional.py --scenario baseline --tag mem_baseline --out-dir "/runexp/artifacts/campaigns/$CAMPAIGN_ID/memory-baseline" 2>&1 | tee "$CAMPAIGN_DIR/logs/memory-baseline.log"
  run2 "cyt-repair-mem-cand" memory-candidate run_functional.py --scenario candidate --tag mem_candidate --out-dir "/runexp/artifacts/campaigns/$CAMPAIGN_ID/memory-candidate" 2>&1 | tee "$CAMPAIGN_DIR/logs/memory-candidate.log"
}
validate_run() {
  local target="${1:-$CAMPAIGN_DIR}"
  python3 "$ROOT/scripts/build_validation.py" "$target" 2>&1 | tee "$CAMPAIGN_DIR/logs/validate.log"
}
cpu_run() {
  python3 "$ROOT/tests/test_slot_identity.py" 2>&1 | tee "$CAMPAIGN_DIR/logs/cpu_slot_identity.txt"
  python3 "$ROOT/tests/test_expert_search.py" 2>&1 | tee "$CAMPAIGN_DIR/logs/cpu_expert_search.txt"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  functional) start_campaign; functional_run ;;
  owner-state) start_campaign; functional_run ;;
  memory) start_campaign; memory_run ;;
  cpu) start_campaign; cpu_run ;;
  validate) [ $# -eq 1 ] || { echo "usage: $0 validate <campaign-directory>" >&2; exit 2; }; python3 "$ROOT/scripts/build_validation.py" "$1" ;;
  all) start_campaign; cpu_run; functional_run; memory_run; validate_run ;;
  *) echo "usage: $0 {functional|owner-state|memory|validate <campaign-dir>|cpu|all}" >&2; exit 2 ;;
esac
