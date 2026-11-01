#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chengyetai/workspace/current_exp/eplb_pd_agentx_20260928
SCRIPT=$ROOT/scripts/start_role.sh
DATA=$ROOT/data/agentx_16k_512.json
SESSION=${SESSION:-$(date +%Y%m%d_%H%M%S)}
DURATION=${DURATION:-600}
CONCURRENCY=${CONCURRENCY:-4}
INPUT_TOKENS=${INPUT_TOKENS:-4096}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-128}
BASE=$ROOT/results/$SESSION
mkdir -p "$BASE"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
log(){ echo "[$(date -Is)] $*" | tee -a "$BASE/orchestrator.log"; }
remote(){ "${SSH[@]}" "10.120.72.$1" "${@:2}"; }
container(){ local role=$1 node=$2; echo "eplb26-${role}-${node}"; }
shutdown(){
  local node role
  log 'stopping experiment containers (logs copied before removal)'
  for node in 23 25; do
    if [[ $node == 25 ]]; then role=decode; else role=prefill; fi
    remote "$node" "sudo -n docker rm -f $(container "$role" "$node")" >/dev/null 2>&1 || true
  done
  remote 23 'sudo -n docker rm -f eplb26-router-23' >/dev/null 2>&1 || true
}
capture_logs(){
  local run=$1 node role logdir
  for node in 23 25; do
    if [[ $node == 25 ]]; then role=decode; else role=prefill; fi
    logdir=$BASE/$run/node_$node; mkdir -p "$logdir"
    remote "$node" "sudo -n docker logs --timestamps $(container "$role" "$node") 2>&1" > "$logdir/docker.log" 2>&1 || true
    remote "$node" "sudo -n docker inspect $(container "$role" "$node")" > "$logdir/docker_inspect.json" 2>&1 || true
    if [[ $node != 23 ]]; then
      local remote_logdir=$logdir
      [[ $run == B_pref32 ]] && remote_logdir=$BASE/B/node_$node
      rsync -a -e 'ssh -o BatchMode=yes -o LogLevel=ERROR' "10.120.72.$node:$remote_logdir/" "$logdir/" > "$logdir/rsync.log" 2>&1 || true
    fi
  done
  remote 23 'sudo -n docker logs --timestamps eplb26-router-23 2>&1' > "$BASE/$run/router.log" 2>&1 || true
}
preflight(){
  [[ -s $DATA && -x $SCRIPT ]] || { log 'dataset or launcher missing'; return 1; }
  local script_sha patch_sha
  script_sha=$(sha256sum "$SCRIPT" | cut -d ' ' -f1)
  patch_sha=$(sha256sum "$ROOT/patches/full_window/expert_distribution_full_window_cpu_gloo.py" | cut -d ' ' -f1)
  local mover_sha
  mover_sha=$(sha256sum "$ROOT/patches/eplb/expert_location_updater_lazy_send.py" | cut -d ' ' -f1)
  for node in 23 25; do
    [[ $(remote "$node" "sha256sum '$SCRIPT' | cut -d ' ' -f1") == "$script_sha" ]] || { log "node$node launcher differs"; return 1; }
    [[ $(remote "$node" "sha256sum '$ROOT/patches/full_window/expert_distribution_full_window_cpu_gloo.py' | cut -d ' ' -f1") == "$patch_sha" ]] || { log "node$node CPU/Gloo patch differs"; return 1; }
    [[ $(remote "$node" "sha256sum '$ROOT/patches/eplb/expert_location_updater_lazy_send.py' | cut -d ' ' -f1") == "$mover_sha" ]] || { log "node$node lazy expert move patch differs"; return 1; }
    remote "$node" "test -d '$ROOT/patches/pr32209' && test -s '$ROOT/patches/readonly_source$([[ $node == 25 ]] && echo 25 || echo 23)/summary.json' && test -f '$SCRIPT' && test -d /mnt/raid/user/data/models/glm-qq && test \"\$(sudo -n docker image inspect sglang-glm52-pd:fe694986 --format '{{.Id}}')\" = 'sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d' && curl -fsS --max-time 8 http://127.0.0.1:30122/health >/dev/null"
    log "node$node image, patches, checkpoint, reference ready"
  done
  sha256sum "$SCRIPT" "$ROOT/patches/full_window/expert_distribution_full_window_cpu_gloo.py" "$ROOT/patches/eplb/expert_location_updater_lazy_send.py" "$ROOT/scripts/bench_agentx.py" "$ROOT/scripts/monitor.py" "$ROOT/scripts/snapshot_heat.py" "$DATA" > "$BASE/sha256sums.txt"
}
wait_health(){
  local node=$1 role=$2 label=$3 timeout=${STARTUP_TIMEOUT:-3600} since=$SECONDS status
  log "waiting $label"
  while (( SECONDS-since < timeout )); do
    if curl -fsS --max-time 3 "http://10.120.72.$node:18423/health" >/dev/null 2>&1; then
      log "$label healthy after $((SECONDS-since)) seconds"; return 0
    fi
    status=$(remote "$node" "sudo -n docker inspect $(container "$role" "$node") --format '{{.State.Status}}'" 2>/dev/null || true)
    if [[ $status == exited || $status == dead || $status == restarting ]]; then
      log "$label status=$status"; return 1
    fi
    sleep 12
  done
  log "$label timeout after $timeout seconds"; return 1
}
start_group(){
  local group=$1 eplb=$2 redundant=$3 node role code fail=0
  log "starting group=$group eplb=$eplb redundancy=$redundant"
  mkdir -p "$BASE/$group"
  printf 'group=%s\neplb=%s\nredundant=%s\n' "$group" "$eplb" "$redundant" > "$BASE/$group/config.txt"
  for node in 23 25; do
    if [[ $node == 25 ]]; then role=decode; else role=prefill; fi
    (remote "$node" "bash '$SCRIPT' '$role' '$node' '$SESSION/$group' '$eplb' '$redundant'" > "$BASE/$group/start_$node.log" 2>&1) &
  done
  for code in $(jobs -pr); do wait "$code" || fail=1; done
  if (( fail )); then log "group=$group role launch failed"; capture_logs "$group"; return 1; fi
  wait_health 23 prefill P23 || fail=1
  wait_health 25 decode D25 || fail=1
  if (( fail )); then log "group=$group startup failed"; capture_logs "$group"; return 1; fi
  remote 23 "bash '$SCRIPT' router 23 '$SESSION/$group' '$eplb' '$redundant'" > "$BASE/$group/start_router.log" 2>&1 || { capture_logs "$group"; return 1; }
  local since=$SECONDS
  until curl -fsS --max-time 4 'http://10.120.72.23:18623/health' >/dev/null 2>&1; do
    if ((SECONDS-since>180));then log 'router health timeout'; capture_logs "$group";return 1;fi
    sleep 5
  done
  log "group=$group router healthy"
}
smoke(){
  local group=$1
  local out=$BASE/$group/smoke.json
  # Natural-language prompt goes through the same router as the performance workload.
  python3 - "$out" <<'PY'
import json,sys,urllib.request
prompt='请用一句简短中文说明，为什么海水是咸的？'
req=urllib.request.Request('http://10.120.72.23:18623/generate',data=json.dumps({'text':prompt,'sampling_params':{'temperature':0,'max_new_tokens':96}}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=300) as resp:
    result=json.loads(resp.read())
text=result.get('text','');open(sys.argv[1],'w').write(json.dumps({'prompt':prompt,'result':result},ensure_ascii=False,indent=2))
print(text[:300]);assert len(text.strip())>=5 and any('一'<=c<='鿿' for c in text), 'bad semantic response'
PY
  log "group=$group semantic smoke saved"
}
run_group(){
  local group=$1 monitorpid snapshotpid monitor_code=0 snapshot_code=0 bench_code=0 mode=off decode_source=$1
  local work=$BASE/$group/bench
  log "group=$group entering duration=${DURATION}s concurrency=$CONCURRENCY input=$INPUT_TOKENS output=$OUTPUT_TOKENS"
  mkdir -p "$BASE/$group"
  python3 "$ROOT/scripts/monitor.py" --out "$BASE/$group/metrics" --stop-file "$BASE/$group/monitor.stop" > "$BASE/$group/monitor.log" 2>&1 & monitorpid=$!
  [[ $group == A* ]] || mode=on
  [[ $group == B_pref32 ]] && decode_source=B
  python3 "$ROOT/scripts/snapshot_heat.py" --session "$SESSION" --group "$group" --decode-source-group "$decode_source" --mode "$mode" --out "$BASE/$group/snapshots" --stop-file "$BASE/$group/monitor.stop" > "$BASE/$group/snapshots.log" 2>&1 & snapshotpid=$!
  python3 "$ROOT/scripts/bench_agentx.py" --dataset "$DATA" --out "$work" --duration "$DURATION" --concurrency "$CONCURRENCY" --input-tokens "$INPUT_TOKENS" --output-tokens "$OUTPUT_TOKENS" > "$BASE/$group/bench.log" 2>&1 || bench_code=$?
  touch "$BASE/$group/monitor.stop"
  wait "$monitorpid" || monitor_code=$?
  wait "$snapshotpid" || snapshot_code=$?
  capture_logs "$group"
  log "group=$group bench_code=$bench_code monitor_code=$monitor_code snapshot_code=$snapshot_code"
  (( bench_code == 0 && monitor_code == 0 ))
}
if [[ ${1:-all} == preflight ]]; then preflight; exit; fi
preflight
if [[ ${1:-all} == resume_B_new ]]; then
  trial=${NEW_B_TRIAL:-B_cpu_gloo}
  shutdown
  start_group "$trial" 1 0
  smoke "$trial"
  run_group "$trial"
  shutdown
fi
if [[ ${1:-all} == resume_B ]]; then
  trial=${RESUME_B_TRIAL:-B_pref32}
  decode_source=${RESUME_B_DECODE_SOURCE:-B}
  mkdir -p "$BASE/$trial"
  printf 'group=%s\neplb=1\nredundant=0\ndecode_source_group=%s\n' "$trial" "$decode_source" > "$BASE/$trial/config.txt"
  wait_health 23 prefill P23
  wait_health 25 decode D25
  if ! curl -fsS --max-time 4 http://10.120.72.23:18623/health >/dev/null 2>&1; then
    remote 23 "bash '$SCRIPT' router 23 '$SESSION/$trial' 1 0" > "$BASE/$trial/start_router.log" 2>&1
  fi
  until curl -fsS --max-time 4 'http://10.120.72.23:18623/health' >/dev/null 2>&1; do sleep 5; done
  smoke "$trial"
  run_group "$trial"
  shutdown
fi
if [[ ${1:-all} == resume_A ]]; then
  trial=${RESUME_A_TRIAL:-A_retry_mem088}
  mkdir -p "$BASE/$trial"
  # The initial A attempt loaded Prefill on 23 successfully. Its failed Decode
  # (mem fraction .82) is retained under A; Decode .88 started separately.
  wait_health 23 prefill P23
  wait_health 25 decode D25
  if ! curl -fsS --max-time 4 http://10.120.72.23:18623/health >/dev/null 2>&1; then
    remote 23 "bash '$SCRIPT' router 23 '$SESSION/$trial' 0 0" > "$BASE/$trial/start_router.log" 2>&1
  fi
  until curl -fsS --max-time 4 'http://10.120.72.23:18623/health' >/dev/null 2>&1; do sleep 5; done
  smoke "$trial"
  run_group "$trial"
  shutdown
fi
GROUPS="${GROUPS:-A B C}"
for pair in 'A 0 0' 'B 1 0' 'C 1 16'; do
  read -r group eplb redundant <<< "$pair"
  if [[ " ${GROUPS} " != *" ${group} "* ]]; then
    log "skip group=$group"
    continue
  fi
  if [[ ${1:-all} == resume_A && $group == A ]]; then continue; fi
  if [[ ( ${1:-all} == resume_B || ${1:-all} == resume_B_new ) && $group == B ]]; then continue; fi
  if [[ $group == C ]]; then
    # EP16 requires (256 + redundant) % 16 == 0; 16 is the smallest
    # valid positive redundancy. Preserve failure evidence if it OOMs.
    candidates=(16)
  else candidates=("$redundant"); fi
  ok=0
  for redundant in "${candidates[@]}"; do
    trial=$group
    [[ $group == C ]] && trial="C_r$redundant"
    shutdown
    if start_group "$trial" "$eplb" "$redundant"; then
      if smoke "$trial"; then ok=1; break; fi
      capture_logs "$trial"
    fi
    if [[ $group == C ]]; then
      if rg -qi 'out of memory|oom|aclrtmalloc|alloc.*fail|memory allocation|free memory|OOM' "$BASE/$trial"/node_*/docker.log "$BASE/$trial"/start_*.log; then
        log "redundant=$redundant OOM or allocation failure, retry lower redundant count"
      else
        log "group=$group failed for cause other than diagnosed memory shortage; refusing to reclassify as OOM"
        exit 1
      fi
    else exit 1; fi
  done
  (( ok )) || { log "group=$group no viable redundancy configuration"; exit 1; }
  run_group "$trial"
  shutdown
  log "group=$group complete; actual redundancy=$redundant"
done
log "three comparable groups complete (plus original A diagnostic): $BASE"
