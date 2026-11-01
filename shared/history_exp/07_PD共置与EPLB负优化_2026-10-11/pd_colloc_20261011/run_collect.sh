#!/usr/bin/env bash
# 采集编排：随机/定时 dump 热度窗口 + 跑 agentX 负载
set -u
D=/home/chengyetai/workspace/current_exp/pd_colloc_20261011
RUN="$1"; DUR="${2:-600}"; CONC="${3:-16}"
EP=http://10.120.72.19:30000
OUT="$D/results/$RUN"; mkdir -p "$OUT"
DATA=/home/chengyetai/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json
BENCH=/home/chengyetai/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/scripts/bench_agentx.py
EV="$OUT/events.jsonl"
say(){ echo "$(date -Is) $*" | tee -a "$OUT/orchestrator.log"; }
dump(){
  local why="$1"; local t0=$(date +%s.%N)
  local code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 -X POST "$EP/dump_expert_distribution_record" 2>/dev/null || echo ERR)
  local t1=$(date +%s.%N)
  python3 -c "import json,sys;print(json.dumps({'at':'$(date -Is)','why':'$why','http':'''$code''','cost_s':round($t1-$t0,3)}))" >> "$EV"
  say "dump why=$why http=$code cost=$(python3 -c "print(round($t1-$t0,3))")s"
}
say "start: duration=${DUR}s concurrency=$CONC"
curl -sS -o /dev/null --max-time 30 -X POST "$EP/start_expert_distribution_record" 2>/dev/null || true
say "recorder started"
python3 "$BENCH" --url "$EP" --dataset "$DATA" --out "$OUT/bench2" \
  --duration "$DUR" --concurrency "$CONC" --input-tokens 16384 --output-tokens 512 \
  > "$OUT/bench.log" 2>&1 &
BPID=$!
while kill -0 $BPID 2>/dev/null; do sleep 60; kill -0 $BPID 2>/dev/null && dump periodic; done
wait $BPID; say "bench done rc=$?"
dump final
say "all done"
