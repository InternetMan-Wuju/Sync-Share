#!/usr/bin/env bash
# Q4Exp 单轮跑批：起服务 -> 等健康 -> 开始采集 -> 打负载(时长内) -> 定时抓指标/清理 -> 收尾归档
# 用法: ./run_q4exp.sh <A0|A|B|C> <agentx|random> [时长秒=900] [并发=8]
set -u
DIR=/home/chengyetai/workspace/current_exp/pd_colloc_20261011
EP=http://10.120.72.19:30000
NODE0=19
BENCH=/home/chengyetai/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/scripts/bench_agentx.py
GROUP="$1"; LOAD="$2"; DUR="${3:-900}"; CONC="${4:-8}"

case "$GROUP" in
  A0) EPLB=0; REC=0; REDUNDANT=0; DUMPMODE=none ;;
  A)  EPLB=0; REC=1; REDUNDANT=0; DUMPMODE=60 ;;
  B)  EPLB=1; REC=1; REDUNDANT=0; DUMPMODE=rebal ;;
  C)  EPLB=1; REC=1; REDUNDANT=64; DUMPMODE=rebal ;;
  *)  echo "group 必须是 A0|A|B|C"; exit 2 ;;
esac
case "$LOAD" in
  agentx) DATA=/home/chengyetai/workspace/history_exp/05_EPLB冗余专家_1P1D_2026-09-28/data/agentx_16k_512.json; IN=16384; OUTL=512 ;;
  random) DATA=$DIR/data/random_8k_64.json; IN=8192; OUTL=64 ;;
  *)      echo "load 必须是 agentx|random"; exit 2 ;;
esac

RUN="${GROUP}_${LOAD}_$(date +%Y%m%d_%H%M%S)"
OUT=$DIR/results/$RUN; mkdir -p "$OUT"/{metrics,npu,windows}
ORCH=$OUT/orchestrator.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }
say "===== RUN=$RUN group=$GROUP load=$LOAD dur=${DUR}s conc=$CONC  EPLB=$EPLB REC=$REC REDUNDANT=$REDUNDANT DUMPMODE=$DUMPMODE ====="

# ---------- 1. 起服务（落盘到 /data_lib/cyt） ----------
LOGDIR_ROOT=/data_lib/cyt EPLB=$EPLB REC=$REC REDUNDANT=$REDUNDANT RECD_BUF=2000 \
  "$DIR/serve4_dp.sh" start "$RUN" >>"$ORCH" 2>&1 || { say "启动失败"; exit 3; }
RUNLOG=/data_lib/cyt/$RUN/node_$NODE0

# ---------- 2. 等健康 ----------
ok=0
for i in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 "$EP/health" 2>/dev/null || echo 000)
  say "health=$code"
  [ "$code" = "200" ] && { ok=1; break; }
  sleep 20
done
[ "$ok" = 1 ] || { say "服务没起来，停掉"; "$DIR/serve4_dp.sh" stop >/dev/null 2>&1; exit 4; }

# ---------- 3. 开始采集 ----------
if [ "$REC" = 1 ]; then
  curl -sS -o /dev/null --max-time 30 -X POST "$EP/start_expert_distribution_record" && say "recorder started"
fi

dump(){ local why="$1"; local t0=$(date +%s.%N)
  curl -sS -o /dev/null -w '%{http_code}' --max-time 300 -X POST "$EP/dump_expert_distribution_record" >/dev/null 2>&1
  local t1=$(date +%s.%N)
  # 立刻删掉 stock 写出的 479MB 大文件（我们自己的窗口文件保留）
  timeout 60 ssh -T -o BatchMode=yes -o LogLevel=ERROR 10.120.72.$NODE0 \
    "rm -f /data_lib/cyt/$RUN/node_$NODE0/eplb_windows/expert_distribution_recorder_*.pt" >/dev/null 2>&1
  say "dump why=$why cost=$(python3 -c "print(round($t1-$t0,2))")s"
}
sample(){ local tag=$(date +%H%M%S)
  curl -sS --max-time 20 "$EP/metrics" > "$OUT/metrics/$tag.txt" 2>/dev/null || true
  for n in 19 20 23 25; do timeout 20 ssh -T -o BatchMode=yes -o LogLevel=ERROR 10.120.72.$n "npu-smi info" 2>/dev/null > "$OUT/npu/${tag}_npu$n.txt" || true; done
}

# ---------- 4. 打负载 ----------
python3 "$BENCH" --url "$EP" --dataset "$DATA" --out "$OUT/bench" \
  --duration "$DUR" --concurrency "$CONC" --input-tokens "$IN" --output-tokens "$OUTL" \
  > "$OUT/bench.log" 2>&1 &
BPID=$!
sleep 20; sample
LAST_DUMP=$(date +%s)
while kill -0 $BPID 2>/dev/null; do
  sleep 60
  kill -0 $BPID 2>/dev/null || break
  sample
  now=$(date +%s)
  if [ "$REC" = 1 ] && [ "$DUMPMODE" = "60" ] && [ $((now-LAST_DUMP)) -ge 55 ]; then
    dump periodic; LAST_DUMP=$now
  fi
done
wait $BPID; rc=$?
say "bench done rc=$rc"

# ---------- 5. 收尾 ----------
if [ "$REC" = 1 ]; then dump final; fi
sample
# 窗口 + 日志 归档
W=/data_lib/cyt/$RUN/node_$NODE0/eplb_windows
scp -q -o LogLevel=ERROR 10.120.72.$NODE0:"$W/*.pt" "$OUT/windows/" 2>/dev/null || true
for n in 19 20 23 25; do
  scp -q -o LogLevel=ERROR 10.120.72.$n:/data_lib/cyt/$RUN/node_$n/server.log "$OUT/logs_node_$n.log" 2>/dev/null || true
done
cat "$OUT"/logs_node_*.log 2>/dev/null | grep -oE "rebalance (start|end time=[0-9.]+s)" | sort | uniq -c > "$OUT/rebalance_events.txt" || true
ls -1 "$OUT/windows"/*.pt 2>/dev/null | wc -l > "$OUT/n_windows.txt"
say "窗口数=$(cat "$OUT/n_windows.txt")  归档完成"

# ---------- 6. 停服务 ----------
"$DIR/serve4_dp.sh" stop >/dev/null 2>&1
say "===== RUN 结束: $RUN ====="
echo "$RUN" >> "$DIR/results/run_index.txt"
