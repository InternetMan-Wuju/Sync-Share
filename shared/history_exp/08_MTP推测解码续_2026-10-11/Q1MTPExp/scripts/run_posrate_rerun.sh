#!/usr/bin/env bash
# 补跑 w1/w2/w3，用新客户端抓 per-position 接受率（P1/P2/P3）
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
OUT="$DIR/results/posrate"; mkdir -p "$OUT"
IMG=${IMG:-quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3}
SUM=$OUT/summary.md
printf '| 场景 | 并发 | 完成 | 输出tok/s | /NPU | TPOT(ms) | TTFT(s) | AL | 接受率 | P1/P2/P3 |\n|---|---|---|---|---|---|---|---|---|---|\n' > "$SUM"
SCEN=("w1_8k_1k|$DIR/datasets/w1_8k_1k.json|4|0" "w2_32k_1k|$DIR/datasets/w2_32k_1k.json|1|0" "w3_1k_32k|$DIR/datasets/w3_1k_32k.json|1|2048")
export LOADER=""
sr="svc_posrate_$(date +%H%M%S)"
MTP=on IMG="$IMG" BACKEND=deepep DEEPEP=normal GRAPH=on MEM=0.90 MRR=8 MAXPREF=32768 \
  bash "$DIR/scripts/serve_mtp_q1.sh" start "$sr" >>"$OUT/run.log" 2>&1
slog="$DIR/logs/$sr/server.log"
for i in $(seq 1 60); do
  h=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:30080/health 2>/dev/null)
  [ "$h" = "200" ] && break; sleep 15
done
kv=$(grep -m1 -oE "#tokens: [0-9]+" "$slog" 2>/dev/null | grep -oE "[0-9]+")
echo "### health=$h kv=$kv" | tee -a "$OUT/run.log"
for s in "${SCEN[@]}"; do
  IFS='|' read -r name ds conc maxtok <<< "$s"
  run="${name}_$(date +%H%M%S)"; log="$OUT/$run.log"; : > "$log"
  echo "=== [$run] conc=$conc ===" | tee -a "$log"
  timeout 2400 python3 "$DIR/scripts/quick_accept_test.py" --url http://127.0.0.1:30080 \
    --dataset "$ds" --out "$OUT/$run.json" --concurrency "$conc" --flush-cache \
    $( [ "$maxtok" -gt 0 ] && echo "--max-tokens $maxtok" ) >>"$log" 2>&1
  if [ -f "$OUT/$run.json" ]; then
    read -r done tok tps per tpot ttft al rate hit pos <<< "$(python3 "$DIR/scripts/summarize_run.py" "$OUT/$run.json" "$kv")"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
      "$name" "$conc" "$done" "$tps" "$per" "$tpot" "$ttft" "$al" "$rate" "$pos" >> "$SUM"
    echo "  -> AL=$al 接受率=$rate P1/P2/P3=$pos" | tee -a "$log"
  else
    echo "  失败" | tee -a "$log"
  fi
  sleep 15
done
bash "$DIR/scripts/serve_mtp_q1.sh" stop >/dev/null 2>&1
echo "补跑完成: $SUM"
