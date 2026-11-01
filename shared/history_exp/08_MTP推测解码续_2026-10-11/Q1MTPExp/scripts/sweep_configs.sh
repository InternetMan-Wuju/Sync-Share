#!/usr/bin/env bash
# 最小场景配置扫描：只验证 ①能否起服务 ②KV 池多大 ③一次最小 MTP 请求能否出 spec 统计
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
OUT=$DIR/results/sweep; mkdir -p "$OUT"
SUM=$OUT/summary.md
IMG16=quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16
IMG18=quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3
CONFIGS=(
  "A_v0516_deepep-auto_graph-on_mem0.88|$IMG16|auto|on|0.88"
  "B_v0516_deepep-auto_graph-off_mem0.88|$IMG16|auto|off|0.88"
  "C_v0516_deepep-normal_graph-off_mem0.92|$IMG16|normal|off|0.92"
  "D_v0516_deepep-normal_graph-off_mem0.88|$IMG16|normal|off|0.88"
)
printf '| 配置 | 镜像 | deepep | graph | MEM | health | KV池(token) | spec生效 | AL | 结论 |\n' > "$SUM"
printf '|---|---|---|---|---|---|---|---|---|---|\n' >> "$SUM"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name img deepep graph mem <<< "$cfg"
  run="${name}_$(date +%H%M%S)"; log="$OUT/$run.log"; : > "$log"
  echo "=== [$name] img=$(basename "$img") deepep=$deepep graph=$graph mem=$mem ===" | tee -a "$log"
  MTP=on IMG="$img" DEEPEP="$deepep" GRAPH="$graph" MEM="$mem" \
    bash "$DIR/scripts/serve_mtp_q1.sh" start "$run" >>"$log" 2>&1
  slog="$DIR/logs/$run/server.log"
  kv=""
  for i in $(seq 1 42); do sleep 10
    kv=$(grep -m1 -oE "#tokens: [0-9]+" "$slog" 2>/dev/null | grep -oE "[0-9]+")
    [ -n "$kv" ] && break
    sudo -n docker ps --filter name=q1-mtp-23 -q | grep -q . || break
  done
  health=000
  for i in $(seq 1 42); do
    health=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:30080/health 2>/dev/null)
    [ "$health" = "200" ] && break
    sudo -n docker ps --filter name=q1-mtp-23 -q | grep -q . || break
    sleep 10
  done
  spec="-"; al="-"; verdict="起不来/失败"
  if [ "$health" = "200" ]; then
    python3 "$DIR/scripts/quick_accept_test.py" --url http://127.0.0.1:30080 \
      --dataset "$DIR/datasets/min_real_64_32.json" --out "$OUT/$run.probe.json" --concurrency 1 >>"$log" 2>&1
    if [ -f "$OUT/$run.probe.json" ]; then
      read -r spec al <<< "$(python3 "$DIR/scripts/parse_probe.py" "$OUT/$run.probe.json")"
      verdict="✅ 跑通"
    else
      verdict="请求失败"
    fi
  else
    err=$(grep -oE "Capture cuda graph failed[^\"]{0,50}|[A-Za-z_]*Error[^\"]{0,50}" "$slog" 2>/dev/null | tail -1)
    spec="err"; verdict="${err:-起不来}"
  fi
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$name" "$(basename "$img")" "$deepep" "$graph" "$mem" "$health" "${kv:--}" "$spec" "$al" "$verdict" >> "$SUM"
  echo "--> [$name] health=$health kv=$kv spec=$spec AL=$al verdict=$verdict" | tee -a "$log"
  bash "$DIR/scripts/serve_mtp_q1.sh" stop >>"$log" 2>&1
  sleep 20
done
echo "扫描完成: $SUM"
