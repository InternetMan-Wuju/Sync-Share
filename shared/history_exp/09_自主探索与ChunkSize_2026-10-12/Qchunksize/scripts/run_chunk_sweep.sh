#!/usr/bin/env bash
# 任务2：Chunk Size 扫描（在目标机上本地跑；MTP 关；DeepEP auto + Graph on）
# 用法: CHUNKS="1024 4096 8192 16384" CONCS="4 8" SCEN=w1_8k_1k bash run_chunk_sweep.sh
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Qchunksize
SERVE="$DIR/scripts/serve_chunk.sh"
CLIENT="$DIR/scripts/chunk_client.py"
SUMTOOL="$DIR/scripts/summarize_chunk.py"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="/data_lib/cyt/qchunk/results/$RUN_ID"; mkdir -p "$OUT"
LOGROOT="/data_lib/cyt/qchunk/$RUN_ID"; mkdir -p "$LOGROOT"
CHUNKS="${CHUNKS:-1024 4096 8192 16384}"
CONCS="${CONCS:-4 8}"
export MEM="${MEM:-0.92}" MAXPREF="${MAXPREF:-32768}" GRAPH="${GRAPH:-on}" DEEPEP="${DEEPEP:-auto}"
export IMG="${IMG:-quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3}"
# 场景: name|dataset文件|每条请求上限(0=全用)
declare -A DS=( [w1_8k_1k]="w1_8k_1k.json" [w2_32k_1k]="w2_32k_1k.json" [w3_1k_32k]="w3_1k_32k.json" [w4_prefix32k_300]="w4_prefix32k_300.json" )
SCEN="${SCEN:-w1_8k_1k}"
DSFILE="$DIR/datasets/${DS[$SCEN]}"
SUMMARY="$OUT/summary.md"
printf '# Chunk Size sweep run %s  场景=%s  镜像=%s  MEM=%s MAXPREF=%s DeepEP=%s GRAPH=%s\n\n' \
  "$RUN_ID" "$SCEN" "$IMG" "$MEM" "$MAXPREF" "$DEEPEP" "$GRAPH" > "$SUMMARY"
printf '| 场景 | 并发 | 请求数 | Chunk | MEM | MAXPREF | Graph生效 | KV池 | 吞吐(tok/s/NPU) | TTFT p50(s) | TTFT p95(s) | TPOT p50(ms) | TPOT p95(ms) | E2E p50(s) | 命中率 | ok | wall(s) |\n' >> "$SUMMARY"
printf '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n' >> "$SUMMARY"

health_wait(){ local i h; for i in $(seq 1 80); do h=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 "http://127.0.0.1:$PORT/health" 2>/dev/null); [ "$h" = 200 ] && { echo 200; return; }; sleep 10; done; echo "$h"; }
PORT="${PORT:-30080}"

for chunk in $CHUNKS; do
  srun="chunk${chunk}_$(date +%H%M%S)"
  export NAME="qchunk-${RUN_ID}-${chunk}"
  echo "### [服务] chunk=$chunk run=$srun $(date +%T)" | tee -a "$OUT/run.log"
  CHUNK="$chunk" bash "$SERVE" start "$srun" >>"$OUT/run.log" 2>&1 || break
  # 真实日志目录（serve 内部用 /data_lib/cyt/qchunk/<run_id>，这里找最新）
  slog=$(ls -t /data_lib/cyt/qchunk/chunk${chunk}_*/server.log 2>/dev/null | head -1)
  h=$(health_wait)
  # graph 是否真生效
  geff=unknown
  if [ -n "$slog" ]; then
    if grep -q "Cuda graph is disabled" "$slog"; then geff=disabled
    elif grep -q "npu graph: True" "$slog"; then geff=on
    else geff=none; fi
    kv=$(grep -m1 -oE "#tokens: [0-9]+" "$slog" | grep -oE "[0-9]+")
  fi
  echo "### [服务] chunk=$chunk health=$h graph=$geff kv=${kv:--} log=$slog $(date +%T)" | tee -a "$OUT/run.log"
  if [ "$h" != 200 ]; then
    echo "!!! chunk=$chunk 起服务失败(health=$h)，中止整轮扫描（见 $slog）" | tee -a "$OUT/run.log"
    CHUNK="$chunk" bash "$SERVE" stop >>"$OUT/run.log" 2>&1
    break
  fi
  for c in $CONCS; do
    n=$(( c * 2 )); [ "$SCEN" = w2_32k_1k ] && n=$c; [ "$SCEN" = w3_1k_32k ] && n=$c
    rid="${SCEN}_chunk${chunk}_c${c}"; log="$OUT/$rid.log"; jf="$OUT/$rid.json"
    MAXX=0
    : > "$log"
    echo "=== [$rid] conc=$c reqs=$n health=$h kv=${kv:--} ===" | tee -a "$log"
    if [ "$h" = 200 ]; then
      timeout 5400 python3 "$CLIENT" --url "http://127.0.0.1:$PORT" --dataset "$DSFILE" \
        --out "$jf" --concurrency "$c" --limit "$n" --npu 16 --flush-cache >>"$log" 2>&1
      if [ -f "$jf" ]; then
        python3 "$SUMTOOL" "$jf" "$chunk" "$MEM" "$MAXPREF" "$geff" "${kv:--}" >> "$SUMMARY"
        tail -3 "$log" | tee -a "$OUT/run.log"
      else
        echo "  请求失败（无 json）" | tee -a "$log" "$OUT/run.log"
      fi
    fi
    sleep 20
  done
  CHUNK="$chunk" bash "$SERVE" stop >>"$OUT/run.log" 2>&1
  sleep 45
done
echo "DONE: $SUMMARY"
