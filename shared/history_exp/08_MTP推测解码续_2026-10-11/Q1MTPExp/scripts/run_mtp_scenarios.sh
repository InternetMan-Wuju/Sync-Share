#!/usr/bin/env bash
# 四组 workload x {MTP OFF, MTP ON} = 8 轮（AboutModels 口径）
# 每轮: 起服务 -> 等 health -> (清缓存+预热前缀) -> 打负载 DUR 秒 -> 归档 -> 停服务
set -uo pipefail
DIR=/home/chengyetai/workspace/current_exp/Q1MTPExp
IMG="${IMG:-quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3}"
SERVER="${SERVER:-$DIR/scripts/serve_mtp_q1.sh}"   # 单机（已验证可用；2 机 + normal 会崩）
DUR="${DUR:-900}"          # 每轮负载秒数
OUT="$DIR/results/scenarios"; mkdir -p "$OUT"
export LOADER=""      # 新镜像(v0.5.19)不能挂 v0.5.16 的 loader 补丁
export DEEPEP=normal  # MTP 与低延迟 DeepEP 不兼容（已实测）
export GRAPH=on       # 传着，但镜像在 normal 下会禁用

SUM="$OUT/summary.md"
printf '| 场景 | MTP | 并发 | 完成请求 | 总输出tok | 输出tok/s | 输出tok/s/NPU | TPOT均(ms) | TTFT均(s) | AL(加权) | 接受率 | 命中率 | **P1/P2/P3** |\n' > "$SUM"
printf '|---|---|---|---|---|---|---|---|---|---|---|---|---|\n' >> "$SUM"

# 场景: 名字|数据集|并发
SCEN=(
  "w1_8k_1k|$DIR/datasets/w1_8k_1k.json|4|0"
  "w2_32k_1k|$DIR/datasets/w2_32k_1k.json|1|0"
  "w3_1k_32k|$DIR/datasets/w3_1k_32k.json|1|2048"
  "w4_prefix32k_300|$DIR/datasets/w4_prefix32k_300.json|1|0"
)
for mode in on; do      # 只跑 MTP ON（用户指示）
  # ---- 每个模式只起一次服务，连跑 4 个输入 ----
  srun="svc_mtp${mode}_$(date +%H%M%S)"; slog="$DIR/logs/$srun/server.log"
  echo "### [服务] mode=$mode run=$srun" | tee -a "$OUT/run.log"
  MTP="$mode" IMG="$IMG" BACKEND=deepep DEEPEP=normal GRAPH=on MEM=0.90 MRR=8 MAXPREF=32768 \
    bash "$SERVER" start "$srun" >>"$OUT/run.log" 2>&1
  kv=$(grep -m1 -oE "#tokens: [0-9]+" "$slog" 2>/dev/null | grep -oE "[0-9]+")
  health=000
  for i in $(seq 1 60); do
    health=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:30080/health 2>/dev/null)
    [ "$health" = "200" ] && break; sleep 15
  done
  echo "### [服务] health=$health kv=$kv" | tee -a "$OUT/run.log"
  for s in "${SCEN[@]}"; do
    IFS='|' read -r name ds conc maxtok <<< "$s"
    run="${name}_mtp${mode}_$(date +%H%M%S)"; log="$OUT/$run.log"; : > "$log"
    echo "=== [$run] conc=$conc dur=${DUR}s kv=$kv health=$health ===" | tee -a "$log"
    if [ "$health" = "200" ]; then
      WP=$(python3 -c "import json;print(json.load(open('$ds')).get('shared_prefix_tokens',0))")
      timeout $((DUR+900)) python3 "$DIR/scripts/quick_accept_test.py" --url http://127.0.0.1:30080 \
        --dataset "$ds" --out "$OUT/$run.json" --concurrency "$conc" \
        --flush-cache $( [ "$WP" -gt 0 ] && echo "--warmup-prefix $WP" ) \
        $( [ "$maxtok" -gt 0 ] && echo "--max-tokens $maxtok" ) >>"$log" 2>&1
      if [ -f "$OUT/$run.json" ]; then
        read -r done tok tps per tpot ttft al rate hit pos <<< "$(python3 "$DIR/scripts/summarize_run.py" "$OUT/$run.json" "$kv")"
        printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
          "$name" "$mode" "$conc" "$done" "$tok" "$tps" "$per" "$tpot" "$ttft" "$al" "$rate" "$hit" "$pos" >> "$SUM"
        echo "  -> 完成=$done tok=$tok tok/s=$tps per-NPU=$per TPOT=${tpot}ms TTFT=${ttft}s AL=$al 命中=$hit" | tee -a "$log"
      else
        echo "  请求失败" | tee -a "$log"
      fi
    fi
    sleep 15
  done
  bash "$SERVER" stop >>"$OUT/run.log" 2>&1
  sleep 20
done
echo "全部完成: $SUM"
