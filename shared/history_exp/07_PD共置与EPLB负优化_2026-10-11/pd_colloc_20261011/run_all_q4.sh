#!/usr/bin/env bash
# Q4Exp 全序列：4 组 x 2 负载 = 8 轮
set -u
DIR=/home/chengyetai/workspace/current_exp/pd_colloc_20261011
LOG=$DIR/results/q4exp_sequence.log
echo "[$(date +%F' '%H:%M:%S)] ================= 全序列开始 =================" >> "$LOG"
for LOAD in agentx random; do
  for G in A0 A B C; do
    echo "[$(date +%H:%M:%S)] >>> $G / $LOAD 开始" >> "$LOG"
    "$DIR/run_q4exp.sh" "$G" "$LOAD" 900 8 >> "$LOG" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] <<< $G / $LOAD 结束 rc=$rc" >> "$LOG"
  done
done
echo "[$(date +%F' '%H:%M:%S)] ================= 全序列结束 =================" >> "$LOG"
