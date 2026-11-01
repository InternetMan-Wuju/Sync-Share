#!/bin/bash
set -euo pipefail

# per_pass 模式采集：跑 Dolly，并收集四机所有 rank 的 expert_distribution_recorder_*.pt。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/../dataset" && pwd)"

URL="${URL:-http://127.0.0.1:30000}"
CONTAINER=glm52-ep64-decode-chengyetai
NODES="${NODES:-10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32}"
PORT="${PORT:-30000}"

LIMIT="${LIMIT:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUN_NAME="${RUN_NAME:-dolly_perpass_$(date +%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${DATASET_DIR}/ep_runs/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"

echo "===== run dir ====="
echo "${RUN_DIR}"

echo "===== 0. health ====="
curl -s -m 5 -o /dev/null -w 'health_http=%{http_code}\n' "${URL}/health"

echo "===== 1. 清理各节点容器内旧 recorder 文件 ====="
for ip in ${NODES}; do
  ssh -o BatchMode=yes "${ip}" \
    "sudo -n docker exec ${CONTAINER} bash -c 'rm -f /tmp/expert_distribution_recorder_*.pt 2>/dev/null || true'" \
    >/dev/null 2>&1 || true
done

echo "===== 2. start record ====="
curl -s -X POST "${URL}/start_expert_distribution_record"
echo

echo "===== 3. run Dolly benchmark ====="
python3 "${DATASET_DIR}/bench_dolly.py" \
  --url "${URL}" \
  --jsonl "${DATASET_DIR}/databricks-dolly-15k.jsonl" \
  --out "${RUN_DIR}/dolly_bench.json" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

echo "===== 4. stop record ====="
curl -s -X POST "${URL}/stop_expert_distribution_record"
echo

echo "===== 5. dump record（每个 rank 各自写文件） ====="
curl -s -X POST "${URL}/dump_expert_distribution_record" >/dev/null
sleep 3

echo "===== 6. 收集四机所有 rank 的 .pt ====="
count=0
for ip in ${NODES}; do
  files=$(ssh -o BatchMode=yes "${ip}" \
    "sudo -n docker exec ${CONTAINER} bash -c 'ls /tmp/expert_distribution_recorder_*.pt 2>/dev/null'" \
    2>/dev/null || true)
  for f in ${files}; do
    base=$(basename "${f}")
    ssh -o BatchMode=yes "${ip}" "sudo -n docker exec ${CONTAINER} cat ${f}" \
      > "${RUN_DIR}/rank_${ip}_${base}" 2>/dev/null || true
    count=$((count + 1))
  done
done
echo "[collect] ${count} 个 .pt 文件"

echo "===== 7. 收集四机 metrics + npu-smi ====="
for ip in ${NODES}; do
  curl -s -m 10 "http://${ip}:${PORT}/metrics" -o "${RUN_DIR}/metrics_${ip}.prom"
  ssh -o BatchMode=yes "${ip}" "npu-smi info" > "${RUN_DIR}/npusmi_${ip}.txt" 2>&1 || true
done

echo "===== 8. summarize ====="
python3 "${SCRIPT_DIR}/summarize_dolly_run.py" --run-dir "${RUN_DIR}"

echo "===== done ====="
ls -l "${RUN_DIR}" | head -80
