#!/bin/bash
# 对一次实验的 run 目录做后处理，产出 ep_exp2.md §3 要的最终指标。
# 用法: bash ep_lab/analyze_run.sh <run_dir> [ep_size]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${1:?usage: analyze_run.sh <run_dir> [ep_size]}"
EP_SIZE="${2:-64}"
CONTAINER=glm52-ep64-decode-chengyetai
HOST=10.120.72.23

echo "===== analyze run: ${RUN_DIR} (EP=${EP_SIZE}) ====="

# 1) 请求级 + token 级性能（宿主机）
python3 "${SCRIPT_DIR}/perf_from_prom.py" \
  --run-dir "${RUN_DIR}" --out "${RUN_DIR}/final_perf.json"

# 2) 专家热度三件套 / 逐层 / rank 负载（容器内，宿主机无 torch）
PT="${RUN_DIR}/expert_distribution.pt"
if [ -f "${PT}" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${HOST}" \
    "sudo docker cp '${PT}' '${CONTAINER}:/tmp/expert_heat_in.pt' && \
     sudo docker cp '${SCRIPT_DIR}/analyze_expert_heat.py' '${CONTAINER}:/tmp/analyze_expert_heat.py' && \
     sudo docker exec '${CONTAINER}' python3 /tmp/analyze_expert_heat.py \
       --pt /tmp/expert_heat_in.pt --out /tmp/expert_heat_out.json --ep-size '${EP_SIZE}'" >/dev/null
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${HOST}" \
    "sudo docker cp '${CONTAINER}:/tmp/expert_heat_out.json' '${RUN_DIR}/expert_heat.json'"
  echo "[expert heat] -> ${RUN_DIR}/expert_heat.json"
else
  echo "[expert heat] WARNING: no expert_distribution.pt in ${RUN_DIR}"
fi

# 3) 数据完整性检查：列出关键产物
echo "===== 产物检查 ====="
for f in multi_bench.json expert_distribution.pt final_perf.json expert_heat.json \
         dataset_stats.json metrics_10.120.72.23.prom npusmi_10.120.72.23.txt; do
  if [ -e "${RUN_DIR}/${f}" ]; then
    echo "  [ok] ${f}"
  else
    echo "  [MISSING] ${f}"
  fi
done

echo "===== done ====="
