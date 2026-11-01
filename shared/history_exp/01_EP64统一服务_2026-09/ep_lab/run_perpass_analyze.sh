#!/bin/bash
set -euo pipefail

# 在容器内（有 torch）合并 per_pass 的 64 个 rank dump，计算 rank 负载。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER=glm52-ep64-decode-chengyetai
RUN_DIR="${1:?usage: $0 <run_dir>}"
EP_SIZE="${2:-64}"

TAR="/tmp/perpass_pt_$$.tar"

if ! ls "${RUN_DIR}"/rank_*.pt >/dev/null 2>&1; then
  echo "没有找到 rank_*.pt: ${RUN_DIR}" >&2
  exit 1
fi

tar -cf "${TAR}" -C "${RUN_DIR}" $(ls "${RUN_DIR}"/rank_*.pt | xargs -n1 basename)

sudo -n docker cp "${TAR}" "${CONTAINER}:/tmp/perpass_pt.tar"
sudo -n docker cp "${SCRIPT_DIR}/analyze_perpass_rank_load.py" \
  "${CONTAINER}:/tmp/analyze_perpass_rank_load.py"

sudo -n docker exec "${CONTAINER}" bash -c '
  rm -rf /tmp/perpass && mkdir -p /tmp/perpass
  tar -xf /tmp/perpass_pt.tar -C /tmp/perpass
  python3 /tmp/analyze_perpass_rank_load.py --dir /tmp/perpass --ep-size ${EP_SIZE} --out /tmp/perpass_rank_load.json
'

sudo -n docker cp "${CONTAINER}:/tmp/perpass_rank_load.json" \
  "${RUN_DIR}/perpass_rank_load.json"
rm -f "${TAR}"

echo "[done] -> ${RUN_DIR}/perpass_rank_load.json"
