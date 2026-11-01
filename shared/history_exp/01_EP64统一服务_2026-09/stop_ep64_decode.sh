#!/bin/bash
set -euo pipefail

HOSTS=(10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32)
CONTAINER=glm52-ep64-decode-chengyetai
DOCKER="${DOCKER:-sudo docker}"

for H in "${HOSTS[@]}"; do
  echo "===== stop ${H} ====="
  ssh "${H}" "${DOCKER} rm -f ${CONTAINER} 2>/dev/null || true"
done

echo "===== all stopped ====="
echo "检查 NPU 是否已释放："
echo "  ssh 10.120.72.23 'npu-smi info | grep -E \"No running processes\" | wc -l'"
