#!/bin/bash
set -euo pipefail

HOSTS=(10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32)
CONTAINERS="pd-prefill-glm52 pd-decode-glm52 pd-mfstore-glm52 pd-router-glm52"

for h in "${HOSTS[@]}"; do
  echo "===== stop ${h} ====="
  ssh "$h" "for c in ${CONTAINERS}; do sudo docker rm -f \$c 2>/dev/null || true; done"
done

echo "===== all stopped ====="
