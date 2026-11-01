#!/bin/bash
set -euo pipefail

HOSTS=(10.120.72.23 10.120.72.25 10.120.72.31 10.120.72.32)
DOCKER_BIN="$(command -v docker || echo /usr/bin/docker)"
SUDOERS_FILE=/etc/sudoers.d/chengyetai-docker
RULE="chengyetai ALL=(ALL) NOPASSWD: ${DOCKER_BIN}"

for H in "${HOSTS[@]}"; do
  echo "===== setup ${H} ====="
  ssh -t "${H}" "
    echo '${RULE}' | sudo tee ${SUDOERS_FILE} >/dev/null
    sudo chmod 440 ${SUDOERS_FILE}
    sudo visudo -c
  "
done

echo "===== done ====="
echo "以后运行 sudo docker 不再需要密码。"
