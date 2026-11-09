#!/usr/bin/env bash
set -euo pipefail
export DOCKER_HOST=unix:///tmp/chengyetai-rootless-docker/docker.sock
TASK_RUN=$(cd "$(dirname "$0")/.." && pwd)
TASK_WEIGHTS=/home/chengyetai/workspace/history_exp/09_自主探索与ChunkSize_2026-10-12/autonomous_20261012/weights
mkdir -p "$TASK_RUN/local/weight_audit" "$TASK_RUN/local/tmp"
printf '%s\n' '{"mode":"off","profile":false,"verify":false,"request":"startup","implementation_revision":2}' > "$TASK_RUN/local/control.json"
devices=();for i in {0..15};do devices+=(--device "/dev/davinci$i");done
exec docker run --rm --name cyt-decision-23 --net host --shm-size 16g --ulimit nofile=65535:65535 --security-opt seccomp=unconfined \
 "${devices[@]}" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /var/queue_schedule:/var/queue_schedule \
 -v /data_lib/data/models/glm-q:/models/glm-q:ro -v "$TASK_WEIGHTS/loader":/study/weights/loader:ro -v "$TASK_WEIGHTS/publisher":/study/weights/publisher:ro \
 -v "$TASK_RUN":/runexp:rw -v /data_lib/cyt/sglang-cache-e0:/root/.cache/sglang:rw \
 -e CYT_EP16_EXPERIMENT=1 -e CYT_RUN=/runexp \
 --entrypoint bash quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3 /runexp/scripts/server.sh
