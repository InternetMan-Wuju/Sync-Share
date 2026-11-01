#!/bin/bash
set -euo pipefail

# 4 机 2P2D PD 分离部署编排：
#   node 23: MemFabric store + prefill0 + router
#   node 25: prefill1
#   node 31: decode0
#   node 32: decode1
# 每台单机 TP16（16 卡跑一个角色实例）。

DIR="$(cd "$(dirname "$0")" && pwd)"
IMG=quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16
MODEL=/mnt/raid/user/data/models/glm-q

PREFILL0=10.120.72.23
PREFILL1=10.120.72.25
DECODE0=10.120.72.31
DECODE1=10.120.72.32
ROUTER_NODE=10.120.72.23

PF_PORT0=18423
PF_PORT1=18425
DC_PORT0=18431
DC_PORT1=18432
BOOTSTRAP=18998
ROUTER_PORT=18623
ROUTER_PROM=18723
MF_STORE_URL="tcp://${PREFILL0}:24727"

CTX="${CTX:-32768}"
MAX_REQ="${MAX_REQ:-64}"
MF_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-sdma}"

echo "===== 1. stop old EP64 (all 4 nodes) ====="
for h in ${PREFILL0} ${PREFILL1} ${DECODE0} ${DECODE1}; do
  ssh "$h" 'sudo docker rm -f glm52-ep64-decode-chengyetai 2>/dev/null || true'
done

echo "===== 2. start MemFabric store on ${PREFILL0} ====="
ssh "${PREFILL0}" "sudo docker rm -f pd-mfstore-glm52 2>/dev/null || true
sudo docker run -d --name pd-mfstore-glm52 --net=host --ipc=host \
  -e ASCEND_MF_STORE_URL=${MF_STORE_URL} \
  --entrypoint python ${IMG} -m memfabric_hybrid.launch_ascend_mf_store"

sleep 3

echo "===== 3. push serve_pd.sh to all nodes ====="
for h in ${PREFILL0} ${PREFILL1} ${DECODE0} ${DECODE1}; do
  scp "${DIR}/serve_pd.sh" "${h}:/tmp/serve_pd.sh"
done

echo "===== 4. start prefill instances ====="
ssh "${PREFILL0}" "CTX=${CTX} MAX_REQ=${MAX_REQ} MF_STORE_URL=${MF_STORE_URL} ASCEND_MF_TRANSFER_PROTOCOL=${MF_PROTOCOL} bash /tmp/serve_pd.sh prefill ${PF_PORT0}"
ssh "${PREFILL1}" "CTX=${CTX} MAX_REQ=${MAX_REQ} MF_STORE_URL=${MF_STORE_URL} ASCEND_MF_TRANSFER_PROTOCOL=${MF_PROTOCOL} bash /tmp/serve_pd.sh prefill ${PF_PORT1}"

echo "===== 5. start decode instances ====="
ssh "${DECODE0}" "CTX=${CTX} MAX_REQ=${MAX_REQ} MF_STORE_URL=${MF_STORE_URL} ASCEND_MF_TRANSFER_PROTOCOL=${MF_PROTOCOL} bash /tmp/serve_pd.sh decode ${DC_PORT0}"
ssh "${DECODE1}" "CTX=${CTX} MAX_REQ=${MAX_REQ} MF_STORE_URL=${MF_STORE_URL} ASCEND_MF_TRANSFER_PROTOCOL=${MF_PROTOCOL} bash /tmp/serve_pd.sh decode ${DC_PORT1}"

echo "===== 6. start router on ${ROUTER_NODE} ====="
ROUTER_CMD="python3 -m sglang_router.launch_router --pd-disaggregation --policy round_robin --prefill http://${PREFILL0}:${PF_PORT0} ${BOOTSTRAP} --prefill http://${PREFILL1}:${PF_PORT1} ${BOOTSTRAP} --decode http://${DECODE0}:${DC_PORT0} --decode http://${DECODE1}:${DC_PORT1} --host 0.0.0.0 --port ${ROUTER_PORT} --prometheus-port ${ROUTER_PROM}"
ssh "${ROUTER_NODE}" "sudo docker rm -f pd-router-glm52 2>/dev/null || true
sudo docker run -d --name pd-router-glm52 --net=host --ipc=host \
  -v ${MODEL}:/models/glm-q:ro \
  --entrypoint bash ${IMG} -c '${ROUTER_CMD}'"

echo "===== all submitted ====="
cat <<EOF
状态检查:
  for h in ${PREFILL0} ${PREFILL1} ${DECODE0} ${DECODE1}; do echo "== \$h =="; ssh \$h 'sudo docker ps --format "{{.Names}} {{.Status}}"'; done
  ssh ${ROUTER_NODE} 'sudo docker ps --format "{{.Names}} {{.Status}}" | grep pd-'
健康检查 (router):
  ssh ${ROUTER_NODE} 'curl -s -m 5 -o /dev/null -w "%{http_code}\\n" http://127.0.0.1:${ROUTER_PORT}/health'
日志:
  ssh ${PREFILL0} 'sudo docker logs --tail 50 pd-prefill-glm52'
EOF
