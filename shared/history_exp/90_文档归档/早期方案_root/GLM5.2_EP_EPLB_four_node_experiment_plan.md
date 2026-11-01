# GLM-5.2 四机 EP / EPLB 实验方案

> 说明：本文是实验设计与部署思路，不是已在本机执行的启动脚本。执行前需要根据四台机器的 IP、网卡、SGLang 版本重新核对参数。

## 1. 目标

在四台 Ascend NPU 机器上，以 SGLang 对 GLM-5.2-w8a8 做专家并行实验，重点测量：

- Expert Parallelism，即 EP
- EPLB，即专家并行负载均衡
- 专家分布、rank 负载、端到端性能

## 2. 模型与硬件

- 模型路径：`/mnt/raid/user/data/models/glm-q`
- 模型类型：GLM-5.2-w8a8
- routed experts：256
- 每 token 激活专家数：8
- 当前单机：8 张 Ascend910，16 个 chip，每 chip 64 GB
- 四机总 chip：64

## 3. EP 实验的拓扑

EP 实验必须是：

> 一个分布式 SGLang 服务，把 MoE 专家切到多个 rank 上，通过 all-to-all 分发 token。

不是四台机器各跑一个完整副本。

Ascend 上使用 DeepEP 时，关键约束：

```text
ep_size == tp_size
```

所以：

| 阶段 | 设备数 | 拓扑 |
|---|---|---|
| A | 16 chip | 单机 `tp16 ep16` |
| B | 32 chip | 两机 `tp32 ep32 nnodes2` |
| C | 64 chip | 四机 `tp64 ep64 nnodes4` |

## 4. 四机 EP 启动模板

在四台机器上运行同一个脚本，脚本根据本机 IP 自动确定 `node-rank`。

```bash
IPS=('<node1_ip>' '<node2_ip>' '<node3_ip>' '<node4_ip>')
IP_MASTER="${IPS[0]}:5000"
MODEL_PATH=/mnt/raid/user/data/models/glm-q

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=<实际跨节点网卡>
export GLOO_SOCKET_IFNAME=<实际跨节点网卡>

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

LOCAL_IP=$(hostname -I | awk '{print $1}')

for i in "${!IPS[@]}"; do
  if [[ "$LOCAL_IP" == "${IPS[$i]}" ]]; then
    python3 -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --attention-backend ascend \
      --device npu \
      --tp-size 64 \
      --ep-size 64 \
      --nnodes 4 \
      --node-rank "$i" \
      --dist-init-addr "$IP_MASTER" \
      --quantization modelslim \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --mem-fraction-static 0.7 \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name glm-5
    break
  fi
done
```

注意：

- `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 必须填四台机器之间的高速网卡。
- 如果 EP64 启动失败或通信太慢，先退到 EP32 两机，确认 DeepEP 和网络没问题。

## 5. EPLB 实验开关

SGLang 基础开关：

```text
--enable-eplb
```

先查当前安装版本支持的参数：

```bash
python3 -m sglang.launch_server --help | rg -i 'eplb|expert|ep-'
```

常见相关参数可能包括：

```text
--enable-eplb
--eplb-rebalance-num-iterations
--eplb-rebalance-layers-per-chunk
--ep-num-redundant-experts
--init-expert-location
--eplb-algorithm
```

具体以本机 SGLang 版本为准。

## 6. 实验矩阵

| 实验 | EP size | EPLB | 说明 |
|---|---|---|---|
| baseline | 16 / 32 / 64 | off | 静态专家放置 |
| EPLB | 16 / 32 / 64 | on | 动态专家负载均衡 |
| EPLB + redundant | 32 / 64 | on | 给热门专家增加物理副本 |

建议优先完成：

```text
EP64 + EPLB off
EP64 + EPLB on
```

## 7. 专家负载测量

SGLang 有 native API 记录专家分布，通常涉及：

```text
/start_expert_distribution_record
/dump_expert_distribution_record
/stop_expert_distribution_record
```

压测一段时间后导出专家分布，观察：

1. 每个逻辑专家被选中的次数。
2. 每个物理专家/rank 实际处理的 token 数。
3. 各 rank 的负载峰值、均值和平衡度：

```text
balancedness = avg_load / max_load
```

越接近 1 越均衡。

同时记录端到端指标：

- output tokens/s
- TTFT p50 / p99
- TPOT / ITL p50 / p99
- all-to-all dispatch / combine 时间
- EPLB 重排次数和耗时

## 8. 实验步骤

1. 确认四台机器 NPU 数量一致：

```bash
npu-smi info
```

2. 确认四台机器都能访问：

```text
/mnt/raid/user/data/models/glm-q
```

3. 先跑单机 EP16，确认权重、`modelslim` 量化、`glm_moe_dsa` 架构能正常加载。

4. 再跑两机 EP32，验证 DeepEP 跨节点通信。

5. 最后跑四机 EP64，对比 EPLB off / on。

6. 固定同一批压测 prompt，分别跑 prefill-heavy、decode-heavy、混合负载。

7. 记录专家分布和端到端指标，比较 EPLB 是否让 `balancedness` 提升，同时确认 P99 没有因为重排开销变差。

## 9. 参考来源

- [SGLang Expert Parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
- [SGLang GLM-5.2 Ascend NPU tutorial](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2)
- [SGLang Ascend NPU Parameter Tuning](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning)
- [SGLang Native API: expert distribution recording](https://docs.sglang.io/docs/basic_usage/native_api)
- [Waterfill and LPLB load balancing](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb)
