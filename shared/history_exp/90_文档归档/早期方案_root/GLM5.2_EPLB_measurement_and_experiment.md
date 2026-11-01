# GLM-5.2 EPLB 专家并行负载均衡：测量与实验方案

> 说明：本文基于本机只读检查与联网检索整理，未在本机实际运行推理实验，也未修改任何模型文件。

## 1. 结论

本机模型仓库中确认存在 GLM-5.2：

- 完整模型目录：`/mnt/raid/user/data/models/glm5.2`
- 量化模型目录：`/mnt/raid/user/data/models/glm-q`，README 显示为 GLM-5.2-w8a8 量化版
- 模型 config 关键参数：
  - `model_type: glm_moe_dsa`
  - `architectures: ["GlmMoeDsaForCausalLM"]`
  - `n_routed_experts: 256`
  - `num_experts_per_tok: 8`
  - `n_shared_experts: 1`
  - `num_hidden_layers: 78`
  - `max_position_embeddings: 1048576`

GLM-5.2 是典型的大 MoE 架构，适合评估 EPLB。

## 2. EPLB 测的是什么

EPLB 是 Expert Parallel Load Balancing，即专家并行负载均衡。

它主要衡量：

> 每个 EP rank 上，各个专家实际处理的 token 数量是否均匀。

它不是训练阶段的 aux loss 负载均衡，而是推理部署时的专家放置与动态重排。

vLLM 内部使用 token 数作为专家负载单位：

- `expert_load_pass`：当前 forward pass 中每个物理专家的负载，形状为 `(num_moe_layers, num_physical_experts)`。
- `expert_load_window`：专家负载滑动窗口，形状为 `(window_size, num_moe_layers, num_physical_experts)`。

## 3. 核心测量指标

### 3.1 全局均衡度

```text
balancedness = avg_tokens / max_tokens
```

- `avg_tokens`：所有 EP rank 的平均负载。
- `max_tokens`：负载最高 rank 的负载。
- 完全均衡时为 1，越小越不均衡。

日志示例：

```text
EPLB step: 2500: avg_tokens=1024, max_tokens=2048, balancedness=0.5000
```

### 3.2 逐层峰均比

```text
imbalance_layer = max_rank_load / mean_rank_load
```

- 最小值为 1。
- 值越大，说明该 MoE 层专家负载越不均衡。

### 3.3 重排前后对比

- `current`：当前实际峰均比。
- `update`：算法估计重排后的峰均比。

vLLM-Ascend 中可搜索日志 `[Expert Hotness]`，获取每层不同 rank 的峰均比及 mean / max。

### 3.4 其他观测指标

- 重排次数
- 重排耗时
- 专家权重 D2D 传输耗时
- TTFT p50 / p99
- TPOT mean / p99
- ITL mean / p99
- output tokens/s
- requests/s
- NPU/GPU 利用率
- all-to-all 通信耗时

## 4. 如何开启测量

### 4.1 NVIDIA / 上游 vLLM

必须同时开启：

```text
--enable-expert-parallel
--enable-eplb
```

新版配置示例：

```bash
vllm serve /path/to/GLM-5.2 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{
    "window_size": 1000,
    "step_interval": 3000,
    "num_redundant_experts": 16,
    "log_balancedness": true,
    "log_balancedness_interval": 1,
    "use_async": true,
    "policy": "default"
  }'
```

旧版 vLLM 可能使用以下参数：

```text
--eplb-window-size
--eplb-step-interval
--eplb-log-balancedness
--num-redundant-experts
```

注意：

- `num_redundant_experts` 通常需要满足：

```text
(256 + N) % EP_size == 0
```

- `EP_size = tensor_parallel_size * data_parallel_size`。
- 过频繁重排会伤害性能。上游 vLLM 一般建议先试 `window_size=1000`、`step_interval=3000`。

### 4.2 vLLM-Ascend

本机是 Ascend 环境，本地有 `glm-q`（GLM-5.2-w8a8）。Ascend vLLM 的动态 EPLB 需要环境变量保护开关：

```bash
export DYNAMIC_EPLB="true"

vllm serve /mnt/raid/user/data/models/glm-q \
  --served-model-name glm-52 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization ascend \
  --additional-config '{
    "eplb_config": {
      "dynamic_eplb": true,
      "expert_heat_collection_interval": 600,
      "algorithm_execution_interval": 50,
      "eplb_policy_type": 2,
      "num_redundant_experts": 16,
      "eplb_heat_collection_stage": "prefill"
    }
  }'
```

Ascend EPLB 常用参数：

| 参数 | 含义 |
|---|---|
| `expert_heat_collection_interval` | 收集专家热度的时间间隔 |
| `algorithm_execution_interval` | 执行重排算法的间隔 |
| `eplb_policy_type` | EPLB 算法策略 |
| `eplb_heat_collection_stage` | `all` / `prefill` / `decode` |

`eplb_policy_type`：

- `0`：Random
- `1`：DefaultEplb
- `2`：SwiftBalanceEplb，Ascend 文档推荐
- `3`：FlashLB，适合高频波动的负载

`eplb_heat_collection_stage`：

- `all`：prefill 和 decode 都统计
- `prefill`：只优化 prefill，更关心 TTFT
- `decode`：只优化 decode，更关心 TPOT

本机示例中 TP8 × DP2，EP 规模相当于 16。GLM-5.2 有 256 个 routed experts：

```text
N=0  -> 256/16 = 16
N=16 -> 272/16 = 17
N=32 -> 288/16 = 18
```

都满足整除条件，建议从 `N=16` 开始测。

## 5. 实验设计

### 5.1 固定对照组

至少包含一组 EPLB 关闭的 baseline：

```text
baseline: --enable-expert-parallel，但不启用 EPLB
experiment: --enable-expert-parallel + EPLB
```

其他条件必须一致：

- 同一个模型路径
- 同一个 `max_model_len`
- 同一个 `max_num_seqs`
- 同一个 `max_num_batched_tokens`
- 同一组 prompt / request trace
- 同一 seed
- 同一 NPU/GPU 拓扑

### 5.2 准备负载

至少三组：

1. prefill 为主：输入 4k / 8k / 32k tokens，输出较短，主要看 TTFT。
2. decode 为主：输入 128 / 512 tokens，输出 2k / 4k tokens，高并发，主要看 TPOT / ITL / 吞吐。
3. 混合流量：尽量接近线上真实比例，例如 60% prefill + 40% decode。

### 5.3 实验矩阵

| 实验 | EPLB | policy | redundant experts | heat stage |
|---|---|---|---|---|
| A | off | - | 0 | - |
| B | on | SwiftBalance | 16 | prefill |
| C | on | SwiftBalance | 32 | prefill |
| D | on | FlashLB | 16 | prefill |
| E | on | SwiftBalance | 16 | decode |
| F | on | SwiftBalance | 16 | all |
| G | static map | - | - | - |

### 5.4 静态 EPLB 对比

Ascend 还可以做静态 EPLB 对比：

记录专家映射：

```text
DYNAMIC_EPLB=true 或 EXPERT_MAP_RECORD=true
expert_map_record_path=/path/to/eplb.json
```

加载已记录映射：

```text
expert_map_path=/path/to/eplb.json
```

这样可以比较：

- 运行时动态重排
- 离线固化专家映射

### 5.5 采集内容

负载均衡侧：

- `balancedness`
- `avg_tokens / max_tokens`
- 逐层 `imbalance`
- 重排前后 `current_mean / current_max`
- 重排前后 `update_mean / update_max`
- 重排次数和耗时
- 专家权重 D2D 传输耗时

性能侧：

- TTFT p50 / p99
- TPOT mean / p99
- ITL mean / p99
- output tokens/s
- requests/s
- NPU/GPU 利用率
- all-to-all 通信耗时

## 6. 结果判断标准

不是 `balancedness` 越高就一定更好，最终判断需要结合端到端性能：

1. `balancedness` 或逐层峰均比明显改善。
2. TTFT / TPOT 的 P99 没有因为重排开销变差。
3. 吞吐不下降，或下降在可接受范围。
4. 重排频率没有过高，避免为了均衡而频繁 stop-the-world。

常见结论：

- 如果负载不均衡且长 prefill 场景明显，开 EPLB 通常对 TTFT/P99 有收益。
- 如果输入序列很短、并发较低，或每 die 专家数很少，EPLB 收益可能抵不过开销，Ascend 文档也建议先 benchmark。
- 如果 `balancedness` 提高但 P99 变差，优先调大 `algorithm_execution_interval` / `step_interval`，或减少 `num_redundant_experts`。

## 7. 参考来源

- [vLLM-Ascend: Expert Parallelism Load Balancer](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/expert_parallelism_load_balancer.html)
- [vLLM: EPLBConfig / ParallelConfig](https://docs.vllm.ai/en/v0.14.1/api/vllm/config/parallel/)
- [vLLM: EPLB metrics RFC](https://github.com/vllm-project/vllm/issues/30696)
- [vLLM-Ascend: GLM-5.2 deployment tutorial](https://github.com/vllm-project/vllm-ascend/blob/8a1b1cec/docs/source/tutorials/models/GLM5.2.md?plain=1)
- [vLLM forum: How to configure EPLB](https://discuss.vllm.ai/t/how-to-configure-the-parameters-of-eplb/1426)
- [vLLM-Ascend PR: Expose experts hotness metrics](https://github.com/vllm-project/vllm-ascend/pull/9536)
