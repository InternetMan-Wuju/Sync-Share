# GLM-5.2 性能榜单（Ascend 910C）

> 说明
> - 本文件只保留 **910C** 的配置与榜单。
> - 每个配置 **只叠加一个优化技术** 并报数，能超过 baseline 才记入榜单（打榜式推进）；
> - 报数必须能解释"为什么变好"，只报数字不算结论。

---

## 1. 默认配置（910C）

```bash
#!/usr/bin/env bash
# 用途：集中定义 GLM-5.2 Ascend united 单机服务的镜像、模型、端口和推理参数。

# ------------------------------ 服务监听 ------------------------------
# SERVICE_HOST 是 SGLang 实际监听地址；0.0.0.0 表示监听所有本机 IPv4 地址。
SERVICE_HOST=${SERVICE_HOST:-0.0.0.0}
SERVICE_PORT=${SERVICE_PORT:-8000}
# 留空时自动读取 COMM_IFNAME 的第一个 IPv4 地址，用于健康检查和外部访问提示。
SERVICE_IP=${SERVICE_IP:-}

# ------------------------------ 镜像与模型 ------------------------------
IMAGE=${IMAGE:-quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16}
MODEL_DIR=${MODEL_DIR:-/mnt/raid/user/data/models/glm-q}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5}
COMM_IFNAME=${COMM_IFNAME:-bond4}
HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-50000}

# ------------------------------ 并行与显存 ------------------------------
NPU_DEVICE_COUNT=${NPU_DEVICE_COUNT:-16}
TP_SIZE=${TP_SIZE:-16}
DP_SIZE=${DP_SIZE:-2}
EP_SIZE=${EP_SIZE:-16}
SHM_SIZE=${SHM_SIZE:-16g}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}

# ------------------------------ 推理参数 ------------------------------
CONTEXT_LENGTH=${CONTEXT_LENGTH:-180000}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-32768}
TOKENIZER_WORKER_NUM=${TOKENIZER_WORKER_NUM:-4}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-9000}

# DP Attention/LMHead 和 DeepEP 默认开启；united 模式推荐 DeepEP auto。
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-1}
ENABLE_DP_LM_HEAD=${ENABLE_DP_LM_HEAD:-1}
ENABLE_DEEPEP=${ENABLE_DEEPEP:-1}
DEEPEP_MODE=${DEEPEP_MODE:-auto}

# Decode graph 只作用于 united 服务中的 Decode 阶段。
ENABLE_DECODE_GRAPH=${ENABLE_DECODE_GRAPH:-1}
DECODE_GRAPH_MAX_BS=${DECODE_GRAPH_MAX_BS:-32}
DECODE_GRAPH_BATCHES=${DECODE_GRAPH_BATCHES:-"8 16 24 32"}

# ------------------------------ 管理参数 ------------------------------
UNITED_CONTAINER=${UNITED_CONTAINER:-glm52-united}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
ENABLE_SYSTEM_TUNING=${ENABLE_SYSTEM_TUNING:-1}
SGLANG_NPU_PROFILING=${SGLANG_NPU_PROFILING:-0}
REFUSE_IF_PD_RUNNING=${REFUSE_IF_PD_RUNNING:-1}

export SERVICE_HOST SERVICE_PORT SERVICE_IP
export IMAGE MODEL_DIR SERVED_MODEL_NAME COMM_IFNAME HCCL_IF_BASE_PORT
export NPU_DEVICE_COUNT TP_SIZE DP_SIZE EP_SIZE SHM_SIZE MEM_FRACTION_STATIC
export CONTEXT_LENGTH MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS
export TOKENIZER_WORKER_NUM WATCHDOG_TIMEOUT
export ENABLE_DP_ATTENTION ENABLE_DP_LM_HEAD ENABLE_DEEPEP DEEPEP_MODE
export ENABLE_DECODE_GRAPH DECODE_GRAPH_MAX_BS DECODE_GRAPH_BATCHES
export UNITED_CONTAINER HEALTH_TIMEOUT ENABLE_SYSTEM_TUNING
export SGLANG_NPU_PROFILING REFUSE_IF_PD_RUNNING
```

---

## 2. Workload 定义

| # | 场景 | 输入 | 输出 | 请求数 | 说明 |
|---|---|---|---|---|---|
| 1 | 8k/1k | 8,192 | 1,024 | **并发 × 2** | 常规短输入中输出 |
| 2 | 32k / 1k 长文本 | 32,768 | 1,024 | **= 并发** | 长输入 |
| 3 | 1k / 32k 长输出 | 1,024 | 32,768 | **= 并发** | 小时级 |
| 4 | 前缀 32K / 输出 300 / 缓存命中 60% | 32,768（含共享前缀） | 300 | **并发 × 2** | 缓存命中率以服务端 `cached_tokens/prompt_tokens` 为准 |

**口径**：吞吐 = **输出 tok/s / NPU**（per-NPU）；TTFT 为客户端感知（**含排队**）；数值口径见各表备注。

---

## 3. 榜单

### 3.1 8k/1k（输入 8,192 / 输出 1,024；请求数 = 并发 × 2）

| Rank | 配置名 | Concurrency | Running Batch | NPU | Parallel | Chunk Size | 其他优化 / 配置开关 | 输出吞吐 (tok/s/NPU) | TTFT (ms) | TPOT (ms) | 相对 Baseline | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline · United NoEP | 32 | 14 | 16 | TP16,DP2,EP1 | 8K | DP Attention/LMHead 开；DeepEP 关 | 5.8179 | 220,960.03 | 50.994 | 0.00% | 本轮 Baseline；当前最佳 |
| — | 优化方案 A | 待填 | 待测 | 待填 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 | 待计算 | 预留；实测后参与排名 |
| — | 优化方案 A+B | 待填 | 待测 | 待填 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 | 待计算 | 预留；实测后参与排名 |

### 3.2 32k / 1k 长文本（输入 32,768 / 输出 1,024；请求数 = 并发）

| Rank | 配置名 | Concurrency | Running Batch | NPU | Parallel | Chunk Size | 其他优化 / 配置开关 | 输出吞吐 (tok/s/NPU) | TTFT (ms) | TPOT (ms) | 相对 Baseline | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline · United NoDp | 2 | 2 | 16 | TP16,DP1,EP16 | 16K | DP Attention/LMHead 关；DeepEP auto | 0.925 | 46,350.00 | 56.32 | 0.00% | 本轮 Baseline；当前最佳 |
| — | 优化方案 A | 待填 | 待测 | 待填 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 | 待计算 | 预留；实测后参与排名 |
| — | 优化方案 A+B | 待填 | 待测 | 待填 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 | 待计算 | 预留；实测后参与排名 |

### 3.3 1k / 32k 长输出（输入 1,024 / 输出 32,768；请求数 = 并发）

（待填）

### 3.4 前缀 32K / 输出 300 / 缓存命中 60%（请求数 = 并发 × 2）

（待填）

---

## 4. 备注与已知问题

1. **`UNITED` = PD 共置**（同一组卡同时做 prefill 与 decode）；本文件所有配置均为单机 16 卡。
2. **DeepEP**：`auto` 会在 decode 走 low-latency 路径；与之相关，**Decode Graph 在该路径下可能无法启用**（镜像行为），报数时要注明 graph 是否真的生效。
3. **TTFT 含排队**：高并发下的 TTFT 会包含长时间排队（如 3.1 的 220s），比较时需与并发数一起看。
4. **缓存命中率**（3.4）以服务端 `meta_info.cached_tokens / prompt_tokens` 为准。
5. 单机 16 卡的 **KV 池容量有限**（实测约 3~4 万 token）→ 长输入（32K+）与长上下文场景要提前确认放得下，必要时考虑 HiCache / KV offload。

