• 先说明：我这次没有在本机执行任何实验或写文件，只做了文件读取和联网检索。下面按“测量什么、怎么测、怎么设计实验”来整理。

  ## 1. 先定义：GLM-5.2 的 EPLB 测的是什么

  EPLB 是 Expert Parallel Load Balancing，用于推理部署时专家并行（EP）场景。它平衡的不是模型训练时的 aux loss，而是：

  > 每个 EP rank 上，各个专家实际处理的 token 数量是否均匀。

  GLM-5.2 本机配置里能看到：

  - n_routed_experts: 256
  - num_experts_per_tok: 8
  - n_shared_experts: 1
  - num_hidden_layers: 78
  - max_position_embeddings: 1048576
  - 架构是 GlmMoeDsaForCausalLM

  所以它是典型的大 MoE，专家数 256、每 token 激活 8 个，EPLB 有意义。

  vLLM 内部的负载单位基本是 token 数：expert_load_pass 的形状是 (num_moe_layers, num_physical_experts)，再维护一个 expert_load_window 滑动窗口。

  ## 2. 怎么测量

  最核心的指标是：

  balancedness = avg_tokens / max_tokens

  其中 avg_tokens 是所有 EP rank 的平均负载，max_tokens 是负载最高 rank 的值。完全均衡时为 1，越小越不均衡。

  vLLM 日志会输出类似：

  EPLB step: 2500: avg_tokens=1024, max_tokens=2048, balancedness=0.5000

  另外还建议看：

  - 每个 MoE 层的峰均比：

  imbalance_layer = max_rank_load / mean_rank_load

  这个值最小为 1，越大说明该层专家越不均衡。

  - 每轮重排前后对比：current 是实际值，update 是算法估计重排后的值。
  - 重排频率和重排耗时。
  - 端到端性能：TTFT、TPOT、ITL、P99、输出 token/s、请求吞吐。

  在 vLLM-Ascend 文档里，还明确说可以从日志中搜索 [Expert Hotness]，它会输出每层不同 rank 的峰均比，并给出 mean 和 max；current 表示当前实际峰均比，update 表示算法调整后的估计峰均比。

  ## 3. 怎么开启测量

  ### 3.1 如果你用 NVIDIA / 上游 vLLM

  必须同时满足：

  --enable-expert-parallel
  --enable-eplb

  可以用新版 EPLB config：

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

  旧版本 vLLM 也可能用这些参数：

  --eplb-window-size
  --eplb-step-interval
  --eplb-log-balancedness
  --num-redundant-experts

  注意：

  - num_redundant_experts 不是随便填的，通常要求 (256 + N) % EP_size == 0。
  - EP_size = tensor_parallel_size * data_parallel_size（对 MoE sharding 来说）。
  - 太频繁重排会伤害性能，上游 vLLM 一般建议先试 window_size=1000、step_interval=3000，不要用 100/100 这种过激值。

  ### 3.2 如果按这台机器用 vLLM-Ascend

  这台机器明显是 Ascend 环境，而且本地有 glm-q，它是 GLM-5.2-w8a8 量化版。Ascend vLLM 的动态 EPLB 需要环境变量保护开关：

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

  Ascend EPLB 常用参数含义：

  - expert_heat_collection_interval：收集专家热度的时间间隔。
  - algorithm_execution_interval：执行重排算法的间隔。
  - eplb_policy_type：
      - 0 Random
      - 1 DefaultEplb
      - 2 SwiftBalanceEplb，Ascend 文档推荐
      - 3 FlashLB，适合高频波动的负载

  - eplb_heat_collection_stage：
      - all：prefill 和 decode 都统计
      - prefill：只优化 prefill，更关心 TTFT
      - decode：只优化 decode，更关心 TPOT

  这里 TP8 × DP2，EP 规模相当于 16。GLM-5.2 有 256 个 routed experts，所以：

  N=0  -> 256/16 = 16
  N=16 -> 272/16 = 17
  N=32 -> 288/16 = 18

  这些都能整除。建议从 N=16 开始测。

  ## 4. 建议的实验设计

  ### 4.1 固定对照组

  至少要有一组 EPLB 关闭的 baseline：

  baseline: --enable-expert-parallel，但不启用 EPLB
  experiment: --enable-expert-parallel + EPLB

  其他条件必须一致：

  - 同一个模型路径
  - 同一个 max_model_len
  - 同一个 max_num_seqs
  - 同一个 max_num_batched_tokens
  - 同一组 prompt / request trace
  - 同一 seed
  - 同一 NPU/GPU 拓扑

  ### 4.2 准备几类负载

  EPLB 很依赖负载模式，不能只测一种流量。

  建议至少三组：

  1. prefill 为主：输入 4k / 8k / 32k tokens，输出较短。
     对 GLM-5.2 这种 1M context 模型很关键，主要看 TTFT。

  2. decode 为主：输入 128 / 512 tokens，输出 2k / 4k tokens，高并发。
     主要看 TPOT / ITL / 吞吐。

  3. 混合流量：尽量接近线上真实比例，例如 60% prefill + 40% decode。

  ### 4.3 可调变量

  可以做一个简单矩阵：

   实验    EPLB          policy          redundant experts    heat stage
  ━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━
   A       off           -               0                    -
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   B       on            SwiftBalance    16                   prefill
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   C       on            SwiftBalance    32                   prefill
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   D       on            FlashLB         16                   prefill
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   E       on            SwiftBalance    16                   decode
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   F       on            SwiftBalance    16                   all
  ──────  ────────────  ──────────────  ───────────────────  ────────────
   G       static map    -               -                    -

  如果是 Ascend，还可以额外做静态 EPLB：

  - 先记录专家映射：

  DYNAMIC_EPLB=true 或 EXPERT_MAP_RECORD=true
  expert_map_record_path=/path/to/eplb.json

  - 下次加载映射：

  expert_map_path=/path/to/eplb.json

  这样可以比较“运行时动态重排”和“离线固化专家映射”的差别。

  ### 4.4 每个实验要采集什么

  负载均衡侧：

  - balancedness
  - avg_tokens / max_tokens
  - 每层 imbalance
  - 重排前后 current_mean / current_max、update_mean / update_max
  - 重排次数、重排耗时、专家权重 D2D 传输耗时

  性能侧：

  - TTFT p50 / p99
  - TPOT mean / p99
  - ITL mean / p99
  - output tokens/s
  - requests/s
  - NPU/GPU 利用率
  - all-to-all 通信耗时，如果 profiling 能拿到

  ## 5. 怎么判断结果

  不是“balancedness 更高就一定好”。判断标准应该是：

  1. balancedness 或每层峰均比明显改善。
  2. 同时 TTFT / TPOT 的 P99 没有因为重排开销变差。
  3. 吞吐不下降，或者下降在可接受范围。
  4. 重排频率没有过高，避免“为了均衡而频繁 stop-the-world”。

  典型结论可能是：

  - 如果负载不均衡且长 prefill 场景明显，开 EPLB 通常对 TTFT/P99 有收益。
  - 如果输入序列很短、并发较低，或者每 die 专家数很少，EPLB 收益可能抵不过开销，Ascend 文档也明确建议先 benchmark。
  - 如果 balancedness 提高但 P99 变差，优先调大 algorithm_execution_interval / step_interval，或减少 num_redundant_experts。

  ## 参考来源

  - vLLM-Ascend: Expert Parallelism Load Balancer (https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/expert_parallelism_load_balancer.html)
  - vLLM: EPLBConfig / ParallelConfig (https://docs.vllm.ai/en/v0.14.1/api/vllm/config/parallel/)
  - vLLM: EPLB metrics RFC (https://github.com/vllm-project/vllm/issues/30696)
  - vLLM-Ascend: GLM-5.2 deployment tutorial (https://github.com/vllm-project/vllm-ascend/blob/8a1b1cec/docs/source/tutorials/models/GLM5.2.md?plain=1)
  - vLLM forum: How to configure EPLB (https://discuss.vllm.ai/t/how-to-configure-the-parameters-of-eplb/1426)
  - vLLM-Ascend PR: Expose experts hotness metrics (https://github.com/vllm-project/vllm-ascend/pull/9536)

  如果你告诉我你实际准备用哪个框架和卡型（vLLM-Ascend 还是 NVIDIA vLLM，A2/A3 或 GPU 数量），我可以把命令和实验矩阵再具体到这台机器上。