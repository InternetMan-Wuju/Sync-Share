# EPLB 重排与冗余专家填充逻辑详解

> 主要源码路径：
>
> ```text
> sglang/srt/eplb/eplb_manager.py
> sglang/srt/eplb/expert_location.py
> sglang/srt/eplb/expert_location_updater.py
> sglang/srt/eplb/expert_location_dispatch.py
> sglang/srt/eplb/eplb_algorithms/deepseek_vec.py
> sglang/srt/layers/moe/topk.py
> ```

## 0. 方法来源

这套 EPLB 思路来自 DeepSeek 的 Expert-Parallel Load Balancing（EPLB），
最初是为 DeepSeek MoE / 大 EP 专家并行设计的。

SGLang 里的核心算法文件：

```text
sglang/srt/eplb/eplb_algorithms/deepseek_vec.py
```

文件头直接注明：

```text
This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py
```

基本思想是：

```text
记录每个 logical expert 在不同时间步被路由到的 token 数；
把这些 expert 按负载重新分配到 physical expert slot；
如果允许冗余，则把热门 expert 复制到额外的 physical slot；
最后跨 rank 搬运专家权重，使真实权重布局和路由映射一致。
```

SGLang 在 DeepSeek EPLB 基础上做了工程化适配：

- 支持 Prefill / Decode 两种 rebalance 路径；
- 支持 `--ep-num-redundant-experts`；
- 支持 NPU/昇腾环境；
- 与 PD 分离、DeepEP、DP attention 等组件组合使用。

## 1. 总体流程

EPLB 一个完整周期：

```text
1. 记录每个 logical expert 的路由 token 数
2. 达到 rebalance 周期后，停止记录当前窗口
3. 用热度计算新的 physical_to_logical_map
4. 如果 redundant > 0，选择热门 expert 复制到额外 physical slot
5. 跨 rank 搬运专家权重到新的 physical slot
6. 更新全局 ExpertLocationMetadata
7. 后续推理把 logical expert id 映射到 physical expert id
```

## 2. 触发条件

在 `EPLBManager`：

默认：

```text
rebalance_num_iterations = 1000
```

因此每 1000 次 forward pass 触发一次 rebalance。

触发后取出窗口数据：
```text
logical_count
    每个时间步、每层、每个 logical expert 被路由了多少 token

average_utilization_rate_over_window
    这个窗口内，专家在 GPU/EP rank 之间的负载均衡程度
```

`utilization_rate` 是负载均衡度。
它的算法是：
```text
utilization_rate
    = 平均每个 GPU 的 expert 负载
      / 最大那个 GPU 的 expert 负载
```
因此：越接近 1.0 = 越均衡


阈值(可变)：
```text
eplb_min_rebalancing_utilization_threshold 默认 1.0
如果当前均衡度已经大于 阈值，就跳过 rebalance，避免白重排。
```


## 3. logical_count 的含义

形状：

```text
logical_count[T, L, E]
```

- `T`：记录窗口内的 forward step 数。
- `L`：MoE layer 数。
- `E`：logical expert 数，GLM-5.2 为 256。

数值表示：

```text
某一步、某一层、某个 logical expert 被路由到的 token 数。
```

它不是去重 token，而是 expert-token assignment。

## 4. physical expert 怎么分到 rank

先按热度排序，再做“蛇形分配”。

流程：

```text
1. 计算每个 physical slot 的负载 score
2. 对 score 做降序排序
3. 把排序结果 reshape 成：
   [num_local_physical_experts, num_ranks]
4. 对奇数行做反转
5. 再 transpose，把列分配给各个 rank
```

示例：

```text
第一行（最热）：
61, 53, 43, 32, 11, ...
-> rank0, rank1, rank2, rank3, rank4, ...

第二行（次热）：
52, 48, 35, 27, 21
先反转：
21, 27, 35, 48, 52
-> rank0, rank1, rank2, rank3, rank4
```

为什么反转？

```text
排序后的值先按列映射到 rank。
奇数行先反转，再 传 到 rank 列。
这样第一行 rank0 拿最热，rank31拿最冷，第二行 rank31 拿最热...
后续行继续交替，避免某个 rank 长期最忙。
```

## 5. Prefill 与 Decode 的算法分支

`deepseek_vec.py` 中：

```python
if enable_hierarchical:
    return prefill_rebalance_experts(...)
else:
    return decode_rebalance_experts(...)
```

### 5.1 Decode 路径

```text
decode_rebalance_experts
  -> make_redundant_experts_chunkwise
```

简单来说，就是所有Decode节点都是一个大chunk，在全部 EP rank 上做一次整体负载均衡。

### 5.2 Prefill 路径

```text
prefill_rebalance_experts
  -> pack_groups
  -> make_redundant_experts_chunkwise
```

先按物理节点把 expert 分组；
每个节点内部再作为独立 chunk 做 rank 负载均衡。(我们是1P+1P两个独立Prefill服务，不是2P)

## 6. 冗余专家填充算法

核心在 `make_redundant_experts_chunkwise()`。


对每一个额外冗余槽位，计算：

```text
score = tokens_per_expert / logical_count
```

并加一个极小扰动，保证稳定排序：

```text
tokens_per_expert + expert_id * 1e-4
```

然后：

```text
redundancy_indices = 当前 chunk 内 score 最合适的 logical expert
```



## 7. ExpertLocationMetadata 保存两份映射

### 7.1 physical_to_logical_map

```text
[layer, physical_slot] -> logical_expert
```

用于：

- 日志打印。
- 推理后把 logical expert id 映射到 physical expert id。

### 7.2 logical_to_all_physical_map

```text
[layer, logical_expert, replica_index] -> physical_slot
```

用于：

- 找某个 logical expert 的所有副本。
- 动态 dispatch 时把 token 分散到多个副本。

额外还有：

```text
logical_to_all_physical_map_num_valid
```

表示每个 logical expert 实际有几个有效副本。

## 8. 专家权重搬运

`expert_location_updater.py` 负责把专家权重从旧物理位置搬到新物理位置。

对每个 rank 的每个本地 physical slot，判断：

```text
case 1: unchanged
    目标 expert 已经在该 slot，不动

case 2: same-gpu
    目标 expert 在同一张卡的另一个 slot，直接 copy

case 3: free-rider
    复用本 rank 之前的物理 slot，直接 copy

case 4: same-node
    目标 expert 在同节点其他 rank，走 P2P receive

case 5: cross-node
    目标 expert 在跨节点 rank，走 P2P receive
```
qwe
