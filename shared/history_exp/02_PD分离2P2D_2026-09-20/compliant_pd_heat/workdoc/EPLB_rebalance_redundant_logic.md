# EPLB 重排与冗余专家填充逻辑详解

> 本文基于镜像 `sglang-glm52-pd:fe694986` 内源码只读梳理，不涉及运行实验。
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

```python
while True:
    for _ in range(rebalance_num_iterations):
        yield
    yield from self.rebalance()
```

默认：

```text
rebalance_num_iterations = 1000
```

因此每 1000 次 forward pass 触发一次 rebalance。

触发后取出窗口数据：

```python
dump_record_output = recorder.dump_record(output_mode="object")
logical_count = dump_record_output["logical_count"]
average_utilization_rate_over_window = dump_record_output[
    "average_utilization_rate_over_window"
]
```

这里两个字段含义不同：

```text
logical_count
    每个时间步、每层、每个 logical expert 被路由了多少 token

average_utilization_rate_over_window
    这个窗口内，专家在 GPU/EP rank 之间的负载均衡程度
```

注意：`utilization_rate` 不是硬件 AICore 占用率，而是负载均衡度。

它的算法是：

```text
utilization_rate
    = 平均每个 GPU 的 expert 负载
      / 最大那个 GPU 的 expert 负载
```

因此：

```text
越接近 1.0 = 越均衡
越小于 1.0 = 某个 rank 比其他 rank 忙得多
```

阈值：

```text
eplb_min_rebalancing_utilization_threshold 默认 1.0
```

- 默认是 1.0 时，通常不记录 history，`average_utilization_rate_over_window` 可能是 None。
- 在 `_check_rebalance_needed()` 中，None 会返回 True，也就是正常 rebalance。
- 如果把阈值改成例如 0.9，含义就是：

```text
如果当前均衡度已经大于 0.9，就跳过 rebalance，避免白白重排。
```

所以这段代码的作用是：

```text
先从 recorder 取出窗口热度；
如果系统已经足够均衡，就跳过；
否则继续计算新布局并搬专家。
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

## 4. 新布局计算入口

在 `ExpertLocationMetadata.init_by_eplb()`：

```python
physical_to_logical_map, logical_to_all_physical_map, expert_count = (
    eplb_algorithms.rebalance_experts(
        tokens_per_expert=logical_count,
        num_physical_experts=num_physical_experts,
        num_local_physical_experts=num_physical_experts // ep_size,
        num_groups=num_groups,
        num_nodes=num_nodes,
        algorithm=...,
    )
)
```

其中：

```text
num_logical_experts       = 256
ep_num_redundant_experts  = 0 或 32
num_physical_experts      = 256 + ep_num_redundant_experts
```

因此：

```text
redundant = 0  -> 256 physical experts
redundant = 32 -> 288 physical experts
```

### 4.1 physical expert 怎么分到 rank

新布局不是简单把 expert 按 ID 依次分到 rank，而是先按热度排序，再做“蛇形分配”。

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

对应代码中的关键操作：

```python
indices = score.argsort(-1, descending=True)
indices = indices.view(
    num_layers,
    num_chunks,
    num_local_physical_experts,
    num_local_groups,
)

# 奇数行反转
indices[:, :, 1::2, :] = indices[:, :, 1::2, :].flip(-1)

# 转置后，每个 rank 拿到各自的一批 physical slot
indices = indices.transpose(2, 3)
```

为什么反转？

```text
排序后的值先按列映射到 rank。
奇数行先反转，再 transpose 到 rank 列。
这样第一行 rank0 拿最热，第二行 rank31 拿最热，
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

特点是：

- 把 256 个 expert 按 chunk 分组。
- 在每个 chunk 内独立选择冗余副本。
- 更关注单 rank 内 expert 热度均衡。

### 5.2 Prefill 路径

```text
prefill_rebalance_experts
  -> pack_groups
  -> make_redundant_experts_chunkwise
```

特点是：

- 先把 expert group 按 node 打包，减少跨节点通信。
- 再做 chunkwise 冗余复制。
- 适合 prefill 的大 token 量场景。

## 6. 冗余专家填充算法

核心在 `make_redundant_experts_chunkwise()`。

### 6.1 初始化映射

先生成主映射：

```text
physical_to_logical_map[layer, chunk, slot]
```

每个 logical expert 先放到一个 primary physical slot：

```text
logical_to_physical_map[layer, expert, 0] = primary_physical_slot
```

### 6.2 选择要复制的热门 expert

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

把它填到下一个冗余 physical slot：

```text
physical_to_logical_map[layer, chunk, primary_count + i]
    = selected_hot_expert
```

同时更新：

```text
logical_to_physical_map[layer, selected_hot_expert, copy_index]
    = redundant_physical_slot
```

并把该 expert 的复制计数加一：

```text
logical_count[layer, selected_hot_expert] += 1
```

### 6.3 直观理解

假设某层最热的是 expert 120：

```text
primary slot 120   -> expert 120
redundant slot 256 -> expert 120
```

这样 expert 120 就有两个物理位置。

`logical_to_physical_map` 保存：

```text
expert 120 -> [primary_slot, redundant_slot, ...]
```

后续 dispatch 可以在这几个副本中分配 token。

### 6.4 跨 GPU 均衡

如果每 rank 有多个 physical expert，算法还会：

```text
1. 把 physical slot 按热度排序
2. 使用交替翻转的方式把热门/冷门 expert 分到不同 rank
3. 重新生成 physical_to_logical_map
```

目的是避免某个 rank 集中太多热 expert。

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

## 8. 推理时的 logical -> physical 映射

在 `topk.py`：

```python
topk_ids = topk_ids_logical_to_physical(
    topk_ids,
    expert_location_dispatch_info,
)
```

根据 `ep_dispatch_algorithm`：

### 8.1 static

```text
直接查预计算的 physical map
```

### 8.2 dynamic

```text
如果 logical expert 有多个 physical 副本，
把 token 分散到这些副本，避免只打一个副本
```

### 8.3 fake

```text
测试用，随机化 router logits
```

当前我们未显式设置 `--ep-dispatch-algorithm`，走默认路径；但 DeepEP 后续还会做
per-rank shared-slot 重排。

## 9. 专家权重搬运

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

执行：

```python
P2POp(op=torch.distributed.isend, ...)
P2POp(op=torch.distributed.irecv, ...)
torch.distributed.batch_isend_irecv(p2p_ops)
```

接收时先写入临时 buffer，最后 copy 回：

```text
routed_experts_weights
```

## 10. 我们两个 NPU 补丁的位置

### 10.1 SGLang 侧补丁

```text
expert_location_updater_patched.py
```

作用：

```text
发送前把 FRACTAL_NZ 专家权重转成 ND 格式
接收 buffer 使用普通 ND 连续内存
接收完成后写回 NZ
```

### 10.2 torch_npu 侧补丁

```text
torch_npu_distributed_c10d_patched.py
```

作用：

```text
NPU 的 batch_isend_irecv 改为一次原子批量提交
避免 HCCL send 阻塞导致 isend/irecv 死锁
```

## 11. 容易混淆的几点

1. `logical_count` 的 `E` 是 logical expert，不是 physical expert。
2. 冗余专家不是固定复制 expert 0..31，而是根据热度动态选择。
3. EPLB off + redundant 时，SGLang 会使用 trivial 布局，冗余槽位是简单顺序复制，不是热点选择。
4. `physical_to_logical_map` 不能跨 rebalance 窗口混用；每次 rebalance 前后布局不同。
5. Prefill 和 Decode 是独立 EP 服务，它们的 EP 大小不同，映射也不能直接相加。
