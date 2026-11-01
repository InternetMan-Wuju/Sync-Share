我查看了镜像里的 EPLB 代码，逻辑如下。

## 1. 触发 rebalance

核心在：

```text
sglang/srt/eplb/eplb_manager.py
```

- 启动后创建生成器，每 `rebalance_num_iterations` 次 forward 调用一次 `rebalance()`。
- 我们配置是 1000 次迭代触发一次。
- 触发后先从 `ExpertDistributionRecorder` 拿：

```text
logical_count = [T, L, E]
```

表示每个时间步、每层、每个逻辑专家被路由了多少 token。

## 2. 计算新的专家布局

核心在：

```text
sglang/srt/eplb/expert_location.py
```

`ExpertLocationMetadata.init_by_eplb()` 调用：

```text
eplb_algorithms.rebalance_experts(...)
```

它生成：

```text
physical_to_logical_map[layer, physical_slot] = logical_expert
```

以及反向：

```text
logical_to_all_physical_map[layer, logical_expert, replica] = physical_slot
```

关键公式：

```text
num_physical_experts = num_logical_experts + ep_num_redundant_experts
```

我们：

```text
256 logical experts
redundant = 0  -> 256 physical slots
redundant = 32 -> 288 physical slots
```

## 3. 冗余专家是怎么“复制”的

算法在：

```text
sglang/srt/eplb/eplb_algorithms/deepseek_vec.py
```

`make_redundant_experts_chunkwise()` 主要逻辑：

1. 先把每个 logical expert 放到一个 primary physical slot：

```text
logical_to_physical_map[layer, expert, 0] = primary_slot
```

2. 对每个额外冗余槽，计算：

```text
score = tokens_per_expert / logical_count
```

并选择当前“最热、但复制次数还不多”的 expert。

3. 把该 expert 再放进下一个 physical slot：

```text
physical_to_logical_map[layer, next_redundant_slot] = hot_logical_expert
logical_to_physical_map[layer, hot_expert, copy_index] = next_redundant_slot
```

所以冗余并不是固定复制 expert 0..31，而是按热度动态挑专家。

4. 如果每张卡上有多个 physical expert，再做跨 GPU 负载均衡：

```text
把 physical slot 按热度排序，分配到不同 rank
```

## 4. 重排时专家权重怎么搬运

核心在：

```text
sglang/srt/eplb/expert_location_updater.py
```

对每一层、每个本地 physical slot，判断：

- 目标 expert 已经在本位置：不动。
- 目标 expert 在同一张卡的另一个 slot：直接 copy。
- 目标 expert 在同节点另一张卡：P2P receive。
- 目标 expert 在跨节点：P2P receive。

然后：

```text
isend 发送源 expert 权重
irecv 接收目标 expert 权重到临时 buffer
batch_isend_irecv 执行
最后把 buffer 拷回 routed_experts_weights
```

## 5. 运行时怎么选择专家副本

核心在：

```text
sglang/srt/eplb/expert_location_dispatch.py
sglang/srt/layers/moe/topk.py
```

router 输出的是 logical expert id，之后：

```text
topk_ids_logical_to_physical(...)
```

根据 `ep_dispatch_algorithm` 把 logical id 映射到 physical id：

- `static`：按预计算映射。
- `dynamic`：如果同一个 logical expert 有多个 physical 副本，就把 token 分散到不同副本，避免只打一个副本。
- 我们当前参数未显式设置 `--ep-dispatch-algorithm`，所以走默认路径；但 EPLB 的 `physical_to_logical_map` 仍会参与后续 DeepEP 的 per-rank shared-slot 重排。

## 6. 我们两个 EPLB 补丁的作用

补丁 1：

```text
expert_location_updater_patched.py
```

- NPU 权重默认是 FRACTAL_NZ 格式。
- 直接切片 `isend` 会报：

```text
For a tensor of internal format, its storage_offset must be 0
```

- 补丁在发送前把专家权重转成 ND 格式，接收 buffer 也用普通 ND，收完再写回 NZ。

补丁 2：

```text
torch_npu_distributed_c10d_patched.py
```

- NPU 的 `batch_isend_irecv` 原来会走 coalescing manager，逐个发 P2P。
- HCCL 的 send 是阻塞的，导致先发的 isend 卡住后面的 irecv，形成死锁。
- 补丁改成一次原子批量提交所有 send/recv。

## 一句话总结

EPLB 的逻辑是：

```text
记录每个 logical expert 的 token 热度
  -> 算出新的 physical_to_logical_map
  -> 用 P2P 把专家权重搬到新 physical slot
  -> router 输出 logical id 后，再映射到 physical id
```

`redundant=32` 会在 256 个 primary physical expert 之外，再多 32 个副本，副本由 EPLB 按热度动态选择，而不是固定复制前 32 个专家。