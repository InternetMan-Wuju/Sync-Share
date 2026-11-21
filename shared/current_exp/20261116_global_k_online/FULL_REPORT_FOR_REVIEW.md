# 全组固定 K 冗余槽的 Online Prefill 负载均衡：完整报告

日期：2026-11-22  
分支：`main`  
当前代码提交：`eab986c`（本报告追加后会更新）  
报告目录：`shared/current_exp/20261116_global_k_online/`

## 0. 一句话结论

已经完成“全组固定 K、非每 rank 固定、非 owner 不分配槽”的槽管理和多类真实 NPU probe；但没有完成真实 GLM full-model external-slot dispatch，因此**没有 global-K 的真实 TTFT 提升结论**。

## 1. 环境核实

### 当前可用机器

- `npu1-23`：16 张 Ascend 910C，当前无 NPU 进程、无运行容器。
- `npu1-25`：16 张 Ascend 910C，当前无 NPU 进程、无运行容器。
- 两节点合计 32 张卡，但本轮没有启动 32 卡模型。

### 之前实际使用的单节点环境

- 节点：`npu1-19`
- 设备：16 张 Ascend 910C
- 服务镜像：`quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3`
- probe 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`
- 量化：ModelSlim W8A8

从模型配置核实：

```text
hidden_size = 6144
moe_intermediate_size = 2048
n_routed_experts = 256
num_experts_per_tok = 8
layers = 78
first 3 layers = dense
remaining 75 = sparse MoE
```

之前实际 layer49 状态测量得到每个 expert state 约：

```text
37,810,176 bytes ≈ 36.06 MiB
```

这是槽大小参考，不是本轮重新加载完整模型测得的。

### 现有 Prefill 调用链

源码检查到的链路：

```text
DeepEPMoE.run_moe_core
→ quant_method.apply(layer=layer, dispatch_output=...)
→ ModelSlimMoEScheme.apply
→ AscendQuantInfo(
      w13_weight,
      w2_weight,
      scales/offsets/biases
   )
→ GroupedMatmul.forward
→ torch.ops.npu.npu_grouped_matmul
```

当前 `GroupedMatmul.forward` 仍只传一个 weight tensor：

```python
weight=[weight]
```

这是接真实 external slot 的关键待改接口。

## 2. 实现了什么

### 2.1 全组固定 K 槽池

代码：`scripts/global_slot_pool.py`

语义：

```text
固定 owner_ranks 列表，长度为 K
每个 owner 可以有 0、1 或多个槽
非 owner 不创建任何槽后备存储
K 不要求是 EP 数量的倍数
```

管理接口：

```text
acquire(request_id, layer_id, expert_id)
→ RESERVED
→ caller writes weight/scale/offset
→ mark_ready
→ use
→ release
→ FREE
```

每个租约有 generation，旧租约不能覆盖新槽。

### 2.2 CPU 多动作搜索验证

代码：`scripts/toy_global_search.py`

它不是最终生产成本模型，只验证：

- 0/1/多个动作；
- 整数 token 配额；
- 新副本成本；
- 已有副本免新复制；
- 不因单步暂时变差而漏掉两动作组合。

测试：`tests/test_global_k.py`

5 项通过：

1. K=5、EP=16、owners=[0,2,5,9,13]，K 非倍数；
2. 非 owner 不分配；
3. 跨层复用；
4. 已有副本免复制；
5. `[40,39,1,1]` 反例。

反例：

```text
手动构造：
Action(source=0, dest=2, q=20)
Action(source=1, dest=3, q=19)

[40,39,1,1]
→ [20,20,21,20]
```

单动作在该 toy 中不赚钱；两动作组合可以赚钱。有限 beam width=32 能找到两动作的 max-load=21 方案。

注意：这里的 40/39/1/1 是 toy load，不是实际测量时间。

### 2.3 单设备 real-NPU 槽池

测试：`tests/test_npu_global_k.py`

配置：

```text
K = 3
logical owners = [2,2,5]
slot_bytes = 37,810,176
```

结果：

```text
committed_bytes = 113,430,528
per_rank_capacity = {2: 2, 5: 1}
active_bytes_after_release = 0
cached_free_bytes_after_release = 113,430,528
torch memory_allocated = 113,431,552
torch memory_reserved = 117,440,512
torch max_memory_allocated = 113,431,552
```

要点：释放槽后，pool 的 active bytes 变成 0，但 HBM 仍然占用 `113,430,528` bytes；不能把 free 槽当成已释放显存。

### 2.4 W8A8 外部槽 GMM probe

测试：`tests/w8a8_external_slot_probe.py`

真实 NPU 上使用小尺寸 synthetic W8A8：

```text
x: [6,64] int8
main weights: [2,64,64] int8
external weight: [64,64] int8
main scale: [2,64] bf16
external scale: 从 GlobalSlotPool 的真实 slot 读出
group_list: [2,2,2]
```

baseline：

```text
weight=[main0, main1, external_bank]
```

split：

```text
weight=[main0, main1, external_slot]
```

结果：

```text
status = PASS
output_shape = [6,64]
max_abs_diff = 0.0
external_slot_used = true
```

这证明 external slot 中的 W8A8 权重和 scale 能被 GMM 直接读取。它不是完整 GLM 的 6144/2048 实际层，也不是完整 routed expert dispatch。

### 2.5 真实 16-rank HCCL probe

测试：`tests/multirank_probe.py`

```text
world_size = 16
K = 3
owners = [2,5,7]
source = rank0
payload = 1024 bytes
```

- owner ranks 各分配 1 个槽；
- 其余 13 个 rank 不分配槽；
- rank0 向 2、5、7 发送不同数据；
- owner 接收、安装、mark ready、读取、release；
- PASS。

这证明了跨 rank 复制和 ready 路径，但没有使用 37.8 MB 真实专家权重，也没有接入真实模型。

## 3. 旧 EPLB/TTFT 对照结果

这部分是之前的 node19 实验，不属于本轮 global-K full-model 结果，但可供审阅。

配置：

- 单节点 16 卡；
- 8K 人类可读 ShareGPT 提示词，8176 tokenizer tokens；
- output 32；
- concurrency=1；
- prefix/radix cache off；
- 1 warmup + 3 measured；
- TTFT 定义为请求到第一个非空 streamed content token。

| 方案 | TTFT 中位数 | 说明 |
|---|---:|---|
| off | 2.6022 s | 不启用 EPLB |
| EPLB 0 冗余，未触发重排 | 2.6073 s | EPLB 已启用但 `rebalance` 没有发生 |
| EPLB 16 冗余，未触发重排 | 2.6432 s | 同上，不能当作完整重排结果 |
| ours，8K gate | 2.6211 s | 8K 被门槛判为不复制，走 no-copy |

差异都很小，不应当宣称 TTFT 加速。

强制原版 EPLB 重排的失败：

```text
rebalance start: 2026-11-15 21:23:57
first OOM:       2026-11-15 21:24:01.659
没有 rebalance end
```

- 每卡约 54.3–54.7 GiB 已分配；
- 临时 buffer 需求 194 MiB 或 386 MiB；
- 当时每卡仅剩约 50–200 MiB free；
- 4.7 s 是失败窗口，不是完成的 rebalance wait。

因此旧结果只能证明：

- 普通 EPLB 无重排路径能启动；
- 完整重排在 61 GiB HBM + 该 54+ GiB 模型上会 OOM；
- 不能用旧 6.415 ms/0.5 ms 作为已验证生产门槛。

## 4. 关键接口阻塞

### 4.1 FusedMoE/expert 数量

当前代码仍假设每 rank 本地专家数由统一的全局专家数整除：

```text
global_routed_experts % ep_size == 0
local_routed_experts = global_routed_experts // ep_size
```

因此不能直接用现有的每层 `FusedMoE` 表示“K 个总槽、某些 rank 没有槽”。

### 4.2 DeepEP

低延迟路径还有：

```text
num_experts % group.size() == 0
num_qps_per_rank = num_experts // group.size()
```

需要改成显式每 rank 容量表或最大容量通信 buffer，但**专家权重不能按最大值在所有 rank 分配**。

### 4.3 量化 wrapper

`ModelSlimMoEScheme.apply` 把 layer 的单一 weight bank 传成 `AscendQuantInfo`。需要支持：

```text
main bank + external slot list
```

并把对应的 weight、scale、offset、bias 都按 expert 列表传入。

### 4.4 GroupedMatmul

真实 W8A8 probe 已证明底层 `npu_grouped_matmul` 支持：

```text
weight=[main0, main1, external_slot]
```

但现有 `GroupedMatmul.forward` 仍传单个 3D tensor；需要改 wrapper，不影响已有主路径。

## 5. 本轮明确没有验证的内容

- 真实 GLM 6144/2048 W8A8 的 external slot GMM；
- 真实 router 的 token 任务从主专家分流到 external slot；
- 原路径扣除副本任务、避免双算；
- 外部槽输出的原 top-k 位置回填和 combine；
- 真实 37.8 MB 权重的跨 rank 复制；
- 完整模型 dispatcher / expert map；
- global-K TTFT。

## 6. 证据分类

| 内容 | 证据 |
|---|---|
| K 非倍数、owner-only 分配 | CPU 测试 + NPU 单设备 |
| 跨层复用和 cached-free 显存 | CPU + NPU 单设备 |
| 外部槽 W8A8 GMM | NPU 实测，synthetic 小尺寸 |
| 16-rank HCCL `K=3 owners=[2,5,7]` | NPU 实测，1024 B |
| 多动作搜索/反例 | CPU 模拟 |
| 完整模型 external slot/TTFT | 未验证 |

## 7. 下一步建议

最值得做的一项：

> 在真实 GLM 的一个 MoE 层上，把 `GroupedMatmul.forward` 接成 main weight bank + external slot weight list，先只跑一个 owner rank 和一个 external slot，验证 token 分流、输出回填和 W8A8 数值一致性。

建议使用 23、25 两节点组成一个 32 卡服务，而不是两台独立 16 卡服务：32 卡可以把每卡权重大约降到一半，给 EPLB 重排临时 buffer 和 external slot 留出 HBM。先验证 TP32/EP32 能否正常启动；再跑真实层、再跑 TTFT。

验收顺序：

1. 32 卡服务健康；
2. 单层 external slot 数值一致；
3. 只复制 K=1；
4. K<32 且非 owner 无槽；
5. 多动作和跨层复用；
6. 最后才测 TTFT。

## 8. 复现

CPU：

```bash
python3 tests/test_global_k.py
```

NPU probe：

```bash
# 在 npu1-19 单卡容器中运行：
python3 tests/test_npu_global_k.py
python3 tests/w8a8_external_slot_probe.py
```

16-rank 多卡：

```bash
# 使用 16 个 davinci 设备的容器：
torchrun --standalone --nproc_per_node=16 tests/multirank_probe.py
```

关键日志在 `artifacts/`：

- `cpu_tests.txt`
- `npu_global_k_memory.txt`
- `w8a8_external_slot_probe.txt`
- `multirank_probe.txt`
