# 全组固定 K 冗余槽的 Online Prefill 负载均衡

## 1. 实际实现了什么

本轮在 `npu1-19` 上完成了最小真实路径和算法逻辑验证。

### 已实现

1. `scripts/global_slot_pool.py`
   - EP 组固定总容量 `K`，由固定 owner rank 持有；
   - `K` 不要求是 EP rank 数量的倍数；
   - 非 owner rank 不创建任何专家槽后备存储；
   - 每个 owner 可以有 0、1 或多个槽；
   - acquire → RESERVED → ready → use → release；
   - generation 防止旧租约覆盖新槽；
   - 统计 committed、active、cached-free 和每 rank 分配量。

2. `scripts/toy_global_search.py`
   - CPU-only 多动作搜索验证；
   - 支持 0/1/多个动作、整数 token 配额、空闲槽和已有副本；
   - 支持受限 beam 搜索，并与小规模穷举参考比较。

3. `tests/test_global_k.py`
   - K=5、EP=16 非倍数 owner 测试；
   - 非 owner 不分配后备存储；
   - 跨层复用；
   - 已有副本免复制；
   - `[40,39,1,1] → [20,20,21,20]` 反例；
   - 单动作不赚钱、两动作组合赚钱的搜索测试。

4. `tests/w8a8_external_slot_probe.py`
   - 真实 Ascend NPU 上的 W8A8 `npu_grouped_matmul`；
   - baseline 使用 `[main0,main1,external]` bank；
   - split 路径使用 `[main0,main1]` + 从 `GlobalSlotPool` 读出的 external slot；
   - 输出 `max_abs_diff=0.0`。

5. `tests/multirank_probe.py`
   - 真实 16-rank HCCL probe；
   - `world_size=16`、`K=3`、owners=`[2,5,7]`；
   - source rank0 向三个 owner 发送不同数据；
   - 非 owner rank 不创建槽；
   - owner 接收、写入槽、mark ready、use 校验、release；
   - PASS。

### 尚未实现

- 没有把 `GlobalSlotPool` 接到实际 SGLang `DeepEPMoE.run_moe_core` 的 full model dispatch。
- 没有完整 GLM W8A8 专家 bank 的真实外部槽计算。
- 没有真实的 `ModelSlimMoEScheme.apply` 多权重参数分发。
- 没有实际 16-rank GLM 权重复制、任务分流、回填和 TTFT。
- CPU 搜索是 toy load-vector，不是最终成本模型。

## 2. 环境核实

从节点19和模型配置核实的静态配置：

```text
host: npu1-19
NPU: Ascend 910
available devices: 16 (0-15)
framework/probe image: quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3
model: /data_lib/data/models/glm-q
quantization: ModelSlim W8A8
hidden_size: 6144
moe_intermediate_size: 2048
routed experts: 256
top-k: 8
layers: 78 total, first 3 dense, 75 sparse
```

本轮 probe 没有加载完整 GLM 权重；完整模型配置从模型 `config.json` 核实。之前的实际 layer49 状态测量给出：

```text
一个 expert state: 37,810,176 bytes (约 36.06 MiB)
```

这作为槽大小参考；不是本轮重新加载完整模型得到的测量。

现有 Prefill 源码调用链已检查：

```text
DeepEPMoE.run_moe_core
→ quant_method.apply(layer=layer, dispatch_output=...)
→ ModelSlimMoEScheme.apply
→ AscendQuantInfo(w13_weight, w2_weight, scales/offsets/biases)
→ GroupedMatmul.forward / npu_grouped_matmul
```

当前 `GroupedMatmul.forward` 仍只传一个 weight tensor；这是下一步模型接入需要修改的接口。

## 3. 全组 K 是否真实成立

### CPU

`K=5, EP=16, owners=[0,2,5,9,13]`：

- allocator 调用 5 次；
- 每个 owner 只分配 1 个槽；
- 非 owner 不分配；
- 总后备存储 `5 × slot_bytes`；
- K 不是 EP 数量的倍数，未使用 dummy rank。

### NPU 单设备

`K=3, logical owners=[2,2,5]`：

```text
slot_bytes = 37,810,176
committed_bytes = 113,430,528
per_rank_capacity = {2: 2, 5: 1}
active_bytes_after_release = 0
cached_free_bytes_after_release = 113,430,528
torch_memory_allocated_after_release = 113,431,552
torch_memory_reserved_after_release = 117,440,512
torch_max_memory_allocated = 113,431,552
```

这证明释放槽后 HBM 仍然保留，不能把 cached-free 当成已释放显存。

### NPU 16-rank HCCL

```text
world_size = 16
K = 3
owners = [2,5,7]
non-owner ranks: no slot
source = rank0
payload = 1024 bytes
```

三个 owner 都成功接收、安装、ready、读取；无冲突。实际 37.8 MB 跨卡复制尚未做。

## 4. W8A8 外部槽是否参与计算

真实 NPU probe：

```text
x: int8 [6,64]
main weights: [2,64,64] int8
external weight: [64,64] int8 from GlobalSlotPool
scale main: [2,64] bf16
external scale from GlobalSlotPool
group_list: [2,2,2]
```

两种路径：

```text
baseline: weight=[main0, main1, external_bank]
split:    weight=[main0, main1, external_slot]
```

结果：

```text
status = PASS
output_shape = [6,64]
max_abs_diff = 0.0
external_slot_used = true
```

限制：这是小尺寸 synthetic W8A8 probe，不是 GLM 的 6144/2048 实际层，也不是完整专家链。它证明了外部槽 W8A8 权重和 scale 可以被 GMM 直接读取。

## 5. 门槛和筛选最终采用什么规则

本轮**没有把门槛公式当成已证明结论**，也没有把它接入真实模型。

CPU toy 搜索使用的只是验证旋钮：

- 允许 0/1/多个动作；
- 每个动作有整数 token 配额；
- 新副本计 `copy_cost=2`（toy 单位）；
- 已有副本不计新复制成本；
- beam width 32；
- 目标按 `max(load)+copy_cost` 排序，并保留非即时盈利的部分状态；
- `[40,39,1,1]` 测试中，单动作不赚钱，但两个动作可以到 `[20,20,21,20]`，beam 搜索找到了两动作方案。

这些不是生产门槛/筛选参数。真实 W8A8、真实层成本、通信和等待尚未标定。

## 6. 同样输入到底快了还是慢了

**本轮没有完整的真实模型全局 K 外部槽 TTFT 对照。**

因此不能回答“同样输入下 global-K online 比原版快还是慢”。

本轮可以确认的是：

- 槽管理、跨层复用、W8A8 外部槽 GMM、16-rank HCCL 复制和 ready 都有真实 NPU 证据；
- 真实模型 dispatch、主/副本 token 分流和 TTFT 未接入；
- 不能把 `max_abs_diff=0.0` 或 CPU toy 搜索当成服务加速。

## 7. 证据分类

| 证据 | 类型 |
|---|---|
| 5 项 CPU K/配额/反例测试 | CPU 实测 |
| K=3 owner 槽生命周期 | NPU 实测 |
| 37.8MB 槽 allocated/reserved/peak | NPU 实测 |
| W8A8 外部槽 GMM diff=0.0 | NPU 实测（synthetic） |
| 16-rank HCCL 1024-byte 复制/ready | NPU 实测（小数据） |
| `[40,39,1,1]` 搜索 | CPU 模拟 |
| 真实 GLM 外部槽计算/TTFT | 未验证 |

## 8. 下一步最值得做的一项实验

先把 `GroupedMatmul.forward` 和 `ModelSlimMoEScheme.apply` 改成支持：

```text
main weight bank + external pool slot weight list
```

然后在**一个真实 GLM MoE 层、一个 owner rank、一个 external slot**上跑：

```text
原路径任务数
vs
主专家 + 外部槽分流后的任务数
```

要求：

- 原路径扣掉副本任务；
- external slot 只由 owner rank 持有；
- 输出按原 token 位置回填；
- 完整 W8A8 scale/offset/bias；
- 与原路径数值一致；
- 最后才测 TTFT 和总时间。

## Reproduction

CPU:

```bash
python3 tests/test_global_k.py
```

NPU probes: see `scripts/` and `tests/`, commands are included in the artifact headers.
