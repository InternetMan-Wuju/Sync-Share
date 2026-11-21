# Global-K Real Layer Execution Report

## 1. 实际实现了什么

### A. 槽生命周期修正

在 `scripts/global_slot_pool.py` 中补齐：

- `GlobalLease` 带全局池身份，外来池租约不能被本池接受；
- `abort()` 失败回收路径；
- `mark_ready()` / `release()` 可选同步回调，真实 NPU probe 传入 `torch.npu.synchronize`，设备写入完成后才发布 ready 或归还；
- CPU identity test 通过：外来池租约拒绝、abort 后 active=0、ready/release 各触发一次同步。

本地 CPU 测试：

```text
tests/test_global_k.py       5 PASS
tests/test_expert_search.py  PASS
tests/test_slot_identity.py  PASS
```

### B. 专家级 CPU 搜索原型

`scripts/expert_search.py` 现在按 `[rank][logical expert]` 计数，而不是按卡总任务量：

- 动作显式包含 source rank、logical expert、target slot、q；
- slot 包含 owner、layer、expert、ready、busy；
- 错误 expert/layer 的槽必须重新复制；
- ready 且身份匹配的槽免复制；
- busy 槽不能使用；
- 支持多动作，目标是降低 rank 的最大任务/激活开销；
- 保留 `[40,39,1,1]` 的双动作反例，搜索输出：

```text
(20,20,21,20)
Action(src=0,expert=0,slot=0,q=20,copy=True)
Action(src=1,expert=0,slot=1,q=19,copy=True)
```

这是 CPU toy 搜索，不是生产成本模型，也没有验证真实学习到的计算时间。

## 2. 真实单卡完整专家链

### 实际模型状态

从 `synced` 模型文件读取：

- 路径：`/data_lib/data/models/glm-q`
- layer49
- expert43
- 实际 gate/up/down 权重、scale、offset

实际槽大小：

```text
37,810,176 bytes
```

字段：

```text
w13 weight + w2 weight + w13/w2 scale + w13/w2 offset
```

所有字段均被复制到 K=1 外部槽并逐字段比较，`state_bytes_equal_all` 全为 true。

### 计算链

使用真实 W8A8 参数运行：

```text
dynamic int8 quant
→ w13 GMM
→ npu_swiglu
→ dynamic requant
→ w2 GMM
```

输入是 synthetic bfloat16 `[6,6144]`；这不是实际捕获 hidden，不能称为完整服务输入。

结果：

```text
external_slot_vs_bank max_abs = 0.0
manual_split_3+3_vs_bank max_abs = 0.0
```

单卡计时（10 次正式重复，单位 ms）：

| 模式 | median | min | max |
|---|---:|---:|---:|
| bank 原路径 | 0.355 | 0.323 | 0.421 |
| external slot 读取 | 0.344 | 0.325 | 0.423 |
| 3+3 token 分流并回填 | 0.773 | 0.637 | 1.198 |

这里的“分流”是手工 token 位置 `[0,2,4]` 走原专家、`[1,3,5]` 走 external slot；它验证了位置回填和数值一致，不是真实 router 的分流。

## 3. 两个真实层复用同一个 K=1 槽

同一进程、同一个 K=1 槽：

1. 装载 layer49 expert43，完整链输出与 bank 一致；
2. 释放槽；
3. 重新装载 layer50 expert43；
4. 同一个槽再次计算，layer50 输出与 bank 一致。

结果：

```text
layer49 external vs bank max_abs = 0.0
layer50 external vs bank max_abs = 0.0
layer49 all state bytes equal = true
layer50 all state bytes equal = true
committed_bytes = 37,810,176
```

这证明同一槽能跨两个真实层复用，且池的 committed 空间不随层数增加；但两次层的 source bank 本身仍在内存中，不能据此声称完整模型省掉了每层备用槽。

显存采样（单卡）：

```text
torch memory allocated = 113,958,400 bytes
torch memory reserved  = 148,897,792 bytes
max memory allocated   = 133,033,472 bytes
```

这些是 probe 进程的总占用，不等于完整 GLM 模型占用。

## 4. 真实两卡 K=1 复制与分流

在两台同节点 NPU（node23 的 device0/device1）上：

```text
source rank0 → destination rank1
K = 1
state payload = 37,810,176 bytes
destination only has the slot; source has no external slot
```

source 从实际 model shard 读取 layer49 expert43 状态；destination 通过 HCCL `recv` 接收完整状态，写入 `GlobalSlotPool`，mark ready 后直接读取槽权重计算。

数值：

```text
cold_diff = 0.0
```

手工 token 分流：

```text
source: token positions [0,2,4]
destination: token positions [1,3,5]
```

回填后的输出与 source bank 全路径一致。

### 两卡时间

去除首次 JIT/懒初始化异常值，10 次原始记录的稳态中位数：

| 项目 | source | destination |
|---|---:|---:|
| cold copy + ready + split compute | 1.145 ms | 1.065 ms |
| existing replica split compute | 0.361 ms | 0.452 ms |

原始首次值包含约 617 ms source / 209 ms destination 的懒初始化异常，已在原始记录中保留，不能当作稳态成本。正式每组 10 次原始值见：
`artifacts/two_rank_timing.txt` 和 `artifacts/two_rank_timing.json`。

这不是完整层 TTFT；只说明 37.8MB 权重的真实两卡复制和 expert-level 分流可运行。

## 5. 门槛/筛选规则

本轮没有把旧 6.415ms、0.5ms 或旧最慢卡公式当作已证明规则。

CPU expert-aware 搜索目前使用：

- 逐 rank、逐 expert 任务计数；
- slot owner/layer/expert/ready/busy 身份；
- 错误身份必须重新准备；
- 正确 ready 副本免复制；
- 多动作联合评分；
- busy 槽拒绝；
- 零配额动作不执行。

仍需后续用真实计算/复制计时替换 toy 分数，并标定 gate。

## 6. 同样输入快了还是慢了

**没有完整服务/完整层 TTFT 结论。**

本轮可说的局部结果：

- 单卡 bank 与 external 的专家链时间相近（约 0.35ms）；
- 单卡手工 3+3 分流约 0.77ms，因为执行了两条专家链；
- 两卡冷复制约 1.1ms，已有副本 expert-level 分流约 0.36/0.45ms；
- 这些均不包含真实 attention、gate、完整层 dispatch/combine、服务调度和多请求。

因此不能声称 online global-K 在任何完整服务场景获得了加速。

## 7. 证据分类

| 证据 | 类型 |
|---|---|
| layer49/50 实际权重、scale、offset | NPU 实测，真实模型 shard |
| 单卡完整专家链 bank vs external 0.0 | NPU 实测，synthetic hidden |
| 单卡跨层同槽复用 | NPU 实测 |
| 两卡 37.8MB 复制与 ready | NPU 实测 |
| 两卡 token 位置分流/回填 | NPU 实测 |
| `[40,39,1,1]` 多动作 | CPU 模拟 |
| 真实 router 分流、完整层、TTFT | 未验证 |

## 8. 未完成/阻塞

- 还没有把外部槽接到 SGLang 真实 `DeepEPMoE.run_moe_core`；
- 还没有替换 `ModelSlimMoEScheme.apply` / `GroupedMatmul.forward` 的单 weight 调用；
- 还没有真实 router 的 token-expert 选择和完整 combine；
- 还没有完整 75 层或完整服务 TTFT；
- 还没有 K=2 多动作真实 NPU；
- 32 卡完整服务未启动。

## 9. 下一步最值得做

在真实单层上改 `GroupedMatmul.forward` 接受：

```text
main weight bank + external slot weight list
```

并将主专家任务与副本任务按 router 逻辑位置切分。先在一个 layer、一个 owner rank、K=1 上做手工配额，再把 gate/filter 接上。

## 10. 复现

单卡真实 expert 链：

```bash
# node23, one davinci device, model mounted at /models/glm-q
PYTHONPATH=/runexp/scripts python3 /runexp/scripts/real_expert_chain_probe.py
```

两卡真实复制/分流：

```bash
torchrun --standalone --nproc_per_node=2 \
  /runexp/scripts/two_rank_real_probe.py
```

两卡稳态计时：

```bash
torchrun --standalone --nproc_per_node=2 \
  /runexp/scripts/two_rank_real_timing.py
```

CPU：

```bash
python3 tests/test_global_k.py
python3 tests/test_expert_search.py
python3 tests/test_slot_identity.py
```

## 11. 文件索引

- `scripts/global_slot_pool.py`：全局 K 槽管理
- `scripts/expert_search.py`：专家级 CPU 多动作搜索
- `scripts/real_expert_chain_probe.py`：真实 layer49/50 expert43 W8A8 链
- `scripts/two_rank_real_probe.py`：两卡真实复制/回填
- `scripts/two_rank_real_timing.py`：两卡稳态计时
- `artifacts/real_chain_result.json`
- `artifacts/two_rank_result.json`
- `artifacts/two_rank_timing.json`
- `artifacts/cpu_*`
