# S0 映射与路由核对结果

状态：**完成（离线，无 NPU）**

## 1. 关键纠错

旧 E1 映射不是按 rank 交错的布局，而是：

```text
slot 0..255  = logical 0..255
slot 256..271 = 额外槽
```

272 个物理槽 / 16 rank = **17 slots/rank**，因此：

- slot `256` 属于 **rank15**
- slot `257` 也属于 **rank15**

运行时 dump 已确认：

```text
layer49 expert43 slots = [43, 256]
layer49 expert40 slots = [40, 257]
rank(256) = 15
rank(257) = 15
```

所以旧 E1 实际测试的是：

> expert43/40 复制到 rank15，而不是原先声称的 rank0/1。

旧 mapping JSON 与运行态 dump 完全一致，说明旧结果本身可复查，但实验解释必须修正。

## 2. 修正后的映射

新增按 rank 交错的映射：

```text
每个 rank：
  16 个 primary expert
  1 个额外 slot
```

layer49 修正布局：

```text
expert43: slot 45 (rank2 primary) + slot 16 (rank0 replica)
expert40: slot 42 (rank2 primary) + slot 33 (rank1 replica)
```

所有层都覆盖完整 256 个 logical experts，每个 rank 17 个槽。

## 3. Layer49 负载代理

calibration 集：

| 布局 | rank max/mean |
|---|---:|
| 原始 256 槽 | 4.951 |
| 旧 272 映射（副本在 rank15） | 3.061 |
| 修正映射（副本在 rank0/1） | **2.988** |

holdout：

| 布局 | rank max/mean |
|---|---:|
| 原始 256 槽 | 5.044 |
| 旧 272 映射 | 3.046 |
| 修正映射 | 3.082 |

结论：

- 旧 E1 实际也减轻了 layer49 的局部负载，只是落点不是预期 rank。
- 修正布局在 calibration 更好，holdout 略差，但都显著优于原始布局。
- 这仍然只是计数代理，不能预测端到端收益。

## 4. 任务守恒与陷阱测试

两组路由的 prefill 任务数均为：

```text
2,457,600 = 4096 × 75 × 8
```

验证器能捕获：

- 重复一个任务：总计数 +1，失败
- 遗漏一个任务：总计数 −1，失败
- 把一个任务放到错误 expert：总计数不变，但集合不一致，失败

注意：当前只有窗口汇总计数，没有 token 级任务多重集，因此只能声明“计数守恒”，不能声明 token 级无遗漏。

## 5. 产物

- `results/S0_EVIDENCE.json`
- `results/S0_MAPPING_AUDIT.json`
- `results/s0_layer49_expert_loads.csv`：layer49 完整 256 专家 cal/hold 计数
- `mappings_s0_corrected/s0_adapter.json`
- `mappings_s0_corrected/s0_layer49_top2.json`
- `mappings_s0_corrected/s0_multi3_top2.json`
- `scripts/s0_mapping_audit.py`

## 6. 下一步

进入 S1：只使用一张 NPU，抽取 layer49 的 e43/e40 和冷专家，验证完整量化状态及本地复制正确性。旧 E1 结论降级为：

> 旧 E1 的特定、错误落点配置没有显示明显 TTFT 改善；不能据此否定修正落点后的策略，也不能据此关闭 E4。
