# S1：单专家完整量化状态与副本数值验证

状态：**PASS（单 NPU，layer 49）**
日期：2026-11-03
设备：仅 `davinci0`，容器内可见 NPU 数 = 1

## 做了什么

- 通过 publisher 只读恢复层读取 layer 49 的 e43、e40、e60。
- 每个专家审计 9 个张量：`gate/up/down` 的 `weight`、`weight_scale`、`weight_offset`。
- 用真实量化算子做完整本地链：
  `dynamic per-token quant → grouped gate/up GMM → SwiGLU → dynamic activation quant → down GMM → restore order`。
- 对 1/17/257 行，以及两副本分流比例 0/100、100/0、50/50、90/10 做逐位一致性检查。
- 做 20 次同输入 A/A，检查确定性。
- 做 FP32 反量化参考比较。
- 做 8 专家 top-8 的 init routing、副本任务分流和 finalize routing 合并检查。

## 权重状态

| 项目 | 结果 |
|---|---:|
| 专家数 | 3（e43/e40/e60） |
| 完整张量数 | 27 / 27 |
| 总权重及量化元数据 | 113,491,968 B |
| `weight` dtype | int8 |
| `weight_scale` 源 dtype | float32 |
| kernel 使用 scale dtype | bfloat16 |
| `weight_offset` | 全部严格为 0 |
| 源检查点副本数 | 每个逻辑张量 1 份 |

权重分片：

- e43、e40：`quant_model_weights-00107-of-00181.safetensors`
- e60：`quant_model_weights-00108-of-00181.safetensors`

## 副本存储与分流

NPU 上的独立 `clone()` 测试表明：

- source / replica A / replica B 的 storage pointer 三者不同；
- 修改 replica B 不影响 source 和 replica A；
- 三个副本内容修改前完全相同；
- 恢复后逐字节回到原哈希。

分流结果：

| 检查 | 结果 |
|---|---:|
| 分流用例 | 36 / 36 通过 |
| 输出恢复后逐位一致 | 36 / 36 |
| 中间激活 re-quant 结果/scale 逐位一致 | 全部通过 |
| 动态量化 q/scale 随分组重排变化 | 无变化 |
| 20 次 A/A 的不同哈希数 | 1 |

## 数值参考

固定量化输入下，完整专家链对 FP32 反量化参考：

```text
NRMSE = 0.01248
上限   = 0.02
```

该参考显式执行了输入反量化、逐通道权重反量化、SwiGLU，以及按原 per-token 规则求激活 scale。

## 8 专家 top-8 合并

- token 数：257
- top-k：8
- 总任务：2056
- 唯一任务：2056
- 重复任务：0
- 每个专家任务数：257
- 两副本分支任务总数：2056
- `npu_moe_finalize_routing` 对任务级参考：逐位一致，最大绝对误差 0

## 结论

在已测试 shape、固定量化边界、真实量化张量和两副本分流下，局部专家计算可以做到逐位一致。该结果证明的是**单卡局部机制正确性**，不证明完整模型质量、跨 rank 通信或端到端性能。

## 产物

- `results/S1_EXPERT_STATE.json`
- `results/S1_NUMERICS.json`
- `scripts/s1_expert_state.py`
- `scripts/s1_npu_numerics.py`
- 权重读取审计：`logs/s1_state_*/weight_audit/`（本地原始日志）
