# 本轮最小修正与验证

## CPU 守恒修正

- `count_rank_load` / `rank_vectors` 改为按**物理 slot index**决定余数归属，不再混合 logical expert 编号和 physical occurrence。
- calibration、holdout 的 old/corrected 两种映射都验证：
  - 每专家计数守恒；
  - 总任务数为 32768。
- 空 adapter：所有 17th extra slot 一律为 0；owner 只使用原 primary slot，但保留与候选相同的 rank1 双专家驻留 buffer/workspace。

## row quota 修正

- 旧逻辑在 `b2` 已含热点时再次加 `n43+n40`，错误地把热点全部迁出。
- 本轮使用独立整数枚举，目标包含未受影响 rank 的 floor。
- 复现执行卡结果：

| split | 最大行数 | e43 迁出 | e40 迁出 | affected loads |
|---|---:|---:|---:|---|
| calibration | 3983 | 3367 | 2790 | 3982 / 3983 / 3983 |
| holdout | 3943 | 3371 | 3015 | 3941 / 3943 / 3943 |

## 实际计时链数值桥

同一小输入上检查了：

- e43/e40 full `weight_offset` 均为 0；
- source scale 为 float32，kernel scale 为 bfloat16；
- timed layout 为 `[1, 6144, 4096]`；
- dense 对 FP32 反量化参考：NRMSE 1.2385% / 1.3189%；
- S1 grouped 对 dense：0 / 4.63e-6；
- 本轮 fused timed chain 对 dense：0 / 4.63e-6；
- e43 9/8 分流对完整 17 行：逐位一致；
- 空分支和单行边界：通过。

## 计时协议

最终数据满足：

- 20 次预热；
- 每块至少 20 次调用；
- 实际每个正式块至少 100 ms；
- 5 个正式块；
- online A/B 交错顺序；
- decode 的 45 个去重工作类型全部满足块长要求；
- 块内调用不作为独立重复，五块范围不称生产 p95。

decode 试跑中曾出现一次槽位向量长度错误和若干低于 100 ms 的块；这些试跑均未进入最终结果。最终版本使用迭代块长校准并复核全部块。

## Decode 输入契约

真实 dump 的 layer49 decode：

- 64 个 step；
- 每个 step 都是 128 tasks；
- 每 step 有 33–44 个非零专家；
- 计数并非全部 16 的倍数。

因此“1 token × TP16 重复”被排除；更符合 16 个有效 token × top8。dump 只保留专家计数，不保留 token top8、router weight 或源 rank，故不能恢复六格所需的 batch1/8 路由。六格使用固定种子人工路由，真实 trace 实验明确标为未完成。
