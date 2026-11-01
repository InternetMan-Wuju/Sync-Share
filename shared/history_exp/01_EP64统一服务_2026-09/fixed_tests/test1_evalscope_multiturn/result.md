# 测试1 结果：多轮对话 + 长上下文 + 前缀缓存（无 MTP）

## 配置

- 模型：GLM-5.2 W8A8（modelslim）
- 部署：SGLang，TP=EP=64，无 EPLB，前缀缓存（radix cache，默认开）
- 硬件：4 机 × 16 卡（910C）
- 负载：16 个对话 × 5 轮，首轮约 16K token，后续每轮新增 2K，每轮输出最多 1024 token
- 数据：`databricks-dolly-15k.jsonl` 拼接成长 prompt

## 结果

| 指标 | 值 |
|---|---|
| 总请求 | 80（16 × 5） |
| wall | 1015.7s（约 17 分钟） |
| e2e P50 / P90 / P99 | 228.9 / 246.3 / 246.3s |
| TTFT 均值 | 137.34s |
| completion tokens/s | 80.7 |
| total tokens/s | 1671.3 |

## 关键结论

- **MTP 在本镜像 + 910C + W8A8 上不支持**：`--speculative-algorithm NEXTN` 报 `Unsupported ModelSlim MoE schemes ... W13='FLOAT', W2='FLOAT'`（MTP 草稿层是 FLOAT，modelslim 不认）。
- 所以本测试**没开 MTP**，只测了“多轮 + 长上下文 + 前缀缓存”。
- 长 prefill 在 910C 上很慢（prefill 约 290~300 tok/s），16K 首轮 × 16 个并发导致 e2e 被 prefill 主导，TTFT 高达 137s。
- 参考测试1（H200 + MTP）报 678→916 tok/s、TPOT 18→13ms，我们不可比：硬件不同、且无 MTP。
