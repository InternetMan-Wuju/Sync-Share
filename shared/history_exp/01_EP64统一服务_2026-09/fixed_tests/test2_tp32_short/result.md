# 测试2 结果：TP32 + 短输出（16 token）+ 并发 128

## 配置

- 模型：GLM-5.2 W8A8（modelslim 量化）
- 部署：SGLang，TP=EP=32，无 EPLB，DeepEP low_latency
- 硬件：2 机 × 16 卡（Ascend 910C），共 32 卡
- 数据集：`databricks-dolly-15k.jsonl`
- 并发：128，max_tokens：16，duration：180s

## 原始结果

| 指标 | 值 |
|---|---|
| 完成请求 | 383 / 0 错误 |
| wall | 262.7s |
| tokens/s（总） | 144.8 |
| completion tokens | 6128 |
| completion tokens/s（e2e 口径） | 23.3 |
| e2e P50 / P90 / P99 | 81.8 / 102.9 / 102.9s |

## 服务端延迟分解（均值）

| 组件 | 耗时 |
|---|---|
| TTFT | 86.58s |
| queue | 44.13s |
| prefill forward | 1.168s |
| 残差（tokenizer↔scheduler IPC + 首 token 回传） | ≈41.3s |

## 关键观察

- 16 token 的短输出下，e2e 几乎全被 **TTFT** 吃掉（P50 81.8s），decode 只占很小一部分。
- TTFT 里，queue 44.1s，残差（之前定位的 tokenizer↔scheduler IPC）约 41.3s，两者是大头。
- 这再次印证：**并发越高，那个 tokenizer↔scheduler IPC 残差越严重**（并发 32 时约 18.3s，并发 128 时约 41.3s）。
- 参考测试2（910B2，32 卡）报的是“约 325 decode tokens/s / 10.16 tok/s/卡”，那是 decode 阶段吞吐；我们这里是 e2e（含排队）口径的 completion tokens/s = 23.3，两者口径不同，不能直接比。
