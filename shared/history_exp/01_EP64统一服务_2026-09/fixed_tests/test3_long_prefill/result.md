# 测试3 结果：长 prefill 单请求（拼接 dolly 到 ~19K 输入）

## 配置

- 模型：GLM-5.2 W8A8
- 部署：SGLang，TP=EP=32，无 EPLB
- 硬件：2 机 × 16 卡（910C）
- 数据：`databricks-dolly-15k.jsonl` 多条拼接成一段长 prompt
- 单请求，并发 1，max_tokens 1024

## 原始结果

| 指标 | 值 |
|---|---|
| 输入 token | 19,109 |
| 输出 token | 87（模型提前 EOS） |
| e2e | 69.8s |
| finish_reason | stop |

## 服务端延迟分解（本次单请求）

| 组件 | 耗时 |
|---|---|
| TTFT | 67.69s |
| queue | 0.00s |
| prefill forward | 64.81s |
| 残差（tokenizer↔scheduler IPC） | 2.88s |

## 关键观察

- 单请求无排队时，**残差只有 2.88s**，而并发 32 时是 18.3s、并发 128 时是 41.3s。
- 这直接证明：那个大残差**随并发增长**，本质是 tokenizer↔scheduler 的 IPC 在高并发下成了瓶颈，而不是 prefill/EP 通信本身。
- 单请求的 prefill 吞吐 ≈ 19,109 / 64.81 ≈ **295 tokens/s**（prefill 阶段）。
- 输出只有 87 token（模型提前 stop），没到 1024。
