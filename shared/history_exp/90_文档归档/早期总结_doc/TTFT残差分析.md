# TTFT - queue - prefill 残差分析

## 问题

在 EP64 实验的 e2e 延迟分解图里，有一段很大的“TTFT - queue - prefill”残差（约 18s）解释不清。会议里白老师怀疑是 EP 的 all-to-all 通信。

## 结论（已定位）

**这段残差不是 EP all-to-all 通信，而是“调度器之外”的 tokenizer + IPC 开销。**

用 exp1（EP64 无 EPLB）的指标反推：

| 指标 | 平均值 | 含义 |
|---|---|---|
| TTFT | 34.78s | HTTP 请求到达 → 第一个 token 到 tokenizer |
| queue_time | 15.55s | 调度器等待队列（wait_queue → forward_entry） |
| prefill_forward | 0.97s | 前向计算（**包含 MoE all-to-all**） |
| 残差 | ≈18.3s | TTFT - queue - prefill |

## 依据（SGLang 源码）

### 指标定义

- `sglang:time_to_first_token_seconds`：
  `first_token_time - created_time`，其中 `created_time = obj.received_time`（HTTP 服务收到请求的时间）。
- `sglang:queue_time_seconds`：
  `forward_entry_time - wait_queue_entry_time`，即调度器等待队列时间。
- `sglang:per_stage_req_latency_seconds{stage="prefill_forward"}`：
  `prefill_finished_time - forward_entry_time`，即真正的前向（compute + MoE dispatch/combine all-to-all）。

### 时间线

```text
received_time（HTTP 到达）
   │
   ├── tokenize + IPC ──→  scheduler_recv_time
   │                        │
   │                        ├── request_process（≈0）
   │                        ├── queue（15.55s）
   │                        └── prefill_forward（0.97s，含 all-to-all）
   │                        │
   │                        └── prefill_finished_time
   │
   ├── 首 token IPC + detokenize ──→ first_token_time
```

所以：

```text
TTFT = received_time → first_token_time
残差 = TTFT - queue - prefill_forward
     = (scheduler_recv_time - received_time)          # HTTP→tokenize→IPC
     + (first_token_time - prefill_finished_time)      # 首 token 出调度器→IPC→tokenizer
```

## 含义

1. **MoE 的 all-to-all 通信已经包含在 `prefill_forward` 的 0.97s 里**，不是那个 18s 大头。
2. 18s 大头在 **tokenizer 侧 + IPC**：请求从 HTTP 到调度器、首 token 从调度器回 tokenizer 这两段。
3. 要压这个大头，方向是：增加 `--tokenizer-worker-num`、看 tokenizer manager 是否串行、或看 IPC 通道是否瓶颈，而不是优化 EP 通信。

## 待验证

- 用一次带 profiling / 更细粒度的冒烟测试，把“tokenize + IPC 进调度器”和“首 token IPC + detokenize”两段分别量出来，确认哪段占主导。

## 已验证（2026-09-14 冒烟，EP64 EPLB，32 条 Dolly，并发 32）

| 指标 | 平均值 |
|---|---|
| TTFT | 42.82s |
| queue_time | 20.49s |
| prefill_forward | 2.44s |
| request_process | ≈0 |
| **残差（TTFT − queue − prefill）** | **19.89s（占 TTFT 46.5%）** |

结论再次确认：残差占近一半 TTFT，且 prefill_forward（含 all-to-all）只有 2.44s，所以残差不是 EP 通信，而是 tokenizer + IPC 侧开销。

## 深挖结果（埋点 tokenizer_manager，2026-09-14）

在 tokenizer manager 里埋点，把 TTFT 拆成 tokenize / IPC 发送 / 调度+首token回传：

| 组件 | 平均耗时 | 占 e2e |
|---|---|---|
| tokenize（HTTP→分词完成） | 0.004s | ~0% |
| IPC 发送（ZMQ send） | 0.000s | ~0% |
| queue（调度等待队列） | 21.665s | 40.2% |
| prefill forward（含 all-to-all） | 2.476s | 4.6% |
| **残差（scheduler 收包 + 首 token 回传）** | **18.331s** | **34.0%** |
| decode | 11.42s | 21.2% |
| **合计（服务端 e2e）** | **53.895s** | 100% |

关键：`tokenize` 和 `IPC send` 都只有毫秒级，所以残差不是“tokenizer 分词慢”，而是 **tokenizer ↔ scheduler 之间的收包/回传 IPC**（请求进调度器 + 首 token 回传）。这两个方向加起来约 18.3s，是除 queue 之外最大的开销。
