# 4 机 PD 共置 + dp-attention：拉起与语义验证（2026-10-11）

## 结论

**4 机 EP64 共置 + `--enable-dp-attention`，服务正常，语义正确（"说人话"）。**

这**推翻了** `bench3/runs/INDEX.md` 里 `tp16dp4_0918` 那条
"语义闪发损坏 ~1/4 请求（dp-attention 路径）"的结论 ——
那个记录与 `sem-fix-0918`（23-qq 162 个坏张量 / 25-qq 5 个坏）是同一天，
**极可能就是当时的坏权重造成的，不是 dp-attention 的问题。**

## 配置

| 项 | 值 |
|---|---|
| 节点/rank | 23(r0, master) / 19(r1) / 31(r2) / 32(r3) |
| 镜像 | `quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3` |
| 模型 | `/data_lib/data/models/glm-q`（四台机字节一致） |
| 并行 | **`--tp-size 64 --dp-size 4 --ep-size 64 --enable-dp-attention`**<br>（= 4 个 DP 组 × 每组 TP16，即吴智承口中的 "tp16dp4"） |
| 其他 | `--moe-a2a-backend deepep --deepep-mode auto`、`MEM=0.65`、`CPS=65536`（开 dp-attn 后源码 //dp_size → 有效 16384）、`MRR=32`、`--cuda-graph-bs "1 2 4 8 16 32"`、`ctx=131072` |
| 环境变量 | `HCCL_BUFFSIZE=1000`、`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64`、`DEEP_NORMAL_MODE_USE_INT8_QUANT=1`、`DEEPEP_NORMAL_LONG_SEQ_*`、`SGLANG_NPU_USE_MULTI_STREAM=1`、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、`STREAMS_PER_DEVICE=32`、`SGLANG_SET_CPU_AFFINITY=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| MTP | **关**（变量隔离） |
| 补丁 | DPFIX(`deepep_patched`) + EPLB 双补丁 + 加载超时补丁 |

## 关键坑（记下来）

1. **`tp_size % dp_size == 0`**（源码 `server_args.py`）；`ep_size ≤ tp_size`。
   所以 64 卡开 dp-attn 是 `tp64 + dp4`，**不是** `tp16 + dp4`。
2. **开 dp-attn 后源码会把 `--chunked-prefill-size` 除以 `dp_size`**，要传 65536 才有有效 16384。
3. dp-attn 下每个 rank 复制完整注意力权重 → 显存更紧；bench3 的 `MEM=0.65` 够用。

## 语义验证（原始输出见 results/）

| 问题 | 输出 | 判定 |
|---|---|---|
| 1+1等于几？ | "1 + 1 = 2" | ✅ |
| 中国的首都是哪座城市？ | "中国的首都是北京" | ✅ |
| 一个星期有多少天？ | "一个星期有7天" | ✅ |
| 你好 / 自我介绍 | 语义正常 | ✅ |

> 输出是 reasoning 风格（把思考过程也吐出来了）；要简洁答复可在请求里传
> `chat_template_kwargs: {enable_thinking: false}`。

## 复现入口

```bash
bash serve4_dp.sh start <run_id>   # 起服务（rank3→2→1→0）
bash serve4_dp.sh status
bash serve4_dp.sh stop
```

---

# 追加：EPLB 开启版（2026-10-11 06:0x）

## 结论：**EPLB 开也完全正常，真实重排 32 次全部成功，重排后语义正确。**

| 检查 | 结果 |
|---|---|
| 启动 | health=200，零报错 |
| EPLB 加载 | 每个 rank：`[EPLBManager] system started, will rebalance per 1000 iterations.` |
| 触发重排 | 灌 64 条请求（并发 8、每条约 256 输出）后触发 |
| **重排执行** | **`rebalance start` = `rebalance end` = 32 次，每次 ~1.53 s，无崩溃/无卡死** |
| 重排期间请求 | 64/64 成功、0 错误 |
| **重排后语义** | 1+1→2 ✅、首都→北京 ✅、一星期→7天 ✅、你好 ✅、水→H₂O ✅ |

## 参数

```
--enable-eplb --ep-num-redundant-experts 0 \
--eplb-rebalance-num-iterations 1000 \
--expert-distribution-recorder-mode stat \
--expert-distribution-recorder-buffer-size 128
```
其余与不开 EPLB 版完全相同（tp64/dp4/ep64 + dp-attention + v0.5.18 + deepep auto）。
额外挂载：`patches/expert_distribution_eplbauto.py`（bench3 的 3 行补丁，解锁 deepep-auto × recorder）。

## ⚠️ 两个必须知道的约束

1. **`eplb_rebalance_num_iterations` 必须 > `expert_distribution_recorder_buffer_size`**
   （源码 `eplb_manager.py:62` 的断言）。我第一次用 100/128 就是因此启动即挂。
   → 想让重排更容易触发，要**同时**把 buffer 调小（例如 200/128 或 32/16）。
2. **共置服务只有一个 scheduler 配置**：prefill 和 decode 共用同一组
   `rebalance-iterations` / `recorder-buffer-size`。**分设只在 PD 分离时成立。**

## 启动方式

```bash
EPLB=1 REBAL=1000 RECD_BUF=128 bash serve4_dp.sh start <run_id>    # 开 EPLB
EPLB=0 bash serve4_dp.sh start <run_id>                            # 不开
bash serve4_dp.sh stop
```
