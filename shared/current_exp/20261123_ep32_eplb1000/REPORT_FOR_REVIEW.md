# EP32 在线 EPLB +32：1000 迭代复测

## 结论

基于 `bf94982`，只重测第二组在线 EPLB +32，将其 `--eplb-rebalance-num-iterations` 从 2 改为 **1000**，其余条件和上一轮相同。

- 两节点全部 32 个 die，同一个 EP32 组。
- `--enable-eplb`、`--ep-num-redundant-experts 32` 保留。
- v0.5.20 NPU 兼容补丁在两节点加载。
- observer 和 prefix cache 关闭。
- 并发 1、输出 2 token。
- 校准和正式计时都保持 1000 迭代设置。
- 仅重测 EPLB 组；原版和我方 K=2 直接引用上一批 `bf94982` 的结果，并在表中注明来源。

最终状态：

**`PASS_EPLB_1000`**

## 1. 运行条件

| 项 | 值 |
|---|---|
| 基线 | `bf94982` |
| 节点 | `npu1-23`、`npu1-25` |
| 拓扑 | 32 die，单 EP32 组 |
| 镜像 | `quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3` |
| 模型 | `/data_lib/data/models/glm-q`，ModelSlim W8A8 |
| EPLB 周期 | **1000 forward-pass iterations** |
| 冗余专家 | +32；每 rank 9 个物理位置 |
| chunked prefill / max prefill | 8192 / 32768 |
| context | 65536 |
| 并发 / 输出长度 | 1 / 2 token |
| cache | radix cache disabled |
| observer | disabled |
| NPU 补丁 | 双节点只读挂载，哈希与前一轮一致 |

输入指纹沿用 `bf94982`：

| 输入 | SHA256 |
|---|---|
| 8K | `86240a227b4f194e7f0e3ab5661942b09c0aa5a44cdcc610f503f54d9e3f3d84` |
| 16K | `ef1b6259ebc4a272df50b9e905887909e9acab82924231e72cff73c981b217c9` |
| 32K | `cd222160c6815f5ce8db984894f40d40880333dfed62139a23b4ffa0464fafe1` |

输入为 AgentX-derived 确定性 token 物化；32K 是两个同 trace 16K 片段的确定性拼接，不是原始人类文本。

## 2. 校准：独立流量走够 1000 迭代

每轮先启动 EPLB 服务，再使用独立的 **64-token AgentX-derived 校准输入**反复请求，直到日志中出现新的 `rebalance end`。校准请求数和重排结果：

| round | 校准请求数 | 新 `rebalance end` | 重排耗时 | layout before→after |
|---|---:|---:|---:|---|
| 1 | 496 | 是 | 24.645 s | 变化 |
| 2 | 496 | 是 | 24.390 s | 变化 |
| 3 | 496 | 是 | 24.655 s | 变化 |

每个 round 在 TP0 观察到一个 group-wide rebalance，所有 rank 的 end 日志都存在。校准期间使用的仍是 1000 迭代配置；没有通过临时降低迭代数来制造重排。

重排时间线、before/after 布局哈希和每轮事件记录在：

`artifacts/EPLB1000_SUMMARY.json`

## 3. 正式样本和重排标记

正式计时在每轮校准完成后进行：

- 8K、16K、32K 各先做 1 次 warmup；
- 每档正式请求数为 `4 + 3 + 3 = 10`；
- 三轮共 30 条正式请求、9 条 warmup、1488 条校准请求。

正式 EPLB 样本是否与重排重叠：

| 输入 | 样本数 | 与重排重叠 | 不与重排重叠 |
|---|---:|---:|---:|
| 8K | 10 | 0 | 10 |
| 16K | 10 | 0 | 10 |
| 32K | 10 | 0 | 10 |

因此本轮的正式 EPLB 中位数是**校准重排完成后的正式服务时间**；校准重排单独计入上面的校准记录。正式期间没有再次触发重排，这一点由逐请求时间区间和日志时间线交叉核对。

## 4. 三组 × 三档更新表

EPLB 行为本轮实测；Original 和 K=2 行来自上一批 `bf94982`，没有在本轮重跑。

表中为 TTFT 中位数（括号内为端到端总耗时中位数），每组每档 n=10。

| 输入 | 原版（bf94982） | EPLB +32（本轮，1000） | K=2（bf94982） | EPLB 相对原版 | K=2 相对原版 |
|---|---:|---:|---:|---:|---:|
| 8K | 2.8025 s（3.3236 s） | **2.5969 s（3.1665 s）** | 2.9358 s（3.4774 s） | **-7.34% TTFT / -4.73% total** | +4.72% TTFT |
| 16K | 5.7147 s（6.2401 s） | **5.3072 s（5.8814 s）** | 5.9785 s（6.4972 s） | **-7.09% TTFT / -5.75% total** | +4.53% TTFT |
| 32K | 11.7875 s（12.3165 s） | **11.2183 s（11.7814 s）** | 12.3013 s（12.8586 s） | **-4.85% TTFT / -4.34% total** | +4.42% TTFT |

本轮 EPLB 的差值是在相同输入、相同 4+3+3 轮次/重复编号上与上一批 Original 配对后的中位百分比；Original 和 K=2 的样本来自上一批运行，不是本轮重新启动的服务。

## 5. 正确性

- 本轮 EPLB 三档 10/10 请求都返回，output IDs 与 `bf94982` 原版对应样本完全一致。
- 所有 formal 请求错误数为 0。
- 本轮不重新声明 K=2 的正确性；K=2 上一批 8K 的已知 token 差异仍在，16K/32K output IDs 与原版一致。相关证据保留在 `bf94982` 批次。
- 生产格式仍为 `UNKNOWN_NO_ACTIVE_SERVICE`。

## 6. 交付数据

- `artifacts/EPLB1000_SUMMARY.json`：统计、配对差值、每轮重排时间线和布局哈希
- `artifacts/EPLB1000_RAW_REQUESTS.csv`：正式、warmup、calibration 的逐请求 TTFT、总耗时、输出 ID、重排重叠标记
- `artifacts/THREE_ARM_TABLE.md`：三组表
- `artifacts/raw/formal/`：30 条 formal 原始 JSON
- `artifacts/raw/warmup/`：9 条 warmup 原始 JSON
- `artifacts/raw/calibration/eplb1000_calibration.jsonl`：1488 条校准原始记录
- `run_eplb1000.sh`：复现入口

复现：

```bash
cd shared/current_exp/20261123_ep32_eplb1000
./run_eplb1000.sh all
```
