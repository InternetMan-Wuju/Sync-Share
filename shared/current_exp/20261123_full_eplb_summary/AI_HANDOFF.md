# 大 EP MoE 专家负载均衡：服务器 AI 交接

**当前基线**：`d0b351d`（本总结整理时 `main`）
**范围**：GLM-5.2 / Ascend 910C / ModelSlim W8A8，Prefill 为主，Decode 有模拟和局部实验
**文件性质**：研究脉络与证据索引，不包含插件打包、新代码或新实验
**结论索引**：本文件第 1 节；三组性能在第 4 节；故障排障在第 5 节；复现路径在第 6 节。

## 0. 先读结论

1. 大 EP 的 router 确实有热点专家；热点在单层只能落在一个 rank 时，单层完成时间由最忙 rank 决定。消除“单层热点”通常需要复制/分流，单纯改变专家摆放不够。
2. 原生 EPLB 在历史 EP64 服务中曾出现反效果：逐层独立装箱把每层热点专家对齐到 rank0，rank max/mean 从约 `1.33` 上升到 `5.63～5.69`，服务吞吐下降约 `4%～7%`。这是改变研究方向的关键负结果。
3. 复制/分流有收益上限，但冷复制、状态准备、跨卡通信和同步很容易吃掉收益。两卡完整子链实验中，`n={512,4096,8192}`、三种配额下临时冷复制均没有稳定回本；已有副本也只有少量组为正。
4. 当前方案不是“每 rank 固定容量”，而是**整个 EP 组共享 K 个真实备用槽，跨层复用**。K 由配置指定，K 不要求是 EP 数据面 rank 数的倍数；非 owner rank 不保留每 rank 备用权重。
5. `bd80f52` 已在两机 32 die、一个 EP32 组上验证：K=2 的 layer49/50 使用同一次 DeepEP dispatch/combine，原专家少算、副本接手、输出合并均有任务账本。32K 挂起已修复；EPLB +32 在线重排也使用 v0.5.20 NPU 补丁完成。
6. `bf94982` 是重复测速，但 EPLB 每 **2 次调度迭代**重排，长请求频繁碰到重排；它适合作为“重排频率过高”的历史对照，不应作为最终 EPLB 性能数字。
7. `d0b351d` 将 EPLB 周期改为 **1000 次调度迭代**：每轮独立校准流量完成重排，正式 8K/16K/32K 期间没有再触发重排。当前主性能表采用 `d0b351d` 的 EPLB +32，原版和 K=2 引用 `bf94982`。
8. 当前条件下的人话结论：**调好布局后的 EPLB +32 最快，原版其次，我方 K=2 的 TTFT 比原版慢约 4%～5%**。这是该服务配置、三档输入和本轮样本的结论，不是生产 p95。

## 1. 问题：热点、忙闲 rank 与单层最慢卡

### 1.1 判定方式

对一个 MoE 层，把每个 rank 本地收到的 token-expert 任务（或本地专家计算时间）求和，层完成时间近似由 `max_r(rank_task/time)` 决定。SGLang/DeepEP 中，dispatch、本地 grouped expert 计算和 combine 完成前，请求不能进入下一层；因此“最慢 rank”是层的 critical path。这里的层级规则是分析模型，不表示每层都被单独 profiler 过。

判断要看三个量：

- logical expert 热度是否集中；
- 按实际 `physical_to_logical_map` 折算后，每 rank 的任务/计算是否集中；
- 最忙 rank 与平均 rank 的比值，以及最忙 rank 是否在多个层重复落在同一设备。

### 1.2 历史热力证据

| 发现 | 环境/输入 | 证据 |
|---|---|---|
| EP64 decode 中 logical expert 极不均衡：global CV≈3.59，peak/mean≈58.6；按 rank 聚合仍 CV≈0.846，peak/mean≈5.71 | 4 机共 64 die，decode，GLM-5.2，2058 步 ×78 层 ×256 专家 | `shared/history_exp/01_EP64统一服务_2026-09/EPLB_实验记录.md` |
| PD 2P2D 必须按每个窗口自己的 physical map 统计；EPLB 后 physical rank 的 CV 约 0.105–0.173，低于 logical CV 0.198–0.267 | Prefill EP16×2 + Decode EP32，16K/1K，并发 10/20/30 | `shared/history_exp/02_PD分离2P2D_2026-09-20/README.md` |
| 无冗余时，单层热点专家只能由一个 rank 承担；该层理论最大 rank 负载至少约为平均的 6.9 倍 | EP64，256 逻辑专家，每 rank 4 槽，真实窗口热力 | `shared/history_exp/07_PD共置与EPLB负优化_2026-10-11/eplb_algo_sim/PROBLEM_EPLB负优化.md` |
| 原生 EPLB 开起来后，EP64 的 rank max/mean 约 `1.33×` 变为 `5.63～5.69×`；约 91% 非空层热点落 rank0 | EP64，PD 共置，agentX/random，600s 级服务 | `shared/history_exp/07_PD共置与EPLB负优化_2026-10-11/Q4Exp/Q4Exp_总结报告.md` |
| 同一份 8K 真实服务在 layer49 上 expert163 有 1,671/65,536 个 top8 任务（均匀约 0.39%，约 6.4 倍） | EP16，真实 human-only 8K，Prefill，layer49 observer | `shared/current_exp/20261123_prefill_eplb_three_way/REPORT_FOR_REVIEW.md` |
| Prefill 由 DSA attention 主导（约占 67%）；Decode 的 MoE low-latency dispatch/combine 约占 15.5% | 23 节点 16 卡 profile | `shared/current_exp/README.md` 与 `20261102_load_balance_e0` profile 结果 |

### 1.3 Prefill 和 Decode 的区别

**Prefill**：每个 chunk 的 token 数大，hot expert 可以积累很多任务。复制后可能减少最忙 rank 的 token-expert 工作量，但复制 37.8 MB 专家状态、做 dispatch/combine、等待 ready 都有成本。`20261109` 的 64K chunk replay 在 routed-expert 子链上看到约 19% 局部净收益；8K/16K/32K 经常不值得复制。该结果不是完整 TTFT，也不是 75 层服务。

**Decode**：batch1 通常只有很少的 token-expert 任务，固定 grouped-kernel 和通信启动成本占比高；EP 扩大后计算空间有限。`20261103` 的六格实验和 `20261103_complex_moe_sim` 表明：batch1 的 EP32/64 没有稳定正收益；人工 batch8 的 EP64 约有 15.6% 计算空间，但新增通信/调度预算仅约 0.041 ms/层。真实 Decode trace 缺 token top8、router weight 和源 rank，六格不能冒充完整 Decode 服务结论。

**判定边界**：历史 EP64 和 EP16 热力是真实采集；EP16/32 的 layer 子链是 NPU 实测；EP64 多层和 EP32/64 batch8 的主要系统性扫描是 CPU 模拟/模型外推。任何“单层省 X ms”都不能直接乘 78 层或等价成 TTFT。

## 2. 研究脉络与负结果

### 2.1 热度采集和原生 EPLB（2026-09）

- `01_EP64统一服务_2026-09`：先发现 EP64 expert/rank 不均衡，再尝试原生 EPLB。遇到两个 NPU 故障：
  - `storage_offset must be 0`：EPLB 从 FRACTAL_NZ 权重中切专家切片直接 `isend`，HCCL 拒绝内部格式；
  - `rebalance start` 后无 `end`：torch_npu P2P 的 coalescing 路径逐个提交阻塞 send/recv，造成死锁。
  历史解决是 `expert_location_updater_patched.py` 在发送前转 ND、接收用 ND buffer、收完写回 NZ，以及 `torch_npu_distributed_c10d_patched.py` 使用原子 `batch_isend_irecv`。证据：`shared/history_exp/01_EP64统一服务_2026-09/sglang_npu_eplb_patches/README.md`。
- `05_EPLB冗余专家_1P1D_2026-09-28`：A（关闭）、B（EPLB 无冗余）、C（EPLB +16）对比未完整交付；A 有有效 600s，B 被跳过，C 启动/OOM；recorder buffer 和 lazy-send 也出现过显存问题。这个阶段的教训是：真实 NPU 重排和统计本身就可能改变显存边界。
- `07_PD共置与EPLB负优化_2026-10-11`：完成了关键反例。stock EPLB 的“每层独立装箱”会把热点对齐到同一 rank；EP64 服务吞吐比关闭 EPLB 下降约 4%～7%，重排日志总耗时约 50s（不是精确服务停顿）。因此不能把“开了 EPLB”当作默认正确方向。

### 2.2 复制与分流：先做局部收益，再发现门槛

- `20261103_complex_moe_sim`：真实 dump 的 75 层、EP16/32/64 CPU 模拟。EP16→EP64 的 MoE-only prefill 计算代理从 `48.67` 降到 `27.20 ms/step`，decode 从 `19.99` 降到 `17.67 ms/step`；但这是计算上限，不是服务结果。prefill K=1 在零 dispatch 成本下的冷复制回本线约 `53.5 µs/专家`，decode 只有约 `1.5–4.3 µs/专家`，说明 Decode 的复制窗口极窄。路径：`shared/current_exp/20261103_complex_moe_sim/RESULTS.md`。
- `20261109_decision`：8K/16K/32K/64K 的 Prefill chunk replay。64K 的 routed-expert 子链有约 19% 局部净收益，8K/16K/32K 决策多是不复制；32K 原生长序列仍不可直接作为单次 dispatch 结果。路径：`shared/current_exp/20261109_decision/20261109_074727/RESULTS.md`。
- `20261116_compare_eplb`：EP16 单机 n=16 服务，off/EPLB/EPLB+16/ours no-copy 的 8K TTFT 均在约 2.6s，差异接近波动；强制重排因临时转换 OOM 没完成。不能拿该轮宣布局部或服务加速。路径：`shared/current_exp/20261116_compare_eplb/RESULTS.md`。
- `20261122_global_k_breakeven`：两卡真实 expert43、完整专家子链，`n={512,4096,8192}`、`q={n/4,n/2,3n/4}`；临时冷复制全部没有形成稳定正收益，已有副本也最多只有 5/12 组为正。结论是：没有稳定的生产回本点，不能只靠“复制一个热专家”做结论。路径：`shared/current_exp/20261122_global_k_breakeven/REPORT_FOR_REVIEW.md`。

### 2.3 从每-rank 槽到全组共享 K

- `20261116_global_k_online`：实现 `GlobalSlotPool`，K 是整个 EP 组的固定总容量；owner 可以不为每个 rank；支持 acquire/reserve/ready/use/release、generation、跨层复用。NPU 小尺寸 W8A8 probe 的 bank/external slot 输出 `max_abs_diff=0.0`，75 层只是 2 个槽、总 committed 约 75.6 MB，证明了跨层复用和外部槽读取，但还不是完整服务。路径：`shared/current_exp/20261116_global_k_online/REPORT_FOR_REVIEW.md` 和 `20261116_slot_pool/RESULTS.md`。
- `20261122_global_k_real_layer` / `rework` / `placeholder_proof` / `unified_adapter`：逐步把外部槽接到真实 GLM W8A8、两卡复制、任务分流和 owner 状态；验证了 37,810,176 bytes 状态、layer49→50→49 复用、真实 HCCL 复制、输入保真、owner 状态校验和可靠验收。这个阶段仍主要是两卡 expert-child 或有限主库，不是 32 卡服务。
- `20261123_prefill_eplb_three_way`：EP16 layer49 replay 和静态 EPLB 能跑，但 K1 服务被原生物理布局 `(256+R) % EP == 0` 拒绝（K=1 得到 257 个物理专家），在线 EPLB 还存在 storage offset/OOM；因此当时三组服务结论仍是 `INCONCLUSIVE`。
- `20261123_global_k_service_ep32`：第一次做出可服务的本地回填版 K=2，但 32K 请求在可见 chunk 后挂起；根因是各 rank 对槽复用/安装的分支不一致，部分 rank 等待通信，部分 rank 直接继续。
- `20261123_global_k_native_service_dispatch`：修复 32K 挂起，采用 owner 真实状态全 gather、MIN 决策、同序 install/reuse；但额外 replica dispatcher 仍在 combine 报 `Invalid attr=601`，所以该轮的 K 主矩阵只能标记为 local-backfill fallback，而不是真正的额外通信编号改道。
- `bd80f52`：完成单次 DeepEP dispatch/combine 的真改道。K=2 的两个 owner 槽、layer49/50、EP16/EP32 均有任务账本；原专家实际少算、owner 收到副本任务、任务守恒、16K/32K output IDs 一致；8K 输出 ID 仍不同，只能按数值容差。EPLB +32 也使用 v0.5.20 NPU 补丁完成在线重排。

## 3. 当前方案与完成度

### 3.1 配置和槽语义

- 配置字段：`K`、`owners`、`hot_experts`、`layers`、`split_num/split_den`、`ep`、`main_per_rank`、`verify`、`quiet`。
- `K` 是全组真实额外专家槽总数；不是每 rank K。当前配置使用固定 owner（例如 EP32 K=2：owners `[0,1]`，hot experts `[163,133]`），先不做动态落点。
- 每个槽的状态包括 owner、local slot、layer、expert、generation、ready、实际 storage pointer 和字节数；复用/重装必须在任意设备收发前由全组一致决定。
- 当前槽大小 `37,810,176 bytes`。K=2 只有两个 owner 各分配一份；其他 rank 不保留每 rank 备用权重。跨 layer49/50 使用同一 storage pointer，generation 递增。
- 不做原生 EPLB 的每 rank +1 冗余；K 通过独立通信编号和外部槽连接，避免 `(256+K) % EP == 0` 的均匀物理布局限制。

### 3.2 真改道流程

```text
真实 router/top-k（逻辑专家和 top-k 权重不变）
→ 选热点任务及其整数配额
→ 在 dispatch 前把选中位置的目标编号改成 owner 的额外通信编号
→ 一次 DeepEP NORMAL dispatch
→ 接收侧按实际 local expert 分组
   主专家：原模型主权重链
   副本：owner 的外部槽 W8A8 链
→ 同一次 DeepEP combine
→ 原 token/top-k 位置正确合并
```

不是“另开一个只发副本的 dispatcher”，也不是把原路径权重置零后本地回填。`bd80f52` 的实际记录中，EP32 的 dispatch 编号数为 288，每 rank 9 个通信位置；任务行数和 owner 接收数均守恒。

### 3.3 已验证范围

| 维度 | 已验证 | 未证明 |
|---|---|---|
| K | EP16 K=1/K=2；EP32 K=2 | 更大 K、任意 K 与原生 layout 的完整组合 |
| 层 | layer49、layer50，单层和两层交替 | 全部 75 层 K 服务 |
| 拓扑 | 两机 32 die 单 EP32；EP16 单机 | 跨更多节点、生产拓扑和非均匀槽 |
| 运行 | 服务请求完整返回，8K/16K/32K | 完整模型生产 graph、真实 Decode K 服务 |
| 槽 | 固定 owner、跨层复用、generation/pointer 校验 | 动态 owner/槽位置、并发多请求、失败恢复全链路 |
| 正确性 | 任务账本、层误差、输出 ID；16K/32K 一致 | 8K token-exact；全部层和全部采样点的 token 一致 |

## 4. 三批结果与主性能表

### 4.1 `bd80f52`：功能验收

- 两机 32 die，单 EP32；原版、在线 EPLB +32、K=2 均跑完 AgentX 派生 8K/16K/32K。
- EPLB +32：10 次 `rebalance end`/rank，布局变化，正式输出 ID 与原版一致；层49/50 的 K=2 任务账本证明原专家少算、owner 副本接手。
- K=2 的 32K 任务例子（隔离首个 chunk）：
  - expert163：逻辑 383 个任务，改道 181 个到 owner0，原专家实际 202；
  - expert133：逻辑 6 个任务，改道 2 个到 owner1，原专家实际 4；
  - owner 接收 `[181,2]`，任务守恒。
- 32K 隔离请求和 8K→16K→32K 连续请求都完成；挂起根因和修复见第 5 节。
- K=2：16K/32K 输出 ID 与原版一致；8K output IDs 不同，但层输出 finite、有界，不能叫 token-exact。

**功能证据**：`shared/current_exp/20261123_ep32_native_reroute_and_online_eplb/REPORT_FOR_REVIEW.md`，`artifacts/EP32_K2_NATIVE_TASK_ACCOUNTING.json`，`artifacts/EP32_EPLB_ONLINE_EVIDENCE.json`，`artifacts/EP16_SLOT_CONSISTENCY.json`。

### 4.2 `bf94982`：重复测速（EPLB 2 迭代历史对照）

本轮保留了原版和 K=2 的 n=10 配对数据；EPLB 当时使用每 **2 个调度迭代**重排，长请求频繁跨重排。其正式 EPLB 样本与重排重叠很多，不能作为稳态 EPLB 数字。

| 输入 | 原版 TTFT | EPLB +32（2 迭代）TTFT | K=2 TTFT | EPLB 相对原版 | K=2 相对原版 |
|---|---:|---:|---:|---:|---:|
| 8K | 2.8025 s | 3.1255 s | 2.9358 s | +11.57% | +4.72% |
| 16K | 5.7147 s | 12.4076 s | 5.9785 s | +117.33% | +4.53% |
| 32K | 11.7875 s | 22.4916 s | 12.3013 s | +90.81% | +4.42% |

这里 EPLB 的 16K/32K 中位数明显变慢，是因为正式请求与重排重叠；这是“重排频率过高”的历史证据，不是最终方案。原版/K=2 的整次请求中位数分别为：8K `3.3236/3.4774`，16K `6.2401/6.4972`，32K `12.3165/12.8586` 秒。路径：`shared/current_exp/20261123_ep32_perf_repeat/REPORT_FOR_REVIEW.md`。

### 4.3 `d0b351d`：EPLB 1000 迭代主结果

- 只重测 EPLB +32，`--eplb-rebalance-num-iterations 1000`，校准和正式计时都保持 1000。
- 每轮使用 64-token AgentX-derived 独立校准流量，**496 条请求/轮**；三轮均出现 `rebalance end`，布局 before/after 变化。
- 单轮重排约 24.390–24.655 s；正式 8K/16K/32K 的 10/10 样本均没有再与重排重叠。
- 原版和 K=2 直接引用 `bf94982`，不是本轮重跑。

主表（TTFT 中位数；括号为整次请求中位数；每组每档 n=10）：

| 输入 | 原版（bf94982） | EPLB +32（d0b351d） | K=2（bf94982） | EPLB ΔTTFT | K=2 ΔTTFT |
|---|---:|---:|---:|---:|---:|
| 8K | 2.8025 s（3.3236 s） | **2.5969 s（3.1665 s）** | 2.9358 s（3.4774 s） | **-7.34%** | +4.72% |
| 16K | 5.7147 s（6.2401 s） | **5.3072 s（5.8814 s）** | 5.9785 s（6.4972 s） | **-7.09%** | +4.53% |
| 32K | 11.7875 s（12.3165 s） | **11.2183 s（11.7814 s）** | 12.3013 s（12.8586 s） | **-4.85%** | +4.42% |

EPLB 的整次请求相对原版为 `-4.73% / -5.75% / -4.34%`。这是同一输入和同一跨批基线下的观测差，不是生产 p95。

**性能证据**：`shared/current_exp/20261123_ep32_eplb1000/REPORT_FOR_REVIEW.md`、`artifacts/EPLB1000_SUMMARY.json`、`artifacts/EPLB1000_RAW_REQUESTS.csv`、`artifacts/raw/formal/`、`artifacts/raw/warmup/`、`artifacts/raw/calibration/eplb1000_calibration.jsonl`。

## 5. 关键故障：发现、修复、验证

### 5.1 `storage_offset must be 0`

**发现**：EPLB 从 NZ 专家权重切片直接 P2P；HCCL 对内部格式要求 storage offset 为 0，重排在第一只专家就失败。

**修复**：`expert_location_updater_patched.py` 在发送前把切片转 ND，接收 buffer 用 ND，收完写回 NZ；配套 `torch_npu_distributed_c10d_patched.py` 使用原子 batch P2P 避免死锁。v0.5.20 版本进一步按 P2P 批次延迟转换，控制临时显存。

**验证**：`d0b351d` 两机 EP32 +32 在线组每轮有 `rebalance end`，layout before/after 变化；补丁在两节点运行时哈希一致。证据：`shared/current_exp/20261123_ep32_eplb1000/artifacts/EPLB1000_SUMMARY.json`、`shared/current_exp/20261123_ep32_native_reroute_and_online_eplb/artifacts/EP32_EPLB_PATCHES.json`。

### 5.2 `Invalid attr=601`

**发现**：早期“额外 replica dispatcher”和“本地回填后的 combine”路径进入 `aclnnCamMoeCombineNormal` 时被拒绝。根因不是 planner，而是 combine 输入行的布局/dtype 边界没有与 dispatch 对齐；后续单次 dispatch 集成中，INT8 dispatch 的接收行不能直接当 BF16 combine 行。

**修复**：保留 dispatch 的 INT8 输入与 per-token scale 给 W8A8 链，但为 combine 分配 BF16 `[received_rows, hidden]`；改为同一 dispatcher 的主+副本任务，不再另开只发副本的 dispatcher。

**验证**：`bd80f52` 的 EP16 K=1/K=2、EP32 K=2 均完成；`EP32_K2_NATIVE_TASK_ACCOUNTING.json` 中 `task_conservation=true`，任务行数/owner 接收数一致。旧失败保留在：`shared/current_exp/20261123_global_k_native_service_dispatch/artifacts/NATIVE_DISPATCH_FAILURE.json`、`shared/current_exp/20261123_ep32_native_reroute_and_online_eplb/artifacts/failures/EP16_SINGLE_DISPATCH_INCIDENTS.json`。

### 5.3 32K 挂起

**发现**：早期 K2 EP32 32K 请求在可见 chunk 后不返回；不是任务守恒或槽数值错误，而是不同 rank 对槽复用/安装的分支不同，部分 rank 在通信等待，部分继续，最后缺少完整的收尾事件。

**修复**：owner 发布真实 slot 状态；所有 rank `all_gather_object`，再做 MIN 决策；只有全组一致 install 才按统一顺序 send/recv，复用则所有 rank 跳过安装。槽身份包含 owner、local slot、layer、expert、generation、ready 和 storage pointer。

**验证**：`bd80f52` 的 32K 隔离请求和 8K→16K→32K 连续请求都完整返回；同层/跨层 reuse 日志和 q=0 通过。证据：`shared/current_exp/20261123_global_k_native_service_dispatch/REPORT_FOR_REVIEW.md` 与 `artifacts/SLOT_CONSISTENCY.json`。

## 6. 复现入口

### 当前功能验收（bd80f52）

```bash
cd shared/current_exp/20261123_ep32_native_reroute_and_online_eplb
./reproduce.sh validate
# 完整服务阶段见同目录 reproduce.sh 的 ep16-*、ep32-original、ep32-eplb-online、ep32-k2
```

### 重复测速（bf94982）

```bash
cd shared/current_exp/20261123_ep32_perf_repeat
./run_perf.sh all
./run_perf.sh analyze
```

该入口包含原版、EPLB 2 迭代和 quiet K=2；在正式运行前会生成/校验输入。

### EPLB 1000 迭代（d0b351d）

```bash
cd shared/current_exp/20261123_ep32_eplb1000
./run_eplb1000.sh all
```

入口使用 32 die、单 EP32、8K/16K/32K、每轮 4+3+3 次正式请求和独立校准流量。日志和生成输入默认不提交；性能原始记录在 `artifacts/raw/`。

### 历史和通用证据

- 精选历史索引：`shared/history_exp/README.md`、`shared/history_exp/MANIFEST.md`
- EP64 EPLB 故障与补丁：`shared/history_exp/01_EP64统一服务_2026-09/EPLB_实验记录.md`、`sglang_npu_eplb_patches/README.md`
- PD 热力：`shared/history_exp/02_PD分离2P2D_2026-09-20/README.md`
- EPLB 负优化：`shared/history_exp/07_PD共置与EPLB负优化_2026-10-11/Q4Exp/Q4Exp_总结报告.md`、`eplb_algo_sim/PROBLEM_EPLB负优化.md`
- 复杂模拟：`shared/current_exp/20261103_complex_moe_sim/RESULTS.md`
- Prefill copy-aware：`shared/current_exp/20261109_decision/20261109_074727/RESULTS.md`、`20261123_prefill_eplb_three_way/REPORT_FOR_REVIEW.md`
- 两卡回本：`shared/current_exp/20261122_global_k_breakeven/REPORT_FOR_REVIEW.md`
- 全组槽：`shared/current_exp/20261116_global_k_online/REPORT_FOR_REVIEW.md`
- 服务修复：`shared/current_exp/20261123_global_k_native_service_dispatch/REPORT_FOR_REVIEW.md`

## 7. 已知限制和下一步边界

- 输入全部是 AgentX 块哈希/长度驱动的确定性 token 物化；没有原始人类文本或原始模型 token ID，不能把它叫原文重放。
- `8K` K=2 有已知 output ID 差异；层误差和任务守恒通过，但不能宣布 token-exact。`16K/32K` 在当前批次 ID 一致。
- K 功能验证主要是 layer49/50、固定 owner、单请求；不能外推全 75 层、Decode、多请求或动态 slot 落点。
- 在线 EPLB 的 `d0b351d` 主表是校准后正式计时，正式样本没有重排重叠；重排约 24.4–24.7 s，校准 496 条请求/轮，成本不能藏在正式中位数里。
- `bf94982` 的 EPLB 2 迭代数据只作为重排频率过高的历史对照；不能与 `d0b351d` 混成同一稳态。
- 当前性能是单服务、单批样本和服务请求中位数，不是生产 p95、全模型 TTFT 保证或吞吐基准。
- 尚未完成：全部层的 K 服务接入、Decode K、动态槽位置、更多 K 值与并发、生产 graph 等价性、完整故障恢复和跨更多节点。
