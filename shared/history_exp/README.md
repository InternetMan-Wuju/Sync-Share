> **远端精选镜像**：这里保留实验总结、重要文档、关键代码/补丁和精选结果。
> 原始 `.pt`、权重、日志、数据集和大结果未提交到 Git，仍保存在本地完整归档中。
> 收录范围与统计见 [`MANIFEST.md`](MANIFEST.md)。

# history_exp 索引（GLM-5.2 / Ascend 910C 实验归档）

整理时间：2026-10-11；2026-11-02 完成第二轮归档。两轮均将原 `current_exp/` 全部清空并按下表归类到此目录。
**所有搬运都是同盘 `mv`（不改字节、不占额外空间）**，各目录内部的相对结构保持原样。

```
history_exp/
├── 01_EP64统一服务_2026-09/          统一服务 EP64 + EPLB（最早一轮）
├── 02_PD分离2P2D_2026-09-20/         PD 分离 2P2P，逐窗口 EP-Rank 热力统计
├── 03_ChunkSize单机_2026-09/         单机 Chunk Size（8192 vs 1024）三轮
├── 04_MTP推测解码_2026-09/           MTP / NEXTN 推测解码
├── 05_EPLB冗余专家_1P1D_2026-09-28/  1P1D + EPLB + 冗余专家 A/B/C
├── 06_四节点服务与权重核查_2026-10-11/ 四节点拉起 + 权重损坏核查
├── 07_PD共置与EPLB负优化_2026-10-11/  PD 共置、Q4 联合实验、EPLB 负优化与层级偏移
├── 08_MTP推测解码续_2026-10-11/      MTP 快速验证、Q1 MTP 汇总与后续补测
├── 09_自主探索与ChunkSize_2026-10-12/ MTP/ChunkSize 自主探索及低并发预实验
├── 90_文档归档/                      方案、需求、总结、分析报告
└── 91_工具/                          通用脚本与依赖
```

---

## 01_EP64统一服务_2026-09
**内容**：19/25/31/32 四机 TP=EP=64 统一服务下的 EP / EPLB 实验，含 dataset、ep_lab 分析工具、patches。
**关键结论**：EPLB 三个坑与解法（FRACTAL_NZ 权重搬运、torch_npu P2P 死锁、rebalance 耗时）；EP64 下 expert 极不均衡（logical CV≈3.59、peak/mean≈58.6，按 rank 聚合后 CV≈0.846）。
**入口**：`EPLB_实验记录.md`、`README.md`。

## 02_PD分离2P2D_2026-09-20
**内容**：prefill 23/25（EP16×2）+ decode 31/32（EP32）的 PD 分离实验，16K/1024，并发 10/20/30。
**关键结论**：`logical_count` 是逻辑专家计数，**必须用每个 rebalance 窗口自己的 `physical_to_logical_map`** 才能算 EP-Rank 热度；EPLB 后物理 rank 明显更均衡（Rank CV 0.105–0.173 远低于 Logical CV 0.198–0.267）。
**入口**：`README.md`、`collect_pd_heat.sh`。

## 03_ChunkSize单机_2026-09
**内容**：23/25 单机 TP16/EP16 上的 `--chunked-prefill-size` 对照，三轮：
- `chunk_single23/`（9/26 首轮，**数据集口径曾搞错**，把"32K 前缀"理解成公共前缀 32K）
- `chunk_manual_20260927/`（9/27 人工轮，**发现权重损坏并建立只读权重外挂**，体积最大 9.2G）
- `chunk_redo_20260928/`（9/28 正式复测，结论以这轮为准）

**关键结论**：8K 优于 1K（TTFT 5.976s vs 10.110s，−40.89%）；长 Prefill 插入时正在生成的请求最长停顿 3.1s（8K）vs 9.3–9.6s（1K）。注意本轮 `--enable-mixed-chunk=False`、Graph 实际关闭。
**权重事故**：见 `chunk_manual_20260927/semantic_debug_20260927/readonly_resume/README.md`。

## 04_MTP推测解码_2026-09
**内容**：`mtp_preview25/`（9/26 探索）、`mtp_full_20260928/`（9/28 三场景各 45 分钟，23=OFF / 25=ON）。
**关键结论**：窗口吞吐变为 OFF 的 3.706× / 3.148× / 2.224×；TPOT 476→186 ms；**TTFT 反而变慢约 18.8%**（32K/300 场景）；接受长度 2.77–4.00；KV 池 63744→43008 token（−32.5%）。
**入口**：`mtp_full_20260928/results/20260928_061159/report.md`。

## 05_EPLB冗余专家_1P1D_2026-09-28
**内容**：23 单机 prefill + 25 单机 decode 的 1P1D，A（EPLB 关）/ B（开，冗余 0）/ C（开，冗余 16）对照，AgentX 派生请求。
**状态**：**未完成**。A 组有有效 600s 数据；B 组被跳过；C 组启动即 OOM。历史上还留下了 CPU/Gloo 统计归并、lazy-send 两个衍生补丁。
**入口**：`README.md`（含完整失败与调整时间线）。

## 06_四节点服务与权重核查_2026-10-11
**内容**：19/23/31/32 四台各起独立单机 TP16/EP16 服务并做真实问询验证；权重损坏根因核查。
**关键结论**：
- 四台服务全部 health=200、真实中文问询正常，无历史故障里的重复怪符号；
- `/mnt/raid` 是**降级 RAID5**（`[4/3] [UUU_]`），读盘仅 250 MiB/s，很可能就是"同一文件两次读出不同字节"的根因；
- 权重路径已统一改为 nvme 的 `/data_lib/data/models/glm-q`（四台机元数据指纹一致，加载 4–5 分钟）；
- 与 9/27 参考快照相比仍有 **333/23093 块（1.44%）不同、30900 个非有限值**，且"黄金参考"本身已部分损坏（4 个块全集群不可复原）。**MTP 层（78 层）未被波及**。
**入口**：`06_.../README.md`；服务管理 `serve.sh {start <run_id>|status|stop} [19 23 31 32]`。

## 07_PD共置与EPLB负优化_2026-10-11
**内容**：`pd_colloc_20261011/` 为四机 PD 共置与 EPLB A0/A/B/C 联合实验（4.1G 原始结果）；`Q4Exp/` 为任务 3+4 的指标、图表与总结；`eplb_algo_sim/` 保存负优化形式化、最小反例和离线仿真；`eplb_layer_shift_review_20261014/` 为后续层级偏移调查。
**关键结论**：EPLB 逐层独立优化会破坏跨层负载抵消，真机 rank 负载由约 1.3–1.4× 恶化到 5.6–6.1×；增加 64 个冗余专家可拉回约 1.26–1.38×，代价是专家权重显存增加约 25%。
**入口**：`pd_colloc_20261011/README.md`、`Q4Exp/Q4Exp_总结报告.md`、`eplb_algo_sim/PROBLEM_EPLB负优化.md`、`eplb_layer_shift_review_20261014/调查结论.md`。

## 08_MTP推测解码续_2026-10-11
**内容**：`mtp_quick_20261011/` 为 MTP 快速探测与启动脚本；`Q1MTPExp/` 汇总 MTP 历史/新增记录、接受率复核、问题定位、W3 补跑和 910C 榜单对照。
**状态**：该阶段 MTP 正式 ON/OFF 全覆盖未完成，索引中保留了实际覆盖范围和缺失项，不将预实验值当作正式结论。
**入口**：`Q1MTPExp/MTP实验阶段汇报_20261012.md`、`Q1MTPExp/MTP全部实验汇总_含自主探索.md`、`Q1MTPExp/W3补跑_20261013.md`。

## 09_自主探索与ChunkSize_2026-10-12
**内容**：`autonomous_20261012/` 保存 MTP/ChunkSize 自主探索、网络依据、权重恢复、原始结果与最终总结；`Qchunksize/` 保存 ChunkSize 四档、五场景及复测数据。
**状态**：ChunkSize 正式 W1 要求为并发 32/64，本轮实际主要为并发 2/4 的低并发预实验；MTP 和 ChunkSize 的未完成项在各自索引中明确标注。
**入口**：`autonomous_20261012/README.md`、`autonomous_20261012/自主实验最终总结.md`、`Qchunksize/ChunkSize要求与自主探索偏差.md`、`Qchunksize/ChunkSize自主实验实测.md`。

---

## 90_文档归档
- `早期方案_root/` ← 原 `history_exp/root/`：`Cluster.sh`、`GLM5.2_EPLB_measurement_and_experiment.md`、`GLM5.2_EP_EPLB_four_node_experiment_plan.md`、`Test1.md`
- `早期总结_doc/` ← 原 `history_exp/doc/`：`FinalTest.md`、`历史总结_2026-09-20_模型交接.md`、`问题总结`、`固定测试*.md`、`ep_exp2.md`、`TTFT残差分析.md`、`会议字稿.txt` 等
- `分析报告_2026-09/` ← 原 `current_exp/doc/`：ChunkSize ×7 篇、MTP ×3 篇、`EPLB_16K_c10_ON_vs_OFF_模拟对照.md`（**该文 OFF 列是推算值，不是实测**）、`模型回复异常与只读权重外挂恢复_2026-09-28.md`
- `需求与分析报告_2026-10/` ← 第二轮归档的 `current_exp/docs/`、`current_exp/README.md` 和 `temp.txt`：会议记录、模型与 ChunkSize 参考资料、本轮批次索引。

## 91_工具
- `pdf_tools/`：pypdf 离线依赖
- `serve_glm52_tp32_23_25.sh`：两机 TP32 统一服务启动脚本

---

## 旧路径 → 新路径 对照

| 旧路径（原样） | 新路径 |
|---|---|
| `history_exp/glm52_ep64_chengyetai/` | `history_exp/01_EP64统一服务_2026-09/` |
| `history_exp/current_exp/` | `history_exp/02_PD分离2P2D_2026-09-20/` |
| `history_exp/root/` | `history_exp/90_文档归档/早期方案_root/` |
| `history_exp/doc/` | `history_exp/90_文档归档/早期总结_doc/` |
| `current_exp/chunk_single23/` | `history_exp/03_ChunkSize单机_2026-09/chunk_single23/` |
| `current_exp/chunk_manual_20260927/` | `history_exp/03_ChunkSize单机_2026-09/chunk_manual_20260927/` |
| `current_exp/chunk_redo_20260928/` | `history_exp/03_ChunkSize单机_2026-09/chunk_redo_20260928/` |
| `current_exp/mtp_preview25/` | `history_exp/04_MTP推测解码_2026-09/mtp_preview25/` |
| `current_exp/mtp_full_20260928/` | `history_exp/04_MTP推测解码_2026-09/mtp_full_20260928/` |
| `current_exp/eplb_pd_agentx_20260928/` | `history_exp/05_EPLB冗余专家_1P1D_2026-09-28/` |
| `current_exp/bringup_4node_20261011/` | `history_exp/06_四节点服务与权重核查_2026-10-11/` |
| `current_exp/doc/` | `history_exp/90_文档归档/分析报告_2026-09/` |
| `current_exp/pdf_tools/` | `history_exp/91_工具/pdf_tools/` |
| `current_exp/serve_glm52_tp32_23_25.sh` | `history_exp/91_工具/serve_glm52_tp32_23_25.sh` |
| `current_exp/pd_colloc_20261011/` | `history_exp/07_PD共置与EPLB负优化_2026-10-11/pd_colloc_20261011/` |
| `current_exp/Q4Exp/` | `history_exp/07_PD共置与EPLB负优化_2026-10-11/Q4Exp/` |
| `current_exp/eplb_algo_sim/` | `history_exp/07_PD共置与EPLB负优化_2026-10-11/eplb_algo_sim/` |
| `current_exp/eplb_layer_shift_review_20261014/` | `history_exp/07_PD共置与EPLB负优化_2026-10-11/eplb_layer_shift_review_20261014/` |
| `current_exp/mtp_quick_20261011/` | `history_exp/08_MTP推测解码续_2026-10-11/mtp_quick_20261011/` |
| `current_exp/Q1MTPExp/` | `history_exp/08_MTP推测解码续_2026-10-11/Q1MTPExp/` |
| `current_exp/autonomous_20261012/` | `history_exp/09_自主探索与ChunkSize_2026-10-12/autonomous_20261012/` |
| `current_exp/Qchunksize/` | `history_exp/09_自主探索与ChunkSize_2026-10-12/Qchunksize/` |
| `current_exp/docs/` | `history_exp/90_文档归档/需求与分析报告_2026-10/` |
| `current_exp/temp.txt` | `history_exp/90_文档归档/需求与分析报告_2026-10/temp_20261012.txt` |
| `current_exp/README.md` | `history_exp/90_文档归档/需求与分析报告_2026-10/current_exp_索引_2026-10-13.md` |

### ⚠️ 归档内的绝对路径已失效
归档脚本/文档里大量出现 `/home/chengyetai/workspace/current_exp/...` 的硬编码绝对路径，第二轮归档的脚本与报告也保留了原路径。
**这些是历史产物，作为记录保留原样，没有批量改写**；若以后要重跑某个旧脚本，需要先手工改路径。
唯一例外：本次的 `06_.../serve.sh` 已更新为新路径（因为服务还在跑）。

### 备注
- 节点 31 上仍有历史遗留的**参考权重块服务**（CPU 进程，`reference_block_server.py`，端口 30121），只读、不占 NPU；其快照路径随本次搬运变为
  `history_exp/03_ChunkSize单机_2026-09/chunk_manual_20260927/semantic_debug_20260927/readonly_resume/reference_snapshot/`。
  如果要重启它，需要按新路径调整 `reference_service.sh`。
- `workspace/common/` 是跨轮次的通用资产，**未搬动**。
