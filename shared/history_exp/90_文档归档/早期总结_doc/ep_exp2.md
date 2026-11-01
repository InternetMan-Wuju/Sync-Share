# GLM-5.2 EP / EPLB 实验方案 v2

## 0. 实验目的与变量

量化大 EP 下的 expert 负载不均衡，并比较 EPLB 以及不同“补丁 / NZ”配置对性能和负载的影响。

变量：

- EP：32、64
- EPLB：关闭、开启
- EPLB 配置：补丁1+2 / NZ开，补丁2 / NZ开（待验证），补丁2 / NZ关
- 测量类型：专家热度、性能指标

## 1. 部署约定

- EP=64：NNODES=4，TP=EP=64，每机 16 卡。
- EP=32：NNODES=2，TP=EP=32，2 机 × 16 卡（减少跨机通信开销）。建议用 10.120.72.23 + 10.120.72.25，其余两台空闲；具体哪两台可按实际拓扑再调整。
- 补丁 / NZ 三种配置：
  - 补丁1+2 / NZ开：补丁1（`expert_location_updater`）+ 补丁2（torch_npu 原子 P2P），NZ 保持开启；
  - 补丁2 / NZ开：只挂补丁2，NZ 保持开启（待验证）；
  - 补丁2 / NZ关：只挂补丁2，同时 `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`。

### 1.1 数据集与请求集

- 使用 `dataset/bench_multi.py` + 三份数据集（dolly / dolly-zh / alpaca-en）。
- 固定 `--sample N --stratify dataset --seed 0`，保证本轮所有实验跑同一批请求、同一顺序。
- 每个实验压测时长：`--duration 1800`（30 分钟，可调）。
- 记录实际 token 统计：每请求 `prompt_tokens` / `completion_tokens` 的 total、mean、min、max、P50、P90、P99，以及实际完成条数和 by_file 分布。
- 数据集格式、字段、脚本参数详见 `doc/datasets_run.md`。

## 2. 实验清单（按重要性顺序）

> 本轮（约 1.5 小时）只执行编号 1–4；编号 5–10 后续再跑。

| 编号 | 配置 | 测量内容 | 备注 |
|---|---|---|---|
| 1 | EP=64，无 EPLB | 专家热度 | 同时记录性能基线，作为 EP64 对照组 |
| 2 | EP=64，EPLB，补丁1+2，NZ开 | 性能 | 推荐配置，已跑通 |
| 3 | EP=32，无 EPLB | 专家热度 | 同时记录性能基线，作为 EP32 对照组 |
| 4 | EP=32，EPLB，补丁1+2，NZ开 | 性能 | 推荐配置 |
| 5 | EP=64，EPLB，补丁2，NZ开 | 性能 | 待验证：可能 storage_offset 崩溃 |
| 6 | EP=32，EPLB，补丁2，NZ开 | 性能 | 待验证，同上 |
| 7 | EP=64，EPLB，补丁2，NZ关 | 性能 | 已跑通（NZ 关） |
| 8 | EP=32，EPLB，补丁2，NZ关 | 性能 | |
| 9 | EP=64，EPLB | 专家热度 | 默认补丁1+2、NZ开 |
| 10 | EP=32，EPLB | 专家热度 | 默认补丁1+2、NZ开 |

## 2.1 执行顺序（本轮）

先做 **0 号「数据可获取性冒烟」**，再按 1 → 2 → 3 → 4 顺序跑正式实验。
理由：正式实验每个 30 分钟、还要重启服务，代价大；先花几分钟确认
§3 要的每一项指标都能从采集产物里算出来，避免“跑完才发现指标没算出来”。

1. **0 冒烟**：用一份已有 run 产物，跑通 §3 的指标计算（专家热度三件套、
   rank 负载、TTFT/ITL/e2e、AICore），确认「能」再往下。
2. **1 → 2 → 3 → 4**：每个 30 分钟，顺序执行。
3. **每个跑完立即做产物齐全检查**：关键文件 + 关键指标缺一就停下报错，
   不盲目进下一个。**至少保证 1、2 完成。**

## 3. 每个实验要采集的具体数据

### 3.1 通用（所有实验都记录）

- 环境快照：SGLang / torch / torch_npu / CANN / driver 版本，启动命令，TP / EP / NNODES，补丁配置，NZ 开关，模型 config hash。
- 请求清单：数据集、请求数、并发、max_tokens、采样参数（temperature / top_p / seed）、请求顺序、清单 hash。
- 原始文件：`dolly_bench.json`、`expert_distribution.json/.pt`、`metrics_*.prom`、`npusmi_*.txt`、`summary.json`。
- 重复：每组至少 3 次，报告每次结果和中位数 / P50。

### 3.2 专家热度实验（编号 1、3、9、10）要的数据

1. 全局专家热度总和
   - `logical_count` 按 time、layer 求和，得到每个逻辑专家（256 个）的总 token 数。
   - 指标：total / mean / min / max / std / CV、peak/mean、零热度专家数。
   - 集中度：top1 / top5 / top10 / top20 专家 token 占比。
2. 逐层专家热度
   - 每层（78 层）× 每专家热度；看热专家集中在哪几层、层间差异。
3. 某一时刻专家热度
   - 多个时间点 t 的 `logical_count[t]` 快照，统计该时刻 per-expert 热度、top-k、零热度数。
   - 判断热度是否随时间漂移。
4. rank 负载
   - 按 physical→logical 映射，把专家热度汇总到每个 rank（32 或 64 个 rank）。
   - 指标：min / max / mean / CV、peak/mean、max/min。
   - 画 rank 负载直方图或热力图。
5. 不均衡结论三件套：CV、peak/mean、max/min。

> 编号 1、3 是无 EPLB，必须同时记录性能基线，否则编号 2、4 等 EPLB 性能没东西可比。

### 3.3 性能实验（编号 2、4、5、6、7、8）要的数据

1. 请求级
   - wall_s、ok / err。
   - e2e：P50 / P90 / P99 / mean / max。
   - requests/s、total tokens/s、completion tokens/s。
2. token 级
   - TTFT（首 token 延迟）：P50 / P90 / P99。
   - ITL / TPOT（每输出 token 间隔）：P50 / P90 / P99。
   - prefill token/s、decode token/s。
3. 硬件
   - 每 NPU 的 AICore% 均值 / 峰值、内存占用、功耗。
   - 各 rank AICore% 的不均衡（CV）。
4. EPLB 专项
   - rebalance 次数、每次耗时。
   - rebalance 前后 expert / rank 负载的 CV、peak/mean（验证 EPLB 是否真的把负载变均衡）。
   - 去掉 rebalance 暂停后的稳态性能：分“rebalance 前 / rebalance 后”统计。

## 4. 数据工具

- 一键跑：`run_exp.sh <编号 1..10>`（自动停旧服务、按配置启动、等健康检查 200、再跑 30 分钟混合数据集采集）。
- 启动：`start_ep64_decode.sh` / `serve_glm52_ep64_decode.sh`（已支持 `EP_SIZE`、`ENABLE_EPLB`、`PATCH12`、`PATCH2`、`NZ_OFF` 环境变量）。
- 采集：`ep_lab/run_multi_ep_cell.sh`（混合数据集 + 时长预算，支持 `NODES`、`DURATION`、`SAMPLE`、`STRATIFY`、`SEED`、`CONCURRENCY`、`MAX_TOKENS`、`RUN_NAME`）。
- 快速冒烟：`ep_lab/run_dolly_ep_cell.sh`（32 条 Dolly，用于先验证服务是否正常）。
- 后处理一键：`ep_lab/analyze_run.sh <run_dir> [ep_size]`（跑完每个实验后调用，产出最终指标）。
- 分析：
  - `ep_lab/analyze_expert_heat.py`（容器内跑：全局专家热度、逐层、时刻快照、rank 负载、不均衡三件套）；
  - `ep_lab/perf_from_prom.py`（宿主机跑：e2e mean/max/P50/P90/P99、requests/s、completion tokens/s、TTFT/ITL/e2e 分位）；
  - `ep_lab/dataset_stats.py`、`ep_lab/summarize_dolly_run.py`、`ep_lab/analyze_expert_distribution.py`、`ep_lab/analyze_perpass_rank_load.py`。

编号到参数的映射在 `run_exp.sh` 的 `case` 里，和上文实验清单一致。
