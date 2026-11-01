# 任务2 Chunk Size —— 本轮实跑配置（2026-10-12）

> **历史方案，不能作为10月12/13日自主实验的配置或节点授权。** 本文件下面记录的是早前node20、uniform IDs及mem0.92方案；当前自主实验使用23、SGLang原生random-ids和mem0.90。实际配置见[RUN_CONFIG_20261013.md](RUN_CONFIG_20261013.md)。早前关于auto下MTP必崩的描述也不适用于全部配置：自主实验已在缩小dispatch/Graph配置后成功运行MTP auto。

## 机器与镜像
| 项 | 值 | 备注 |
|---|---|---|
| 机器 | **node20 (10.120.72.20)** | 19 被别人占(64G HBM)、23 我的 MTP 在跑、25 被别人 ep-pool 占；只剩 20 空闲 |
| 驱动 | 26.1.1 | node19/23 是 25.5.0（**跨机结果别直接混比**） |
| 镜像 | `quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3` | KV 池最大（历史实测 70912 @ mem0.92） |
| 模型 | `/data_lib/data/models/glm-q` | 只读挂载 |

## 固定配置（除 chunk 外全不变）
| 参数 | 值 |
|---|---|
| 并行 | `--tp-size 16 --dp-size 1 --ep-size 16`（单机 16 卡，同 ChunkSize 历史实验口径） |
| DeepEP | `--moe-a2a-backend deepep --deepep-mode auto`（**快路径**：decode 走 low-latency） |
| Decode Graph | 开（`full`, max-bs 32, bs 8/16/24/32）；实跑会记录是否真生效（`npu graph: True`） |
| MTP | **关**（本任务主变量不是投机解码） |
| MEM / MAXPREF / CTX / MRR | 0.92 / 32768 / 131072 / 16 |
| 环境 | `SGLANG_NPU_USE_MULTI_STREAM=1`、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、`HCCL_BUFFSIZE=650` 等（同既有脚本） |

> 注意：`--deepep-mode auto` 下 **MTP 会崩**（tiling/log_inner 报错）；本任务 MTP=关，所以能用快路径。
> 历史上 ChunkSize 实验是在 `deepep normal` 下跑的（graph 被禁），本轮**换成 auto+graph on**，
> 与榜单 3.2 的环境一致 —— 报数时要注明这一点。

## 数据集口径
- 均匀随机 token id（1..154879），每条 case 独立种子固定，`temperature=0`、`ignore_eos=True`。
- **8k/1k**：8192 输入 / 1024 输出；请求数 = 并发 × 2（AboutModels）。
- 每轮前 `flush_cache`，保证冷启动（对比 chunk 时口径一致）。
- 一个场景一个数据集文件；并发 N 用 `--limit N*2` 取前 N*2 条（跨 chunk 值**完全相同的请求集**）。

## 本轮扫的网格（最小起步）
- 场景：**w1_8k_1k**
- chunk：**1024 / 4096 / 8192 / 16384**
- 并发：**4 / 8**
- 每轮：起服务(~6-7min) → 等 health → 逐并发 flush+打负载 → 停服务 → 留 45s
