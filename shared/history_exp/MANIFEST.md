# history_exp 远端精选镜像清单

生成日期：2026-11-02

- 本地完整归档：`/data_lib/cyt/workspace/history_exp`
- 精选镜像：`/data_lib/cyt/workspace/Share/shared/history_exp`
- 已同步：**652 个文件，14.9 MiB**
- 原始归档：**22708 个文件，17.6 GiB**

## 收录规则

- 全部 `.md` 实验文档、README、总结、结论与分析。
- 实验脚本、分析脚本、shell 启动脚本、补丁和配置。
- 名称含 summary/report/index/status/coverage/overview/汇总/指标/结果/实测/覆盖/核对/分析的小型 JSON/CSV。
- 顶层 `tables/` 下的 CSV 汇总表。
- 顶层 `figs/`、`figures/` 和分析目录中的关键图。

## 未同步内容

- `results/` 中除 Markdown 和精选图表外的原始结果。
- `logs/`、`metrics/`、`datasets/`、`weights/`。
- `.pt`、`.safetensors`、大型 manifest、原始请求/响应和缓存。
- 上游源码完整快照、重复 patch tree、Triton cache 和 PDF 工具依赖。
- 单文件超过 1 MiB 的文本/代码或超过 2 MiB 的图表。

原始数据仍保存在本地 `/data_lib/cyt/workspace/history_exp/`；需要时可单独索取具体路径。

## 文件统计

| 目录 | 镜像文件数 | 镜像大小 | 原归档文件数 | 原归档大小 |
|---|---:|---:|---:|---:|
| `01_EP64统一服务_2026-09` | 51 | 417.6 KiB | 51 | 417.6 KiB |
| `02_PD分离2P2D_2026-09-20` | 26 | 268.2 KiB | 35 | 963.3 KiB |
| `03_ChunkSize单机_2026-09` | 160 | 5.3 MiB | 5418 | 9.1 GiB |
| `04_MTP推测解码_2026-09` | 30 | 151.8 KiB | 1184 | 223.5 MiB |
| `05_EPLB冗余专家_1P1D_2026-09-28` | 17 | 258.7 KiB | 4900 | 2.6 GiB |
| `06_四节点服务与权重核查_2026-10-11` | 19 | 36.2 KiB | 452 | 67.5 MiB |
| `07_PD共置与EPLB负优化_2026-10-11` | 126 | 4.4 MiB | 4668 | 4.1 GiB |
| `08_MTP推测解码续_2026-10-11` | 64 | 619.6 KiB | 234 | 15.5 MiB |
| `09_自主探索与ChunkSize_2026-10-12` | 121 | 3.0 MiB | 5595 | 1.5 GiB |
| `90_文档归档` | 36 | 481.9 KiB | 37 | 487.5 KiB |
| `91_工具` | 1 | 5.6 KiB | 133 | 3.1 MiB |
| `README.md` | 1 | 10.8 KiB | 1 | 10.8 KiB |

## 校验

文件哈希见仓库中的 `SHA256SUMS`。重新生成镜像可执行：

```bash
python3 tools/sync_history_snapshot.py
```
