# current_exp

2026-10-13 最新任务：正在23补跑完整 W3（1K/32K）的 MTP ON 与四档 ChunkSize，进度及结果见 [W3补跑](autonomous_20261012/w3_supplement_20261013/W3补跑结果.md)。用户已取消本轮 MTP OFF 对照；下文覆盖数为补跑前的探索阶段记录。

当前包含2026-10-11之后的PD/EPLB、MTP及ChunkSize实验。2026-10-13自主探索阶段已结束，**MTP四组正式任务尚未完成：规定长度ON覆盖3/4，OFF性能基线0/4**。23节点自有容器于2026-10-13 12:44核对为全部停止并保留，NPU当时空闲。

**ChunkSize也未完成W1正式要求：应并发32、64请求，实际仅测并发2/4。** 当前34单元属于低并发预实验，不能作为W1榜单的正式Chunk结论。

| 工作 | 入口 |
|---|---|
| ChunkSize W1并发32要求与实际偏差 | [ChunkSize要求核对](Qchunksize/ChunkSize要求与自主探索偏差.md) |
| 仅自主探索的MTP四组要求、现有数值与缺项 | [四组覆盖核对](autonomous_20261012/MTP自主探索四组覆盖.md) |
| 最新MTP公开参考、位置率口径及OFF循环复现诊断 | [补充检索报告](autonomous_20261012/review_20261013/MTP一般情况检索与补充诊断.md) |
| 自主探索阶段结论、配置差异和数据索引 | [MTP与ChunkSize阶段总结](autonomous_20261012/自主实验最终总结.md) |
| MTP全部历史/有效新增记录（86条） | [MTP合表](Q1MTPExp/MTP全部实验汇总_含自主探索.md) |
| ChunkSize四档/五个场景并发组合及复测（34单元） | [ChunkSize实测表](Qchunksize/ChunkSize自主实验实测.md) |
| 本轮原始结果、网络依据、权重恢复、图表 | [自主实验目录](autonomous_20261012/README.md) |
| PD共置/EPLB实验 | [PD目录](pd_colloc_20261011/README.md)、[Q4分析产出](Q4Exp/README_分析产出.md) |
| 会议及要求 | `docs/lastmeet.txt`、`docs/AboutModels.md`、`docs/MTP.md`、`docs/ChunkSize_ref/` |

2026-10-11归档的旧内容仍在 `../history_exp/`，索引见 [历史实验](../history_exp/README.md)。

| 原内容 | 现位置 |
|---|---|
| EP64 统一服务 | `../history_exp/01_EP64统一服务_2026-09/` |
| PD 分离 2P2D | `../history_exp/02_PD分离2P2D_2026-09-20/` |
| Chunk Size 单机 | `../history_exp/03_ChunkSize单机_2026-09/` |
| MTP 推测解码 | `../history_exp/04_MTP推测解码_2026-09/` |
| EPLB 冗余专家 1P1D | `../history_exp/05_EPLB冗余专家_1P1D_2026-09-28/` |
| 四节点服务与权重核查 | `../history_exp/06_四节点服务与权重核查_2026-10-11/` |
| 分析报告 / 工具 | `../history_exp/90_文档归档/` 、 `../history_exp/91_工具/` |

机器授权和占用不能从历史服务记录推断。本轮只使用23；节点释放实查见 [节点释放核对](autonomous_20261012/节点释放核对.json)。新实验继续使用独立目录保留配置与原始数据。
