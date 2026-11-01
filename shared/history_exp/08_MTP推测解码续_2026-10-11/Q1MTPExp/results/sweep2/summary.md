| 配置 | 镜像 | deepep | graph | MEM | health | KV池(token) | spec生效 | AL | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| F_v0519_deepep-normal_graph-on_mem0.92 | sglang:v0.5.19-cann9.0.0-a3 | normal | on | 0.92 | 200 | 70912 | yes | 2.133 | ✅ 跑通 |
| G_v0520_deepep-normal_graph-on_mem0.92 | sglang:v0.5.20-cann9.0.0-a3 | normal | on | 0.92 | 200 | 59904 | yes | 2.133 | ✅ 跑通 |
| H_v0520_deepep-auto_graph-on_mem0.92 | sglang:v0.5.20-cann9.0.0-a3 | auto | on | 0.92 | 000 | 59904 | err | - | ReportCallError][FILE:log_inner.cpp][LINE:148] |
