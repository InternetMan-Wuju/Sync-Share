| 配置 | 镜像 | deepep | graph | MEM | health | KV池(token) | spec生效 | AL | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| A_v0516_deepep-auto_graph-on_mem0.88 | sglang:cann9.0.0-a3-v0.5.16 | auto | on | 0.88 | 000 | 29568 | err | - | AclNN_Inner_Error(EZ9999):  MoeLowLatencyCombineV2 do tiling failed |
| B_v0516_deepep-auto_graph-off_mem0.88 | sglang:cann9.0.0-a3-v0.5.16 | auto | off | 0.88 | 503 | 29568 | err | - | AclNN_Inner_Error(EZ9999):  MoeLowLatencyDispatchV2 do tiling faile |
| C_v0516_deepep-normal_graph-off_mem0.92 | sglang:cann9.0.0-a3-v0.5.16 | normal | off | 0.92 | 200 | 55296 | yes | 2.133 | ✅ 跑通 |
| D_v0516_deepep-normal_graph-off_mem0.88 | sglang:cann9.0.0-a3-v0.5.16 | normal | off | 0.88 | 200 | 29568 | yes | 2.133 | ✅ 跑通 |
