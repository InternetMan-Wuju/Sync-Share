# 25 单机：仅替换 DeepEP 包的对照

日期：2026-09-27，北京时间。运行标识：`20260927_215657`。

## 操作

先保存 25 原容器信息、日志和原版回复，然后仅在 25 重启本实验容器，将工作区 DeepEP 包只读挂载到实际 Python 包路径。23 原服务继续运行。

- 原始包：[工作区副本](../../patches/deepep_fix_20260927/)。237 个文件在两节点均通过逐文件 SHA256 校验。
- 新容器导入：[import_fixed25.txt](import_fixed25.txt)。已确认实际导入新扩展，SHA256 为 `62f7915d337a4531d9a4ca7264398c68417f9c310c587af0687ac2cf0fdb0ae2`。
- 配置对照：[container_comparison.json](container_comparison.json)。新旧镜像 ID、启动命令、环境变量、设备映射相同；增加 DeepEP 只读挂载，日志目录随运行标识变化。
- 模型仍为 `/mnt/raid/user/data/models/glm-qq:/models/glm-q:ro`。单机 TP16/EP16，Chunk=8192，MTP OFF，DeepEP normal。
- 容器日志使用 UTC；本记录使用北京时间（UTC+8）。

## 真实文本结果

提示词：“请用中文回答：1加1等于几？只回复一句话。”；temperature=0、max_tokens=16、enable_thinking=false。两个请求体完全相同。

| 25 上的版本 | 请求时间 | HTTP | 回复 | 结束原因 | 人工判断 |
|---|---|---|---|---|---|
| 镜像自带 DeepEP | 21:45:42 | 200 | `!!!!!!!!!!!!!!!!` | length | 未通过 |
| 工作区外挂 DeepEP | 22:04:56 | 200 | `!!!!!!!!!!!!!!!!` | length | 未通过 |

原始请求及完整响应：

- [原版请求记录](../probes/20260927_214542_806281_node25_chat/)。
- [外挂版请求记录](../probes/20260927_220456_208815_node25_chat/)。
- [结构化对照结果](result.json)。

**结论：仅替换此 DeepEP 包，没有修复当前异常回复。** 导入、启动及这一条短请求均能完成；尚未测 hybrid、长上下文或并发性能，因此不能由本次结果判断这些场景下的修复效果。此前只读权重检查发现的非有限值仍需单独处理；本轮没有外挂权重源，没有修改模型文件。

## 当前状态和复现

23 使用原镜像 DeepEP；25 使用工作区外挂 DeepEP；两台服务地址均为原端口 30080。25 的启动命令与日志保存在 [本轮结果目录](../../results/20260927_215657/node_25/chunk_8192/)。

端口空闲、原实验容器已停止后，启用外挂：

```bash
DEEPEP_FIX=1 bash /home/chengyetai/workspace/current_exp/chunk_manual_20260927/serve.sh start 25 8192
```

恢复镜像自带包时，先停止本实验容器，再以新运行标识执行默认启动（`DEEPEP_FIX=0`）。这两种方式均只读加载原始模型。

## 后续服务状态更新

25 在运行 `20260928_001720` 中切回镜像自带 DeepEP，并启用只读参考权重外挂，真实中文检查通过。本页保留此前仅更换 DeepEP 包的历史对照；当前状态见 [权重修复记录](../readonly_resume/README.md)。
