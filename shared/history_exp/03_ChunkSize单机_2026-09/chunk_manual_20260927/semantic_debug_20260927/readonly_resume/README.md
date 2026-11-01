# 中文回复异常：只读外挂修复记录

操作记录跨 2026-09-27 至 2026-09-28，以北京时间产物时间戳为准。

## 结果

23、25 的算术、中文自我介绍、翻译共六条短文本检查全部通过，HTTP=200，结束原因为 `stop`。两台均为独立单机 TP16 / EP16，Chunk=8192、MTP OFF、DeepEP normal、graph OFF、mixed chunk OFF、mem-fraction=0.88，继续占用各自 16 个 NPU die。

| 节点 | “1加1等于几” | Hello, world. 译文 | 自我介绍 | 正常结束 |
|---|---|---|---|---|
| 23 | 1加1等于2。 | 你好，世界。 | 可读、切题 | 3/3 stop |
| 25 | 1加1等于2。 | 你好，世界。 | 可读、切题 | 3/3 stop |

完整请求、原始 SSE、正文和助手语义判断：[semantic_checks.json](semantic_checks.json) 中记录各次目录，用户可复核。这里只检查短文本正确性，后续长上下文、Chunk 干扰和 MTP 仍按实验流程开展。

## 为什么处理权重读取

此前真实提示得到连续感叹号；只替换 DeepEP 包的对照仍未通过。只读流式扫描发现原副本读出了 NaN/Inf 和极端浮点值，31 的参考快照全部分片扫描通过。

| 流式快照 | safetensors 文件 | NaN/Inf 数 | 有限值绝对值 > 1e10 数 |
|---|---:|---:|---:|
| 23 原副本 | 182 | 130029 | 3429848 |
| 25 原副本 | 182 | 142706 | 4507587 |
| 31 参考副本 | 182 | 0 | 0 |

快照为当次读取结果：[snapshot_node23](../snapshot_node23/summary.json)、[snapshot_node25](../snapshot_node25/summary.json)、[snapshot_node31](../snapshot_node31/summary.json)。

还观察到同一 mtime 下读出的数据块摘要变化，加载时再次触发了校验拒绝。造成底层摘要波动的原因尚未确定；本次恢复由每块参考 SHA256 保证输入一致性，不将旧快照当作当前文件读出的保证。

## 外挂实现

- 保留原始模型目录和公共镜像；模型及参考差异数据都以只读方式挂载。
- 工作区保存参考副本的差异页：23 约 1.164 GiB，25 约 0.659 GiB。
- [加载器](../../patches/readonly_weight_source/README.md) 重建 32 MiB 数据块，校验参考 SHA256；重建不匹配时，经回环地址的 SSH 转发从 31 取完整块，验证后使用。
- 返回私有 CPU 张量中的参考原样字节，后续 ModelSlim / SGLang / NPU 推理沿用原流程。
- 31 的服务只提供启动时的权重块，使用 CPU；Prefill/Decode 仍在各自本机 NPU 上。
- 两台目标参考文件 SHA256 全部相同：[shared_reference_equal.json](shared_reference_equal.json)。23 的配置、量化描述、tokenizer、索引与模型代码也与参考副本一致：[model_metadata_comparison.json](model_metadata_comparison.json)。

两节点各 16 个加载进程，都读完全部 182 个分片、23093 个数据块；逐块通过参考 SHA256，无加载校验错误：[load_audit_summary.json](load_audit_summary.json)。关键输出头、词嵌入、旋转矩阵无非有限值，张量摘要一致：[23 检查](verified_key_tensors.json)、[25 检查](verified_key_tensors25.json)。

加载阶段优化了 Torch 导入时机、CPU 字节组装和验证块缓存。缓存最多每进程 512 块（16 GiB），适用于当前 2 TiB RAM 主机；该缓存用于模型加载。前缀 KV 缓存仍由 SGLang 管理。

当前两台使用镜像自带 DeepEP。用户指定的修复包副本保留在 `patches/deepep_fix_20260927`，先前对照结果单独存档。

## 当前服务

| 节点 | 运行标识 | health | HBM 占用（MiB / die） |
|---|---|---:|---|
| 23 | `20260928_000535` | 200 | 61798–62103 / 65536 |
| 25 | `20260928_001720` | 200 | 61780–62098 / 65536 |

完整生效参数、挂载及占用：[after_status.json](after_status.json)。

- [23 启动命令和日志](../../results/20260928_000535/node_23/chunk_8192/)。
- [25 启动命令和日志](../../results/20260928_001720/node_25/chunk_8192/)。

## 人工流程入口

在 23 发提示词：

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_manual_20260927/chat_once.py '你好，请用一句中文介绍你自己。'
```

向 25 发请求时加 `--url http://10.120.72.25:30080`。

以后切换 Chunk，从 23 终端使用 [serve_verified.sh](../../serve_verified.sh)，它先检查或恢复参考读取服务，再保留权重外挂启动。本实验容器已停止、端口空闲后：

```bash
bash /home/chengyetai/workspace/current_exp/chunk_manual_20260927/serve_verified.sh start 23 1024
bash /home/chengyetai/workspace/current_exp/chunk_manual_20260927/serve_verified.sh start 25 8192
```

停止单节点、检查状态、同步 25 日志的子命令保持 `stop` / `status` / `sync`。每次重启后重新核对参数、设备占用，并发送真实提示词检查正文和结束原因。
