# Chunk Size 人工实验：逐步记录

日期：2026-09-27。实验与工作流按实际完成的步骤补充。

## 第一步：服务准备

23 主测；25 启动同参数的独立服务。每台均为单机 TP16 / EP16 / DP1，使用本机 16 个 NPU die。

| 配置 | 当前设置 |
|---|---|
| 模型 | GLM-5.2 ModelSlim W8A8；`/mnt/raid/user/data/models/glm-qq` |
| 镜像 | `sglang-glm52-pd:fe694986`；ID `sha256:30b20a3c29e8699be3323472deb3259d67ad6c53771824c2950a68902ac9251d` |
| Chunk / MTP / mixed chunk | 8192 / OFF / 默认 OFF |
| graph / mem-fraction-static | OFF / 0.88 |
| context / max-prefill / max-running | 131072 / 65536 / 4 |
| DeepEP / HCCL buffer | normal / 650 MB |
| 服务地址 | 23：`http://10.120.72.23:30080`；25：`http://10.120.72.25:30080` |

启动脚本：[serve.sh](serve.sh)。沿用上一轮模型加载 barrier 从 480 秒延至 3600 秒的单行补丁，副本放在 `patches/`。

本步运行标识：`20260927_154700`；实际启动时间以 `manifest.txt` 为准。

- 23 完整环境变量和启动命令：[launch.sh](results/20260927_154700/node_23/chunk_8192/launch.sh)。
- 25 完整环境变量和启动命令：[launch.sh](results/20260927_154700/node_25/chunk_8192/launch.sh)。
- 启动日志、模型配置哈希、KV 容量和健康记录保存在对应节点目录；替换前的自有服务日志保存在 `results/20260927_154700/before/`。
- 两台镜像 ID、模型 config 与量化描述哈希一致。启动后继续核对实际生效参数和容量。

### 操作命令

从本目录执行：

```bash
bash serve.sh start 23 8192 20260927_154700
bash serve.sh start 25 8192 20260927_154700
bash serve.sh status 23
bash serve.sh status 25
bash serve.sh sync 25 20260927_154700
```

`start` 要求该节点本轮容器与端口空闲；需切换 Chunk 时先 `bash serve.sh stop 23`，再以新运行标识启动。25 的目录是独立副本，使用 `sync` 将远端日志同步回 23 工作区。

### 本步结果

两台均已就绪，完整核对结果保存在各节点的 `readiness.json`、`health_ready.txt` 与 `npu_smi_ready.txt`。

| 节点 | 核对时间（北京时间） | 健康状态 | KV token 容量 | 16 个 NPU die 的 HBM 占用（MiB） |
|---|---|---|---:|---|
| 23 | 2026-09-27T16:00:02+08:00 | health_http=200 | 63744 | 61797–62104 / 65536 |
| 25 | 2026-09-27T16:00:01+08:00 | health_http=200 | 63616 | 61781–62098 / 65536 |

实际生效的 Chunk=8192、mixed chunk=False、MTP=None、graph 关闭、mem-fraction=0.88、TP16/EP16/nnodes=1 均已从启动日志核对。后续测量在 23 上逐步进行。

## 停电后复查（2026-09-27 19:13，北京时间）

两台 SSH、`/health` 与 `/metrics` 均正常，容器重启计数均为 0，未见 OOM 状态。参数保持单机 TP16/EP16、Chunk=8192、MTP OFF、mixed chunk OFF、graph OFF；无需恢复或重启。

| 节点 | KV token 容量 | NPU die | HBM 占用（MiB） |
|---|---:|---:|---|
| 23 | 63744 | 16 | 61797–62105 / 65536 |
| 25 | 63616 | 16 | 61781–62098 / 65536 |

复查记录：[checks/20260927_191312](checks/20260927_191312/)。本步仅检查服务状态，人工实验流程仍停留在服务准备。

## 第二步：真实文本回复检查

在 23 的终端运行，替换引号里的提示词：

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_manual_20260927/chat_once.py '你好，请用一句中文介绍你自己。'
```

也可以不带提示词参数，运行后输入：

```bash
python3 /home/chengyetai/workspace/current_exp/chunk_manual_20260927/chat_once.py
```

默认请求 23 的 `/v1/chat/completions`，模型名 `glm-5.2`，流式打印正文；temperature=0、max_tokens=256，按本地模型模板设置 `enable_thinking=false`。使用正常 EOS；需要更长回复时加 `--max-tokens 512`。这是单次语言检查，其耗时不作为性能实验结果。

每次请求保存到 `chat_checks/<时间>/`：`request.json`、原始 `events.jsonl`、`reply.txt` 和 `summary.json`。人工确认回复切题、中文可读、无异常重复，并检查结束原因；`length` 表示达到输出上限。脚本退出码只反映请求异常或空正文，非空回复的语义仍须人工判断。

### 首次检查结果：未通过

2026-09-27 19:17（北京时间），提示词为“请用中文回答：1加1等于几？只回复一句话。”。HTTP=200，但正文为重复的 `!`、`_cnt`，未回答问题；输出 64 token，结束原因为 `length`。原始 SSE 的 `delta.content` 也是这些异常内容，原因尚未定位。

原始记录：[chat_checks/20260927_191739_714869](chat_checks/20260927_191739_714869/)；人工判断：[review.json](chat_checks/20260927_191739_714869/review.json)。本步未通过，后续正式性能实验需先完成语言正确性检查。

## DeepEP 外挂对照（2026-09-27 22:05，北京时间）

用户指定的 `deepep_fix_pkg/deep_ep` 已完整复制到 [patches/deepep_fix_20260927](patches/deepep_fix_20260927/)，237 个文件逐文件 SHA256 一致。25 仅外挂此副本并重启，模型与其他启动参数保持一致；23 原服务继续运行。

25 新旧包对同一“1加1等于几”提示均返回 `!!!!!!!!!!!!!!!!`，HTTP=200、finish_reason=length，语言检查仍未通过。完整对照见 [deepep_ab/README.md](semantic_debug_20260927/deepep_ab/README.md)。此时 25 使用外挂 DeepEP；正式性能实验仍等待语言正确性恢复。

## 只读权重外挂修复：短文本检查已通过

本次记录跨 2026-09-27 至 2026-09-28。23、25 均已恢复正常中文回复；算术、自我介绍、翻译共六条请求均正常 `stop`。原始模型目录和公共镜像保持原样，运行时由工作区加载器提供通过逐块 SHA256 校验的参考权重字节。

23 运行标识 `20260928_000535`；25 为 `20260928_001720`。两台仍独立单机 TP16/EP16、Chunk=8192、MTP OFF、DeepEP normal、graph OFF、mixed chunk OFF。两台各 16 个 NPU die 继续占用。

完整说明和证据：[修复记录](semantic_debug_20260927/readonly_resume/README.md)。后续切换 Chunk 使用 [serve_verified.sh](serve_verified.sh)，保留权重外挂与参考读取服务检查；前文 `serve.sh` 原生启动命令是修复前的历史记录。

当前标准流程已完成服务准备与真实短文本检查。后续性能负载另行按人工流程组织。
