# 四节点服务拉起 + 权重问题核查（2026-10-11）

## 1. 结论

- **19 / 23 / 31 / 32 四台各起一个独立单机 TP16/EP16 服务，全部 `health=200`，真实中文问询全部正常，没有出现历史故障里的"重复怪符号"。**
- 每台 4 条提示词（`1加1等于几` / `Hello, world.` 翻译 / 自我介绍 / 水的化学式）**全部通过**，另有用程序自动检测"单字符连续重复 >=5、低字符多样性、重复子串"的退化检测，均未触发。详见 `chat_checks/`（逐请求原始 SSE）与 `semantic_battery.py`。
- 权重方面有三个必须记住的事实，见第 4 节。

## 2. 服务配置

| 项 | 值 |
|---|---|
| 拓扑 | 每节点独立单机 `TP16 / EP16 / DP1`，16 个 NPU die |
| 镜像 | `sglang-glm52-pd:fe694986` = `sha256:30b20a3c…9251d`（32 号机原先没有，本次从 23 `docker save/load` 分发） |
| 模型 | **`/data_lib/data/models/glm-q`**（nvme），四台机 size/mtime 指纹完全一致 |
| 容器名 | `bringup-glm52-<node>`，端口 `30080` |
| 关键参数 | mem-fraction-static 0.88、chunk 8192、context 131072、max-prefill 65536、max-running 4、DeepEP normal、**graph 关闭**、MTP 关闭、HCCL_BUFFSIZE 650 |
| 外挂 | 只读了 `load_model_utils.py`（加载 barrier 480s → 3600s） |
| 入口 | `serve.sh start <run_id> [nodes...]` / `status` / `stop` |

> 注意：`--cuda-graph-backend-decode full` 这套参数在本镜像的 DeepEP normal 路径下**会被自动禁用**，所以实际是无 graph 运行；这与 9/26–9/28 的结论一致。

## 3. 为什么不用原来的 /mnt/raid

`/mnt/raid` 是 **降级的 RAID5**：

```
md127 : active raid5 nvme1n1[1] nvme0n1[0] nvme2n1[2]
        11251817472 blocks super 1.2 level 5, 512k chunk [4/3] [UUU_]
```

4 盘阵列缺 1 盘，读盘只有 ~250 MiB/s（要靠校验重建），单机加载 721G 权重 >40 分钟；
`/data_lib`（nvme，非 raid）单机加载只要 **4–5 分钟**。这很可能就是历史上"同一文件两次读出不同内容"的根因：
RAID5 缺盘时靠校验重建，校验与数据不一致就会读出错误字节，且 mtime/inode 完全不变。

## 4. 权重现状（重要）

用 9/27 在 31 号机 `/mnt/raid/.../glm-qq` 上做的"参考快照"（23093 个 32MiB 块的 SHA256，扫描时 0 个 NaN/Inf）作为基准，今天实测：

| 副本 | 与参考的差异 |
|---|---|
| `/data_lib/.../glm-q`（四台机**内容一致**） | 182 个文件里 **116 个**与参考不同；334 个块需要修；扫描到 **30900** 个非有限值 |
| `/mnt/raid/.../glm-qq`（23） | 116/182 文件漂移（9/27 记录：60） |
| `/mnt/raid/.../glm-qq`（31，参考源头） | 有块读出的内容**稳定地不等于** 9/27 参考值 |

**关键点：**

1. **四个副本都不干净**，但 `/data_lib` 在四台机上字节一致（size/mtime 指纹相同、抽样坏块相同），
   所以四台机跑同一份字节，做 A/B 对照是成立的。
2. **9/27 的"黄金参考"已经部分丢失**：把 /data_lib 需要的 334 个参考块拿去 23、31 的 glm-qq 里找，
   合起来只覆盖 **329/333**；有 4 个块的原始字节在所有现存副本里都已找不到：
   - `quant_model_weights-00017-of-00181.safetensors` @ 536870912 / 603979776 / 3724541952
   - `quant_model_weights-00119-of-00181.safetensors` @ 3959422976
   旧 payload（`readonly_source23/25`）只能覆盖 28/333，救不回来。
3. 坏块落在 **routed MoE 专家权重**上（layer 13/16/40/43 的 `experts.N.{gate,up,down}_proj`），
   **不在** MTP 层（layer 78）或注意力矩阵（kv_b_proj / wk / weights_proj）上。
4. 尽管有这些损坏，四台机的真实对话输出都正常——说明损坏足够稀疏，没有打断常用路径。
   但这**不等于**所有 token 都正确，长上下文/高并发/特定专家命中时仍可能有偏差。

## 5. 已归档的证据

- `chat_checks/`：每次 `chat_once.py` 的 `request.json` / `events.jsonl` / `reply.txt` / `summary.json`
- `logs/<run_id>/node_*/launch.sh`：每台机的完整启动命令
- `weight_audit/make_blocks.log` + `snapshot_datalib/`：/data_lib 全量扫描（块 SHA + NaN 统计）
- `weight_audit/build_compact.log` + `readonly_source_data/fetch_stderr.log`：参考块获取失败的原始堆栈
  （`RuntimeError: reference SHA changed: .../00015-of-00181.safetensors 2483027968`）
- `semantic_battery.py`：可复跑的退化检测
- `/tmp/ok_{23,31,32}.json`：各节点能正确提供哪些参考块的清单（临时文件，可重跑生成）

## 6. 尚未做

- 参考校验外挂（read-only weight source）没有启用：因为参考源本身已损坏，无法把四台机修到"同一份已验证字节"。
  目前四台机靠 `/data_lib` 天然同字节。
- `/data_lib` 只抽了 364 个块做跨机比对；跨机全量逐块一致性还没有证明（metadata 指纹一致是强证据，但不是全量内容证明）。
