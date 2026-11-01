# MTP配置排列组合与AboutModels对照

核对依据是 [AboutModels.md](../docs/AboutModels.md)、实际 `launch.sh`、服务日志中的 `server_args`、KV分配/Graph状态，以及保存的短请求结果。配置清单保存了30份启动命令记录，包含重试和后续复用；核心A～H扫描为八组配置，不能把30次启动称为30种不同配置。全部为既有记录的只读核查，本次没有启动服务或补跑。

## 1. 文档中存在三个不同的参考配置

| 配置项 | AboutModels默认配置（§1） | 8K/1K榜单基线（§3.1 NoEP） | 32K/1K榜单基线（§3.2 NoDp） |
|---|---|---|---|
| 镜像 | v0.5.16 | 沿用默认v0.5.16 | 沿用默认v0.5.16 |
| 节点/NPU | 单机/16 | 单机/16 | 单机/16 |
| TP / DP / EP | **16 / 2 / 16** | **16 / 2 / 1** | **16 / 1 / 16** |
| DP Attention / LMHead | 开 / 开 | 开 / 开 | 关 / 关 |
| DeepEP | auto | **关闭** | auto |
| Decode Graph | 默认请求开启，bs=8/16/24/32 | 默认设置未列为改动；实际状态需日志核验 | 默认设置未列为改动；实际状态需日志核验 |
| MEM | 0.90 | 沿用默认0.90 | 沿用默认0.90 |
| Chunk Size | 16384 | **8192** | 16384 |
| Max Prefill Tokens | 32768 | 沿用默认32768 | 沿用默认32768 |
| Context Length | 180000 | 沿用默认180000 | 沿用默认180000 |
| Max Running Requests | 64 | 沿用默认64 | 沿用默认64 |
| Tokenizer Workers | 4 | 沿用默认4 | 沿用默认4 |
| MTP | 默认配置未启用 | 未给出启用MTP参数 | 未给出启用MTP参数 |
| 榜单客户端并发 | 无统一值 | **32** | **2** |

“沿用默认”表示榜单行未列出该项的修改，不是重新取得运行时证据。文档备注本身提醒Graph可能无法启用，因此不把默认开关直接当作已核验的实际Graph状态。

## 2. A～H八组主要扫描：实际改变了什么

八组共同使用23单机16卡，实际TP16/DP1/EP16、DP Attention/LMHead关闭；MTP都是NEXTN、steps3/topk1/draft4、草稿unquant。旧启动命令部分未显式写 `--dp-size`，服务日志确认其值为1。

| 配置项 | AboutModels默认 | A～H实际 | 核查结论 |
|---|---|---|---|
| TP / DP / EP | 16 / 2 / 16 | **16 / 1 / 16** | 更接近§3.2的并行方式 |
| DP Attention / LMHead | 开 / 开 | **关 / 关** | 八组均未比较DP开关的效果 |
| 镜像 | v0.5.16 | **v0.5.16 / v0.5.19 / v0.5.20** | 扫描变量 |
| DeepEP | auto | **auto / normal** | 扫描变量 |
| Graph请求参数 | 开 | **开 / 关** | 扫描变量；实际状态另列 |
| MEM | 0.90 | **0.88 / 0.92** | 扫描变量 |
| Chunk Size | 16384 | 16384 | 一致 |
| Max Prefill Tokens | 32768 | **65536** | 已改变，旧简表容易漏掉 |
| Context Length | 180000 | **131072** | 已改变 |
| Max Running Requests | 64 | **8** | 服务端上限已改变，不等于客户端并发 |
| Tokenizer Workers | 4 | 4 | 一致 |
| 模型来源 | `/mnt/raid/user/data/models/glm-q` | `/data_lib/data/models/glm-q` | 权重来源不能只按同一模型名认定等价 |
| MTP | 未启用 | **启用** | 本次待验证功能 |
| 测量请求 | 四组标准workload | 单条短探针 | 功能/兼容性验证 |

探针文件名含“64_32”，但检查数据和服务端meta后，**五个成功探针实际均为34输入/32输出、并发1**。旧阶段文档写“64输入”不准确。它们只能用于短请求功能确认。

## 3. 八组完整组合表

Graph“请求开启”表示传了Graph参数，“实际关闭”表示日志确认normal路径禁用了Graph。KV是服务日志中的分配结果。

| 编号 | 镜像 | DeepEP | Graph请求 | Graph实际状态 | MEM | KV池token | 短请求AL | 实测结果 |
|---|---|---|---|---|---:|---:|---:|---|
| A | v0.5.16 | auto | 开 | 捕获失败，未形成有效服务 | 0.88 | 29,568 | — | LowLatency **Combine** tiling错误 |
| B | v0.5.16 | auto | 关 | 关闭 | 0.88 | 29,568 | — | LowLatency **Dispatch** tiling错误 |
| C | v0.5.16 | normal | 关 | 关闭 | 0.92 | 55,296 | 2.133 | 34/32短请求完成 |
| D | v0.5.16 | normal | 关 | 关闭 | 0.88 | 29,568 | 2.133 | 34/32短请求完成 |
| E | v0.5.16 | normal | 开 | **被normal路径禁用** | 0.92 | 55,424 | 2.133 | 请求完成；Graph未实际开启 |
| F | v0.5.19 | normal | 开 | **被normal路径禁用** | 0.92 | 70,912 | 2.133 | 移除不兼容loader外挂后完成 |
| G | v0.5.20 | normal | 开 | **被normal路径禁用** | 0.92 | 59,904 | 2.133 | 移除不兼容loader外挂后完成 |
| H | v0.5.20 | auto | 开 | 捕获失败，未形成有效服务 | 0.92 | 59,904 | — | 移除loader外挂后仍有LowLatency **Combine**错误 |

F/G/H各存在首次loader接口导入错误的重试记录；上表采用移除旧loader外挂后的有效尝试。因此不能把首次ImportError解释成这三个镜像本身都不支持MTP。

八组扫描覆盖如下，空格代表没有该组合的主扫描记录：

| 镜像 | auto + Graph请求开 | auto + Graph请求关 | normal + Graph请求开 | normal + Graph请求关 |
|---|---|---|---|---|
| v0.5.16 | A：MEM0.88，失败 | B：MEM0.88，失败 | E：MEM0.92，完成 | C：MEM0.92、D：MEM0.88，完成 |
| v0.5.19 | 未测 | 未测 | F：MEM0.92，完成 | 未测 |
| v0.5.20 | H：MEM0.92，失败 | 未测 | G：MEM0.92，完成 | 未测 |

这是有针对性的组合扫描，未覆盖镜像×DeepEP×Graph×MEM的完整矩阵。成功normal组合的实际Graph均关闭，所以没有获得“同一配置实际Graph开/关”的性能对照。

## 4. 这些组合分别能回答什么

| 比较 | 保持相同的主要条件 | 变化 | 得到的证据 | 解释边界 |
|---|---|---|---|---|
| A→B | v0.5.16、auto、MEM0.88 | Graph请求开→关 | 关Graph后仍有Dispatch错误 | 故障不只出现在Graph捕获环节 |
| B→D | v0.5.16、Graph关、MEM0.88 | auto→normal | D完成短请求 | 当前单机DP1配置存在可运行normal路径 |
| D→C | v0.5.16、normal、Graph关 | MEM0.88→0.92 | KV 29,568→55,296 | 可用KV容量增大，未据此证明吞吐提高 |
| C→E | v0.5.16、normal、MEM0.92 | Graph请求关→开 | 两组实际均关闭，均完成请求 | 不能报告为Graph开启收益 |
| E→F→G | normal、Graph请求开、MEM0.92 | 镜像16→19→20 | 均完成请求；KV 55,424/70,912/59,904 | F/G还移除了旧loader外挂，镜像环境及加载方式需一起注明 |
| G→H | v0.5.20、Graph请求开、MEM0.92、无旧loader外挂 | normal→auto | H捕获失败、G请求完成 | 限定于本次该组合，未证明所有auto配置都失败 |

五个成功探针的AL都为32/15≈2.133。这是同一极短输入的功能结果，不能用于宣称正式workload接受率一致，也不能证明模型精度正确。当前权重异常另见 [问题定位报告](MTP问题定位_20261012.md)。

## 5. 八组以外，实际还尝试过哪些组合

### 5.1 接近文档默认并行配置的两次尝试

| 尝试 | 镜像 | TP/DP/EP | DP Attention/LMHead | DeepEP | Graph请求 | MEM | Context / MaxPrefill / MRR | KV | 结果 |
|---|---|---|---|---|---|---:|---|---:|---|
| 13:38启动 | **v0.5.18** | 16/2/16 | 开/开 | auto | 开 | 0.90 | 180000 / 32768 / 64 | 40,064 | LowLatency Combine错误 |
| 13:45启动 | **v0.5.18** | 16/2/16 | 开/开 | auto | 关 | 0.90 | 180000 / 32768 / 64 | 40,064 | LowLatency Dispatch错误 |

这两组并行、显存、context、max-prefill、MRR与默认配置接近，镜像为v0.5.18而默认为v0.5.16；模型也来自data_lib。不能称为默认环境只叠加MTP的完整复现。镜像标签由历史启动回执核验，出处在配置清单中保留。

### 5.2 EP1 / 关闭DeepEP的尝试

| 配置项 | §3.1 NoEP榜单 | 我们的ROUTE_A实际启动 |
|---|---|---|
| 镜像 | 默认v0.5.16 | **v0.5.19** |
| TP/DP/EP | 16/2/1 | 16/2/1 |
| DP Attention/LMHead | **开/开** | **关/关，启动命令遗漏了flags** |
| DeepEP | 关 | 关 |
| Graph | 默认请求开，实效待核验 | 请求开，未形成有效请求结果 |
| MEM | 默认0.90 | **0.92** |
| Chunk / MaxPrefill | 8192 / 默认32768 | **16384 / 65536** |
| Context / MRR | 默认180000 / 64 | **131072 / 8** |
| MTP | 未列启用 | 启用 |
| 结果 | 有榜单性能结果 | 未取得有效短请求/KV结果 |

这组先遇到loader接口错误，移除外挂后有启动记录，但仍未取得有效测量。服务日志明确DP2、两个DP开关False，并记录分别启动DP0/DP1；不能把这个不完整启动当作“正确NoEP配置也不支持MTP”的证据。后续启动脚本虽已修复DP flags拼接，本次没有补跑这条路线。

### 5.3 双机TP32/EP32尝试

三组均为23+20两节点、v0.5.19、TP32/DP1/EP32、DP Attention/LMHead关闭、DeepEP normal、Graph关闭、MEM0.88、chunk16384、max-prefill32768、context65536、MRR8、HCCL缓冲1024MB。

| 组别 | MTP | normal额外环境设置 | KV分配token | 结果 |
|---|---|---|---:|---|
| q1mtp2n_164815 | ON | 初始设置 | 299,264 | aclnnNotifyDispatch错误，health503 |
| q1mtp2n_170043 | ON | INT8通信及long-seq相关开关 | 299,264 | 仍有aclnnNotifyDispatch错误，health503 |
| q1mtp2n_off_170502 | OFF | 同上一组相关开关 | 315,264 | 仍有aclnnNotifyDispatch错误，health503 |

这里OFF也出现同类错误，因此不能将双机故障全部归因于MTP。KV已分配也不代表服务能够稳定执行请求；context仍限制为65536，不能凭这个KV值宣称已经支持100K输入。该表仅整理既有尝试，本次未重新操作20节点。

## 6. 后续负载使用了哪组配置

| 阶段 | 镜像 | TP/DP/EP | DP两开关 | DeepEP | Graph请求→实际 | MEM | MaxPrefill | MRR | KV |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 早期短对照 | v0.5.16 | 16/1/16 | 关 | normal | 关→关 | 0.88 | 65536 | 8，后改4 | 29,568 |
| 主扫描F及复用 | v0.5.19 | 16/1/16 | 关 | normal | 开→关 | 0.92 | 65536 | 8 | 70,912 |
| 后续一组负载尝试 | v0.5.19 | 16/1/16 | 关 | normal | 开→关 | 0.92 | **32768** | 8 | 71,040 |
| 逐位置补测、原生短测、完整重跑 | **v0.5.19** | **16/1/16** | **关** | **normal** | **开→关** | **0.90** | **32768** | **8** | **57,088** |

上述单机阶段均为chunk16384、context131072、tokenizer-workers4。最后一行是目前正式记录所用的服务配置；客户端并发由workload另行设定，8K/1K用4，另外三个场景用1。没有完成并发容量扫描。

## 7. 汇报时可直接使用的结论

“我们以AboutModels为参考，完成了八组主要配置验证，覆盖三个镜像版本、DeepEP auto/normal、Graph请求开关及两档显存比例。关掉Graph仍出现auto路径错误，改用normal后多个组合完成短请求；显存比例调整显著增加了KV容量。通过实际日志，我们确认normal路径关闭了Graph，避免把请求开关误报为已启用。

此外，我们尝试过DP2默认并行路线、EP1关闭DeepEP路线和双机TP32/EP32路线。其中EP1尝试存在DP flags遗漏，双机在MTP开关两侧都出现通信算子错误，尚未形成有效基线。后续固定负载选择了已可运行的v0.5.19、TP16/DP1/EP16、normal配置。

这阶段得到的是兼容性和容量证据。相对文档默认仍有并行、Graph、context、MRR及权重来源差异，性能收益需要在同一可信配置上单独测量。”

## 原始证据

- [配置清单JSON](causal_review_20261012/configuration_inventory.json)：启动命令SHA、请求参数、运行时DP开关、KV、Graph与错误类型、实际探针长度。
- [A～D汇总](results/sweep/summary.md)、[F～H汇总](results/sweep2/summary.md)。E另有独立日志和探针。
- [E启动命令](logs/E_v0516_normal_graph-on_mem0.92_151509/launch.sh)、[E短请求](results/sweep/E_v0516_normal_graph-on_mem0.92_151509.probe.json)。
- [F启动命令](logs/F_v0519_deepep-normal_graph-on_mem0.92_152553/launch.sh)、[G启动命令](logs/G_v0520_deepep-normal_graph-on_mem0.92_153127/launch.sh)、[H启动命令](logs/H_v0520_deepep-auto_graph-on_mem0.92_153700/launch.sh)。
- [接近默认配置的初次启动](logs/q1mtp_20261012_133851/launch.sh)、[关闭Graph重试](logs/q1mtp_20261012_134512/launch.sh)。
- [EP1尝试命令](logs/ROUTE_A_ep1_nodeepep_graph_161347/launch.sh)、[实际日志](logs/ROUTE_A_ep1_nodeepep_graph_161347/server.log)。
- 双机原件：`/data_lib/cyt/q1mtp2n/{q1mtp2n_164815,q1mtp2n_170043,q1mtp2n_off_170502}/node_23/`；主节点的命令和日志已纳入JSON清单。
- [当前完整重跑启动命令](results/native_full_20261012_190926/logs/native_full_20261012_190926/launch.sh)。
