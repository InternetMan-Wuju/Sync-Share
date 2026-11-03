# 昇腾910C：当前Prefill路由驱动的专家复制与配额分流实现方案

日期：2026-11-03

对象：GLM-5.2量化版、SGLang昇腾后端；以服务器实际导入代码为准。

状态：实现设计与源码调研，尚未实现此补丁，未新增NPU实验。

本轮交付：服务器AI完成代码、CPU逻辑验证和待执行的NPU验证入口。用户此前暂停实验，因此不自动启动NPU实验、重启模型服务或续跑旧执行卡。

## 1. 要解决什么问题

用户提出：在Prefill当前层得到routing后，按专家收到的token数量排序，只挑超过平均数量2倍的专家，把它复制到冷卡，分走一部分token。

这里一个token可以被top-k选中送到多个不同专家。本文的“某专家收到多少token”，指发给该专家的计算任务数；全体专家任务数通常是有效token数乘以k。我们只改变某项任务由哪个物理副本执行，不改变模型原本选择的逻辑专家或路由权重。

改进后的规则：

1. 超过2倍平均值只用于筛候选，不能单独决定复制。
2. 目的卡按整张卡的预计完成时间选择；卡上有一个冷专家，不代表整张卡空闲。
3. 优先使用已经就绪、身份正确的副本；没有副本时才考虑复制。
4. 计算挪走多少token后源卡和目的卡分别要多久，再扣除统计、规划、复制、格式准备和额外通信时间。
5. 只有本次Prefill层调用的预计净收益超过误差余量，才执行。首版不靠未来decode收益或假设的重叠“摊薄”成本。

**需要交付完整流程：当前逻辑top-k → 统计 → 规划 → 必要时复制 → 精确分流 → 原量化专家计算 → 按原权重合并。**

上一轮已证明局部计算存在改善空间，但没有证明在线复制值得做。960/64到512/512案例的局部计算时间代理从315.766降到247.821微秒，差67.945微秒；同卡复制74.985微秒、简单CPU规划1.039微秒来自分项测量，不能把相减结果当作完整服务的实测收益。该案例也不能提供跨卡HCCL复制成本。[L1]

## 2. 已知环境与必须现场核实的事实

| 项目 | 仓库已有证据 | 实现前必须核实 |
|---|---|---|
| 设备 | 昇腾910C；当前只确定有一卡可用 | 可见物理设备、驱动、实际可用显存 |
| 推理栈 | SGLang 0.5.18；镜像v0.5.18-cann9.0.0-a3；torch 2.10.0+cpu、torch_npu 2.10.0、CANN 9.0.0 | 运行中容器的镜像摘要、实际包版本、源码路径及哈希 |
| 模型 | 探针记录model_type=glm_moe_dsa，架构GlmMoeDsaForCausalLM | 实际导入的模型类、MoE类、top-k和dispatcher；不能直接认定使用归档glm4_moe.py |
| 量化 | modelslim；专家运行态涉及权重、scale等 | 目标层全部运行态张量、格式、量化参数、预打包缓存 |
| 既有多卡配置 | TP16、EP16、DP1，DeepEP auto；曾开启16个额外物理槽 | 专家自身的TP分片数、当前物理槽数、逻辑→物理布局 |
| 当前任务 | 单层优先，目标layer_id=49 | 是零基编号49还是日志标签；记录真实模块路径，不凭编号猜 |
| 历史权重问题 | 旧E0依赖经过审计的恢复/加载流程 | 复用已验证的只读权重来源，不能绕过校验直接重读有坏块的目录 |

证据见[L2–L5]。服务器不能仅凭“版本号相同”认为源码相同。公共main/master源码用于识别接口；真正补丁必须针对服务器实际导入文件生成。官方安装文档要求匹配的依赖组合，本任务不自动升级任何依赖。[S1]

## 3. 在昇腾上走哪条实现路线

### 3.1 首选复用现有Prefill normal通信路径

当前SGLang NPU top-k实现存在“逻辑专家ID转换为物理专家ID”的边界；normal dispatcher再根据这些物理ID生成dispatch布局。应在逻辑top-k已确定、物理映射尚未完成的位置加入控制器。[S2][S3]

流程如下：

    gate/top-k：原逻辑专家ID、原路由权重
        ↓ 只统计有效路由分支，排除padding和shared额外槽
    各输入来源汇总计数 → 统一计划
        ↓ 若需复制，完成量化状态传输、格式准备及全组就绪
    按配额把每个逻辑分支映射到唯一物理槽
        ↓ 从更新后的物理ID重新生成dispatch布局
    原normal dispatch → 原量化MoE runner → 同一handle对应的combine

保留原top-k权重、token行号、top-k列号；不能复制一个路由分支后让两个副本都算并重复相加。自定义映射之后不能再调用一次原随机dynamic映射。

昇腾sgl-kernel-npu的deep_ep区分normal和low_latency策略：normal默认路径、normal alltoall参考路径、low_latency ops路径不是一个东西。官方SGLang文档也将normal与Prefill关联。必须记录此次实际命中的策略类。[S4][S5]

优先接入当前已经工作的normal路径。如果当前版本不能接受固定额外槽布局，再检查同版本alltoall参考实现；只有接口和正确性验证通过才能选用，且对照组必须使用相同后端。不能为了本策略全局切换decode后端：仓库E3的alltoall low-latency结果明显更慢，ops组合还出现过错误。[L2]

### 3.2 三个容易走错的接口

- torch_npu.npu_moe_distribute_dispatch_v2中的copy_expert_num表示恒等特殊专家，不是“复制多少份专家权重”；expert_scales是路由权重，不是分流配额；elastic_info在所查文档中是预留字段。该接口不能直接替我们实现当前算法。[S6]
- SGLang现有dynamic映射可选副本，但不等于本方案的全局精确token配额。当前公共代码的lp路径还有CUDA限制，不应直接当成910C可用实现。[S3]
- deepseek-ai/DeepEP-Ascend与sgl-kernel-npu提供的deep_ep不是同一个项目。前者所查README验证环境为950DT/CANN9.2，且权重预取通信能力尚有未实现项，不能作为本机910C现成的在线复制能力。[S7]

## 4. 算法合同：什么条件下复制、分多少

### 4.1 输入与候选

对本层、本次Prefill chunk，令n[e]为逻辑专家e接收的有效token任务数，E为路由专家总数。平均值必须是sum(n)/E，不能只对活跃专家取平均。候选要求n[e] > 2×sum(n)/E。

首版候选上限建议：最多2个热专家，每个专家最多3张有空闲或可安全替换副本槽的接收卡，一次层调用最多执行1个动作。它们是为了控制规划开销的初始工程参数，并非最优性结论。

首版每个逻辑专家固定一个主位置；最多一个额外副本用于本次分流。使用已有副本但配额为0时，该专家全部任务回主位置。禁止与另一套周期EPLB同时修改布局。若启动时已经有其他布局，先明确规范化后的主位置与权重身份。

### 4.2 配额候选

令L[i]、L[j]为分流前源计算卡i、目的卡j的全部token任务数。粗略均衡点：

    q0 = min(n[e], max(0, floor((L[i] - L[j]) / 2)))
    Q = 去重后的 {0, floor(q0 / 2), q0}

这是线性近似下的候选生成器，用来减少搜索；最终不能只凭它决定。某张卡的多个专家分别收到多少token会影响真实耗时，需用完整专家槽向量评估。首版不声称此三点搜索一定找到最佳配额。

对于已在目的卡缓存的正确副本，同样比较q=0和正配额；“已经复制过”不意味着一定要用。

### 4.3 时间判定

F[r]表示卡r在完整专家槽token向量下的局部计算耗时估计。它应来自目标量化runner的成本表；没有实测支持的区域标记为unknown，不能随意外推。

    G = max_r F_old[r] - max_r F_new[r] - P - C_prepare - D_extra

- P：本方案新增的有效token统计、同步、控制通信、规划、配额映射和提交开销。只测纯CPU排序不够。
- C_prepare：新副本从权重状态读取、必要的ND staging、跨卡传输、格式转换/预打包到就绪同步的完整成本。缓存命中且无需准备时才为0。
- D_extra：相同后端下，分流引起的dispatch/combine额外成本；若某项已包含在F里则不得再次扣除。
- epsilon：为估计误差保留的余量。通过成本预测误差和重复计时波动确定；未校准时active模式必须拒绝启用，不能把epsilon=0作为默认。
- 只在G > epsilon时选择正配额；否则无动作。应记录无动作路径本身的P，避免只汇报成功候选。

该式是首版的保守预测规则，不是性能保证：并发通信、排队和集合通信关键路径最终必须用完整层计时验证。未知P、跨卡C_prepare或D_extra时只允许shadow记录，不能把旧同卡74.985微秒冒充跨卡成本。[L1]

新复制的全部成本由当前层调用承担；暖缓存与冷复制分别报告。若后续探索异步预取，必须另做正确性和可重叠窗口验证，本版不预先减掉重叠成本。

## 5. 应实现的模块与接入点

下列名称是建议的新接口，不是声称SGLang已有这些类。

| 模块 | 责任 | 接入/输出 |
|---|---|---|
| PrefillReplicaController | 检查本次调用是否适用，协调全组步骤与版本 | 逻辑top-k后、物理映射前；返回唯一映射结果或原布局 |
| RoutingCountCollector | 统计各输入来源的逻辑专家任务数，排除padding和重复输入 | 全局计数与各来源计数、输入所有权说明 |
| CopyAwarePlanner | 生成有限候选，算配额和保守收益 | 一个统一Plan或no-op；记录各成本来源 |
| ReplicaSlotManager / StateTransport | 管理槽身份、完整量化状态复制、格式准备、生命周期 | 可被dispatch安全读取的READY槽和epoch |
| QuotaRemapper | 根据全局配额稳定地选择各路由分支的物理位置 | 与原top-k同形状的物理ID，原权重保持不变 |

服务器需要在IMPLEMENTATION.md逐项列出实际文件、函数、关键调用关系及补丁位置。优先检查：

- 当前版本NPU top-k中的finalize / logical_to_physical边界；旧版路径可能在layers/moe/topk.py，不能只按公共main路径打补丁。[S2][S3]
- 实际模型类的普通Prefill forward以及重叠执行入口；归档glm4_moe_v0518.py的forward_deepep、op_select_experts只是定位参考，实际模型类型不同。[L6]
- token_dispatcher/deepep.py中get_dispatch_layout与dispatch之前的物理ID。
- 原量化runner实际读取的参数、buffer和预打包缓存。
- expert_location_updater只复用经过核验的格式处理思想；不要直接每步调用旧全模型更新器。旧代码涉及全层缓冲、empty_cache及通信组假设，不适合直接放入每次Prefill。

控制协议必须从所有EP成员都会到达的MoE调度边界进入，保证同一step/layer的参与次数一致。不能只把collective放进fused_topk_npu的非空分支：实际模型可能对零token直接返回empty_topk_output或跳过top-k。服务器须补齐这些分支，或将控制协议提到共同调用点；逻辑路由结果仍在物理映射前交给控制器。

统一Plan至少包含step_id、layer_id、epoch、action、logical_expert、weight_source_rank、primary_compute_rank、destination_rank、destination_local_slot、q_global、q_by_input_source、copy_required、布局/权重版本、成本估计与来源。权重发送者与产生token的输入来源是不同概念，不能混用。

## 6. 精确token配额：多卡不能各自分走q个

计数之前先明确“有效token由哪个输入来源负责”。TP路径中可能存在重复输入，DP路径中又可能是不同输入；不能不看实际图就all_reduce，也不能简单除以TP16。不能根据hidden值相同来去重。

计数、前缀配额和实际dispatch必须针对同一组路由分支。不能统计时去掉TP重复输入，却仍让所有重复来源各自发送完整token。首版先核实normal dispatcher实际接收的是唯一分片；如果需要分片，应复用框架现有输入所有权，统计及remap也使用相同索引。尚未建立这一对应关系的部署不允许active。

每个有效路由分支必须只计一次，满足：

    sum_e n[e] = 本次唯一有效token数 × 每token有效路由专家数

如部分位置有无效专家ID，应以有效分支数为准。保存源序列/分片所有权、padding mask和计数守恒证据；shared额外槽单独处理。没有本地token的rank也必须参加要求的控制通信。

设输入来源s给专家e的任务数为c[s,e]，固定来源顺序，前缀：

    offset[s,e] = sum_{u < s} c[u,e]

该来源内按token行号及top-k列号稳定编号j。满足offset[s,e]+j < q的任务送副本，其余送主位置。

例：两个来源分别有6和4个任务，全局需要分走7个；实际应分别分走6和1个。若两个来源都执行“取前7个”，会错误分走10个。

验证必须保留(token行, top-k列, 逻辑专家, 原权重, 物理目标)对应关系。只证明总数相同，无法排除把token发给了错误专家。历史recorder在物理映射后采集的计数，不能直接充当这里的逻辑热度。[L6]

## 7. 副本槽与布局：先保证权重、ID和通信一致

首版使用启动时固定的物理槽容量；不在层执行途中改变num_experts或通信buffer大小。

对EP组内大小M、每rank固定S个物理槽：

    physical_id = ep_rank × S + local_slot
    num_physical_experts = M × S

此公式必须与实际后端布局核实，Plan中同时保存组内rank与全局rank映射。逻辑专家数与物理槽数分别传递，不能把扩容后的物理数误传成router输出维度。

每个目标层、每rank最多预留一个本策略副本槽；没有安全槽时不复制，不覆盖仍需计算的主专家。若框架全层统一配置，实际分配层数按第9节报告；首版不实现跨层共享槽。空槽/占位身份永远没有路由配额。加载器、量化参数数组、dispatcher和runner必须使用同一布局。

仓库S0已出现过“把物理256/257误以为在两张卡上”的问题：272槽/EP16意味着每rank17槽，256和257都在rank15。[L3] 不能把历史逻辑ID直接当成扩容后的本地主专家槽。

即使仅Prefill layer49启用策略，decode及无动作路径也必须使用扩容后正确的主位置映射。off对照不是“仍使用旧物理ID”；它是相同合法槽布局下不执行新策略。额外槽配置与原始无额外槽配置的显存和性能差异另行说明。

## 8. 910C复制必须处理完整量化状态和就绪协议

### 8.1 复制什么

针对实际运行态生成manifest：张量名、语义、shape、dtype、device format、物理storage大小、量化布局、TP分片、权重版本，以及runner是否还依赖其他缓存。

历史S1中单专家checkpoint是三个投影的weight/scale/offset共9个张量，合计37,830,656字节。[L4] 旧P2/Online计时链的准备后状态则为37,769,216字节，属于另一套运行态打包/计时证据。[L1][L7] 二者都不能硬编码成所有运行态副本的大小；打包后张量集合和scale dtype会变化，此前仅少数专家的offset为0也不能推广到全部专家。

复制必须包含运行时有效的所有专家状态。不能只复制weight而保留旧scale，不能在换专家后继续复用旧预打包缓存。共享层参数只在确认与expert无关后排除。

### 8.2 首版选择可审计的ND传输

昇腾NZ格式切片不应被当成普通连续字节；旧补丁已有NZ→ND发送、ND接收再恢复格式的处理。官方npu_format_cast提供ND/FRACTAL_NZ转换，但返回新张量，格式准备和额外内存必须计入。[S8][L6]

建议首版：核实专家切片 → 准备ND发送缓冲 → 接收ND缓冲 → 恢复目标runner需要的格式/缓存 → 检查全部状态就绪。优先预分配缓冲，不在热路径做empty_cache或全模型暂存。若实际版本证明原生格式可以安全直接传输，可另列替代实现，但必须先验证内容和计时。

格式转换返回的新Tensor必须实际写入或安装到runner读取的slot存储，并刷新相关缓存；仅生成临时NZ张量不能发布READY。如果融合NZ专家bank不支持安全的独立slot更新，必须报告为实现障碍。若采用整bank重新打包，需要另报耗时、峰值显存及引用更新方式，不能仍按单专家复制成本计算。

通信先采用固定manifest顺序的匹配发送/接收调用，或经过验证的单个ND打包缓冲。所查HCCL批量接口对同一通信对同方向的消息块有约束，不能未经核实就把多张量同方向传输塞进一个batch。[S9] dtype、元素数、零长度边界也须按本版本验证。

### 8.3 槽生命周期与发布顺序

    EMPTY或READY(old)
      → DRAINING：确认旧读者全部结束
      → FILLING：复制完整状态
      → PREPARING：完成格式转换和缓存准备
      → READY(new, epoch)：所有必要设备工作已完成
      → 全EP组提交相同epoch后，允许新dispatch引用

缓存键至少包括layer、logical_expert、expert-TP shard、量化布局、权重版本。只凭“槽里上次是专家43”不足以判断可复用。

首版同步执行复制和格式准备，之后才计算。Work.wait、stream wait_event等接口的语义要核实：建立后续设备工作的流依赖，不等于CPU此刻已经观察到设备全部完成。初版采用明确的完成同步后再发布ready，并把同步时间计入成本。[S10]

替换时禁止复用仍被旧dispatch读取的buffer；发送源、ND暂存和接收buffer也须存活到通信完成。复制失败时不能发布部分更新的槽身份。

### 8.4 通信组与故障处理

显式使用对应MoE EP组或预先建立的同成员复制组，不使用隐式world默认组。源/目的peer的全局编号与组内编号必须按当前API转换。默认new_group要求主world的所有进程，包括非该组成员，按相同顺序调用；不能只让EP子组成员调用。若采用特殊local synchronization模式，必须另核实其限制，不作为首版默认。不要按某次热专家临时随意建组。[S11]

若使用batch_isend_irecv，还须满足其首次通信的全组参与契约；按安装版本安排显式的全成员通信初始化/预热，再使用仅源/目的参与的已初始化路径。不能把其他通信组已经初始化当作这个复制组已就绪。组初始化与预热放在验证入口或服务启动阶段，不按token热度临时触发。[S10][S11]

全组以相同步骤执行：资格判断 → 计数 → 统一计划 → 资源准备 → 必要的复制与就绪 → 提交版本 → dispatch → combine。no-op和空输入路径也遵守协议。

复制开始前发现显存不足、成本未知或条件不支持，可以全组一致回退。通信已失败或状态不一致后不能让某rank自行回退继续跑；应终止本次验证/请求，报告故障并清理，防止其他rank卡死或使用半更新权重。初版不承诺通信故障后的服务无损恢复。

## 9. 显存是前置约束，不能只算一份专家

历史配置的额外槽选项可能对全部MoE层生效。若75层每层每卡多一份37,769,216字节状态，仅专家副本就约2.83GB/卡，还未包含ND收发暂存、格式转换峰值和通信工作区。

**策略只开layer49，不等于只给layer49分配显存。** 服务器必须报告实际分配范围及峰值预算。

优先在框架支持的情况下为目标层配置额外槽；若后端只支持全层统一额外槽，就明确全层成本，不偷偷声称单层开销。若layer-local需要大改，首轮先交付限制及可应用补丁边界，不自动做跨层共享buffer设计。

layer-local必须检查DeepEP buffer是否跨层共享、是否缓存num_experts/容量，及各层runner的布局约定。不能只扩layer49权重数组就认定通信兼容；不兼容时选择经核验的全层统一容量并报告显存，或提交明确限制，不在执行中动态更换共享buffer规格。

保留原KV缓存预算。若为了容纳副本缩小KV容量，必须单独报告；不能把改变服务容量后的性能直接归因于负载均衡。原E0在mem-fraction=0.90有过OOM，0.88时剩余显存约3.2–3.5GB，说明额外槽并非免费。[L2]

## 10. 开关与明确支持范围

建议新增配置；命名可随实际框架规范调整，报告中必须区分新增选项与上游原生选项：

- mode：off / shadow / active，默认off。off不做新增计数和通信；shadow做计划但不复制、不改变分流。
- target_layers：首版只选已确认的layer49。
- hot_factor=2、max_hot_experts=2、max_receivers=3、max_actions=1。
- replica_slots_per_rank=1；槽容量属于启动配置，不是运行时临时扩容。
- cost_table及min_gain_us：active必须有匹配设备、量化、后端和拓扑的有效标定来源。
- debug_trace：受控记录分流证据，性能计时关闭大规模逐token日志。

| 情况 | 首版行为 |
|---|---|
| 目标层、纯Prefill、eager、DP1、expert-TP=1 | 进入控制器，其他条件满足才允许active |
| expert-TP不为1 | 整组回退；不能仅凭总TP=16判断专家张量无需分片 |
| decode、混合batch、draft/verify或未支持的重叠路径 | 集体一致回退原主位置映射 |
| graph路径 | 不在图内做动态复制/CPU计划；首版仅支持明确的eager目标调用 |
| 无热专家、目的槽不安全、成本未知 | no-op或shadow，记录原因 |
| 周期EPLB正在改布局 | 不允许同时启用，避免两个管理器竞争权重与映射 |

禁用功能的回归需与相同槽布局的基线比较；相同槽布局的shadow与active对比用于区分观察成本和实际动作成本。不能把shadow称作没有overhead。

## 11. 分阶段交付和验收

### M0：本次服务器任务——实际补丁与CPU逻辑验证

首轮上限3小时。先完成实际环境与调用路径定位、默认off的补丁、计数/规划/配额/槽状态模块及CPU测试。能写出的NPU验证入口一并交付，未验证的硬件路径明确标记。

本轮不为补齐manifest而加载模型或新建NPU状态。没有已保存运行态证据的张量格式、storage、缓存及HBM峰值，先通过源码与历史记录审查并标记“待设备验证”，不能填成现场实测。

必须测试有实际故障含义的性质：

1. 无效ID、padding、shared槽和空输入被正确处理；全局有效分支守恒。
2. 多来源6+4、配额7得到6+1；q=0、q=n及不同来源排列的总配额准确。
3. 分流前后每个token仍对应原逻辑专家与原权重，只有物理位置改变。
4. cached与cold动作的成本区别、unknown成本不触发active、无收益不复制。
5. 槽不能在旧读者未完成或新状态未就绪时发布；epoch/身份不匹配被拒绝。
6. 当前routing改变后Plan随之改变，执行层真正消费该Plan；不能硬编码旧448配额。
7. 模拟多rank控制状态时，no-op、空输入和不支持路径不产生协议分歧。

CPU测试通过只证明这些逻辑，不证明HCCL可用或NPU性能改善。无硬件验证时报告“实现完成/待设备验证”，不能写“功能已在910C验证”。

交付针对实际源码的可审阅patch、基础文件哈希、应用检查与撤销方式。不要覆盖整份安装包，不把公共main代码硬换进0.5.18环境。若完整接线超过3小时，提交可审阅部分和明确缺口，不延长为自动续跑。

### M1：用户恢复实验后——单NPU完整串行功能链

此阶段是待执行方案，本次不启动。

为打破“未知成本不能active、却需要执行来测成本”的循环，独立验证入口允许显式指定合法测试Plan，或注入标为test-only的成本表，驱动与补丁相同的实际执行器。它只用于正确性验证和标定，不是在线策略收益证据；测试配置禁止进入服务active模式。下面M2同样遵守此规则，完成真实标定后才允许相应部署的服务active。

使用与实际补丁相同的控制器、状态复制和配额模块，在一张NPU上串行表示两个虚拟rank。输入可先用明确标注的合成top-k；不能称为真实GLM routing。完整执行输入→Plan→复制→分流→原量化专家计算→按原权重还原。

覆盖单专家、原专家与复制专家混合bank、完整top-8结构、空目的卡、冷复制、暖缓存、热点变化与换槽。目的槽初始放不同专家或受控哨兵，确保“忘了复制”会被测试发现。使用既有模型精度要求的误差阈值，对齐baseline输出；不能只看结果有限或单专家自一致。

记录真实复制字节、复制计数、Plan身份、各阶段时间和完整串行总时间。单NPU串行总时间是串行功能测试结果；把两个虚拟rank时间取max仍只是多卡计算代理，不能叫双卡实测加速。

### M2：资源到位后——真实跨卡层级验证

需要至少两个独立物理NPU，且满足所选后端的最小拓扑要求。不能用同一NPU上的两个进程代替跨卡。无需先装下完整GLM模型，可用目标层运行态专家和同一后端构成层级测试。

先验证量化状态跨卡传输与内容，再验证真实dispatch、专家计算、combine完整路径。比较相同槽布局、相同后端、相同输入的off/shadow/active；分别报告首次冷复制和后续暖缓存，报告无动作调用的额外时间。

完整计时应覆盖：统计与控制、CPU计划及同步、源端准备、HCCL传输、目标格式准备、ready提交、dispatch、计算、combine。用设备完成时间界定边界；至少报告重复次数、分布和计时方法。收益以完整层调用时间差为准，分项用于解释，不能把分项中位数之和冒充总耗时。

只有此阶段的数据才能标定该拓扑下的跨卡C_prepare和D_extra。评估成本模型预测误差后再启用相应active门槛。

### M3：完整模型资源恢复后——真实Prefill接线

在已确认的实际模型类和目标层启用，采集真实当前chunk路由。检查与禁用策略时输出的误差、逐层时间、Prefill吞吐、TTFT和显存。保留decode回退路径，确认没有被额外布局改坏。

先layer49，再决定是否推广到其他层。不能把单层收益乘75推算整模型收益；既有profile显示其他算子占有大量时间。[L2] 同时报告输入长度、batch、chunk大小、KV预算、策略触发比例及冷/暖比例。

## 12. 服务器应提交什么

建议目录：shared/current_exp/20261103_prefill_copy_aware/<run_id>/。

- IMPLEMENTATION.md：实际模型/框架路径、接入函数、数据流程、支持范围、未完成点。
- ENVIRONMENT.json：版本、镜像摘要、源码哈希、逻辑/物理布局、expert-TP、量化状态manifest。
- patches/：最小实现补丁、基础哈希、应用检查和撤销说明。
- tests/：CPU逻辑验证与独立NPU验证入口；代码必须导入实际实现，避免复制另一套“测试专用算法”。
- VALIDATION.md：实际执行的命令、结果及未执行原因；清楚分级CPU实算/NPU实测/估计/未测。
- PLAN_EXAMPLES.json：输入计数、候选、成本来源、最终配额、实际消费路径；无NPU执行时不得虚构实测时间。
- 若恢复实验再追加TIMINGS.csv、状态/分流校验摘要与MANIFEST哈希；本轮不制造空白“已完成实验报告”。

完成后按仓库约定提交并推送，不强推，不覆盖他人的新提交。报告commit、实际改动、测试状态；达到首轮边界就停止。

## 13. 可直接发给服务器AI的提示词

> 请拉取Sync-Share最新main，阅读handoff/2026-11-03_ascend_prefill_copy_aware_implementation.md，并参考handoff/2026-11-03_online_decode_review.md。请按新方案实现“当前Prefill路由驱动、考虑复制成本的专家副本与精确token配额分流”。
>
> 本轮只执行M0：检查服务器实际SGLang/torch_npu/sgl-kernel-npu版本和导入路径，针对实际代码制作默认off的最小补丁，并完成CPU逻辑测试。首轮最多3小时；不自动升级依赖、不启动NPU实验、不重启模型服务、不续跑旧实验卡。资源现在只确定有一卡；多卡结果不得模拟成实测。
>
> 核心要求是当前逻辑top-k产生的Plan真正驱动状态准备与物理ID映射，再进入原dispatch、量化runner及combine。保持原逻辑专家和路由权重；明确唯一token输入来源，精确实现全局配额；使用固定且权重匹配的物理槽。复制需包含完整量化运行态，处理ND/NZ、显式EP通信组、槽生命周期和ready后发布。不能把copy_expert_num当权重副本接口，也不能硬编码旧448配额。
>
> 优先支持目标layer49、纯Prefill、eager、DP1、expert-TP=1。未知成本只能shadow/no-op。先报告额外槽是否影响全部层及其显存成本。把单NPU完整串行链、真实跨卡完整层和整模型验证入口作为待执行交付，清楚标记未测。
>
> 将补丁、环境及源码哈希、应用/撤销方式、CPU测试和验证入口放到shared/current_exp/20261103_prefill_copy_aware/<run_id>/，提交并推送。不要只交计算/复制分项脚本。若3小时内未完整接通，准确提交已实现部分和缺口，停止并等待用户；不得把代码完成写成910C验证通过。

## 14. 调研来源及使用边界

以下公网源码于2026-11-03查阅。main/master会变化，服务器须把实际依据的commit或文件哈希写进ENVIRONMENT；这些链接不证明当前镜像包含同样实现。本方案的候选上限、成本门槛、生命周期及阶段划分是工程设计建议。

- [S1] [SGLang昇腾安装与兼容组合](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx)。
- [S2] [SGLang NPU top-k与最终映射边界](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/npu/moe/topk.py)。
- [S3] [SGLang逻辑到物理专家映射](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/eplb/expert_location_dispatch.py)；[DeepEP token dispatcher](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)。
- [S4] [sgl-kernel-npu的deep_ep后端说明](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)；[normal策略及alltoall参考路径](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/deep_ep/strategies/normal_strategy.py)。
- [S5] [SGLang专家并行与通信模式](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)。
- [S6] [Ascend op-plugin：npu_moe_distribute_dispatch_v2参数定义](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_dispatch_v2.md)。
- [S7] [DeepEP-Ascend官方仓库：验证环境及实现状态](https://github.com/deepseek-ai/DeepEP-Ascend)。
- [S8] [昇腾npu_format_cast文档](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/torchnpuCustomapi/docs/en/custom_APIs/torch_npu/%28beta%29torch_npu-npu_format_cast.md)。
- [S9] [HCCL集合通信库用户指南](https://cann-doc.obs.cn-north-4.myhuaweicloud.com/hccl/HCCL%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E5%BA%93%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97.pdf)；[CANN 9.0 HcclAlltoAllV](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclcpp_07_0027.html)。具体批量/P2P约束以安装版本为准。
- [S10] [torch_npu v2.10.0分布式API封装](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/distributed/distributed_c10d.py)；[ProcessGroupHCCL及Work完成语义](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp)；[NPU stream/event封装](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/streams.py)。
- [S11] [PyTorch 2.10 new_group](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.new_group)；[batch_isend_irecv](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.batch_isend_irecv)。

仓库证据：

- [L1] [上一轮独立复核与结论边界](2026-11-03_online_decode_review.md)；[服务器原始审阅报告](../shared/current_exp/20261103_3h_online_decode/20261103_051202/REPORT_FOR_REVIEW.md)。
- [L2] [E0环境、显存及权重审计](../shared/current_exp/20261102_load_balance_e0/RESULTS_E0.md)；[E3通信后端结果](../shared/current_exp/20261102_load_balance_e0/RESULTS_E3.md)；[Profile](../shared/current_exp/20261102_load_balance_e0/RESULTS_PROFILE.md)。
- [L3] [S0物理槽映射纠错](../shared/current_exp/20261102_load_balance_e0/RESULTS_S0.md)。
- [L4] [S1量化状态与正确性](../shared/current_exp/20261102_load_balance_e0/RESULTS_S1.md)；[S1状态清单](../shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/S1_EXPERT_STATE.json)。
- [L5] [实际模型与服务配置探针](../shared/current_exp/20261102_load_balance_e0/evidence/e1/adapter_service_probe.json)。
- [L6] [归档SGLang 0.5.18模型/MoE参考源码](../shared/history_exp/07_PD共置与EPLB负优化_2026-10-11/eplb_layer_shift_review_20261014/sources/)；[归档更新器与原版EPLB代码](../shared/history_exp/07_PD共置与EPLB负优化_2026-10-11/pd_colloc_20261011/patches/)。
- [L7] [S2局部计算与准备状态](../shared/current_exp/20261102_load_balance_e0/RESULTS_S2.md)；[P2单槽及状态复制成本](../shared/current_exp/20261102_load_balance_e0/RESULTS_P2.md)。
