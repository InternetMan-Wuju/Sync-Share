# EPLB后续会议报告：Online效果与开销、Decode大EP模拟

**日期：**2026-11-03。**场景：**GLM-5.2 W8A8／昇腾910C／SGLang。

**承接：**初始《EPLB问题.pdf》末页的两项任务：①选一个online方法，测效果及overhead；②少卡采decode trace，再模拟大EP。

**证据版本：**服务器结果125423a；3小时执行卡初版a2189ea。本次检查远端时，没有发现比a2189ea更晚的服务器结果提交。以下是对125423a原始记录和代码的整合复核，未在本机重新运行NPU实验。

## 1. 本次先回答两个问题

| 上次的问题 | 现在可以回答 | 尚未完成 |
|---|---|---|
| Online方法、效果、overhead | 主方向选UltraEP式当前路由复制与配额。已有聚合路由下的局部计算降幅45.1%／47.8%；已打包专家状态的同设备复制实测0.117 ms，host等待完成0.203 ms | 逐step在线planner、全局交换、远端复制、完整层净收益；尚未复现UltraEP原始NPU runtime |
| Decode少卡trace→大EP | 已有真实量化权重的eager局部成本曲线，证明不能按token数线性缩放；已有原始dump入口可查 | decode逐step有效计数仍未核验；EP16／32／64六格尚未生成；跨节点通信未标定 |

**核心判断：新结果补上了局部机制证据，仍不足以宣布online回本或给出EP64 TPOT。** 本轮发现两处确定的配额／计数代码错误，以及调用链正确性和计时口径缺口。原始测量保留，受影响的解释需要收紧。

原PDF中，EP64增加副本后TTFT约20.3→17.0秒，整体输出吞吐91→88 tok/s；额外槽从256变320，即routed专家权重部分约增加25%，不等于总HBM增加25%。本轮继续回答“节省的计算能否覆盖管理、搬运和通信”，不以负载图更均匀代替服务收益。

## 2. Online：方法已选，当前现场测到的是哪一部分

### 2.1 选择UltraEP式当前路由复制与配额

本层路由产生后，按当前任务量选择副本和整数配额，任务交给合法实例执行；临时专家buffer跨层复用。它解决当前热点与历史预测不一致、永久副本随层数累积的问题。代价是统计、规划、状态准备和依赖等待。

作者在BF16、EP40／64、机架级高速互联的prefill实验中报告：吞吐为SGLang的1.56倍、EPLB的1.29倍。论文表3的合成负载规划实验平均solver耗时0.111 ms；另一个Qwen3训练案例中，每MoE层平均的前向non-MoE部分比理想参考多0.33 ms。这些是不同口径的作者结果，不能代入910C量化部署。[UltraEP论文][ultra]

**本地已经运行的是固定候选副本上的离线配额与局部回放，不是完整UltraEP在线实现。** 因而下一步用“参考UltraEP的当前路由复制机制验证”命名，避免把算法思路、CPU参考planner和生产NPU实现混成一项成绩。

### 2.2 S3的局部效果可以保留，但必须标明聚合输入

从S3原始CSV复算：

| 数据集标签 | 关闭e43/e40跨rank副本时的局部最大值 | 时间成本指导配额后的局部最大值 | 差值 | 降幅 |
|---|---:|---:|---:|---:|
| calibration | 2.687796 ms | 1.474721 ms | 1.213075 ms | 45.13% |
| holdout | 2.755812 ms | 1.437332 ms | 1.318480 ms | 47.84% |

算术与原始记录一致，但有五个限定：

1. 输入由**4个prefill step的计数相加**得到，layer49共32768个token–expert任务，再作为一个虚拟大批次回放。因此1.213 ms不是某个真实step的收益预算。
2. 使用人工hidden；各rank在一张NPU上顺序计时，取局部最大值形成计算代理，没有测真实并行关键路径。
3. 受影响rank0／1／2使用对应专家权重；其余rank沿用代表性17-group计时，不能称全rank精确权重实测。
4. owner-only只关闭e43/e40跨rank副本，其他重复槽仍可能分担同rank任务，不能当作严格的“额外槽全部禁用”adapter。
5. holdout按当前输入重新求配额，适合检验当前负载决策；不代表一份冻结配额已经跨流量验证。

**可以说：该聚合工作负载下，合法配额有明显局部计算空间。不能说：online已加速45%，或每次在线规划有1.2 ms预算。** [原始S3数据][s3data]、[回放实现][s3replay]

### 2.3 Overhead已有局部数字，总账仍不完整

| 成本 | 已有结果 | 已包含／未包含 |
|---|---:|---|
| 同一NPU内复制37,769,216 B已打包状态 | device中位数0.117140 ms；host等待完成0.202902 ms | 目标buffer已分配；包含w13、down和两组scale的copy；未含远端通信、首次格式准备 |
| H2D复制同一状态 | device 1.805760 ms；host 1.893276 ms | 独立本地路径，不能替代跨NPU传输 |
| CPU参考planner | 未测 | 脚本产生了计划，没有记录其执行耗时 |
| 全局统计、远端权重复制、dispatch/combine增量 | 未测 | 当前一卡不能直接测量 |
| 完整online净收益 | 未测 | 没有同输入、同计划的A／B／C／D完整层对照 |

0.117 ms来自一次预热后5次计时，复制源已在本NPU驻留。计时时旧读者已完成，所以也没有测到buffer冲突等待。跨层生命周期测试仅证明两个完全同步顺序case的输出一致，不能据此宣布并发复用安全。[P2原始记录][p2data]、[复制实现][p2code]

不得直接计算“1.213−0.117=净收益”：前者是聚合输入的局部计算代理，后者是单状态同设备copy，两者都没有覆盖真实online执行。

真实验收仍使用同输入、同预算的：A原路径；A_adapter额外槽零任务；B预计算计划＋预装副本；C当前路由统计／全局交换／规划＋预装副本；D再加入真实远端复制。C/D共用相同统计、planner及dispatch/combine路径，D只新增远端状态传输和相应就绪等待。由C−B、D−C、A−D分别观察统计与规划增量、复制增量和最终净收益，并检查重叠与缓存状态。

## 3. Decode：小计算量已有证据，大EP预测尚未完成

### 3.1 新成本曲线说明不能线性缩放

S2的单活跃专家、8物理槽局部链：

| 有效行数M | 完整局部链中位数 |
|---:|---:|
| 1 | 0.22448 ms |
| 8 | 0.18952 ms |
| 64 | 0.18904 ms |
| 256 | 0.19326 ms |
| 1024 | 0.31728 ms |

256→8行，任务量减少32倍，记录中的耗时却只从0.19326降至0.18952 ms。相同1024行，从1个活跃专家变成8个均匀活跃专家，耗时0.31742→0.45219 ms。

这说明**该eager局部链的时间由工作量之外的因素共同决定**。它没有分离Python提交、设备启动、权重读取等贡献；不能直接将平台区归因于某一种硬件瓶颈。该路径也没有证明与服务中的融合、graph decode路径等价。[S2原始表][s2data]、[调用链][s2code]

### 3.2 不能直接把旧S3回归用于小batch

S3的4.60%留出NRMSE只覆盖较大行数的17-group向量。将原系数直接用于已有S2小M点：

| 输入 | S2实测 | 旧模型预测 | 低估 |
|---|---:|---:|---:|
| M=1，单活跃专家 | 0.22448 ms | 0.13064 ms | 41.8% |
| M=8，单活跃专家 | 0.18952 ms | 0.12783 ms | 32.5% |

这两个S2小M点实际已参与模型拟合，仍有上述大误差；4.60%的留出NRMSE仅覆盖较大M，不能证明小M有效。模型也没有区分4／8／16／17物理槽的特征。六格模拟应直接标定实际出现的完整槽向量，不能拿旧模型输出填表。

### 3.3 原始decode计数还需要核验

两份S0记录的decode总计都是614400。若64个step均只有1个有效token，理论任务数应为64×75×8=38400，前者恰为16倍；**这只是待核实的重复因子，不是允许直接除16的依据**。有效batch、TP重复与graph padding尚未逐step证明。

原dump曾被服务器成功读取。下一轮优先取layer49逐step计数，不再先sum掉step维。只有确认某步恰为1个有效token，且规范化后8个不同专家各1次，才能恢复top-8集合；仍不能恢复router weight、原始顺序和dispatch源归属。batch>1时，仅专家直方图不能恢复token共选关系。[S0证据][s0data]、[读取代码][s0code]

### 3.4 大EP六格应该交什么

固定全局有效batch为1／8，分别映射EP16／32／64；每个EP使用同一份逻辑路由。逐step生成每rank的16／8／4槽向量，保留零槽和真实padding，再用单NPU标定各工作类型。

每格输出：活跃rank、R、计算代理max_r F_r、相对EP16的计算节省Δ，以及允许新增的通信／调度成本。覆盖不足的格留空，不只测任务最多的rank，也不把单层结果乘75。

若Δ为正，只有额外暴露通信与调度成本小于Δ，才有机会获得净收益；这个预算属于指定输入和执行路径下的模型，仍需少卡并行回放校准。无网络标定时不给出“已验证EP64 TPOT”。

同时，大EP可能受任务离散性限制：只有1个token、top-8且专家任务不可再分时，EP64即使分配最好也只有8个rank执行有效任务，最小R仍是8。高R本身不能证明复制有收益。实际通信还需要目标attention-DP／TP下的源rank映射，不能沿用EP16源rank标签。

## 4. 复核发现：哪些结论必须先修正

### 4.1 行数均衡对照重复计算热点

s3_plan.py的row_quota_search中，b2已含e43/e40，却又使用b2+n43+n40−target计算迁出需求，导致两热点全部迁走。按同一映射、同一预算独立枚举整数配额：

| 数据 | 原“行数minimax”实际最大行数 | 正确最小最大行数 | 一个合法配额：e43／e40迁出 |
|---|---:|---:|---:|
| calibration | 5171 | 3983 | 3367／2790 |
| holdout | 4900 | 3943 | 3371／3015 |

所以，**“时间配额优于正确行数均衡”目前没有成立的对照证据**。正确配额的NPU时间尚未测得，不能从行数反推。45.1%／47.8%相对原已测owner配置的算术仍保留。[求解代码][s3plan]

### 4.2 两处余数分配不守恒

S0 count_rank_load与S3 calibration rank_vectors把逻辑expert值和物理slot位置混用，导致分配余数错误。复算总数：

| 映射 | calibration原函数 | holdout原函数 | 正确总数 |
|---|---:|---:|---:|
| 旧映射 | 32773 | 32774 | 32768 |
| 修正映射 | 32763 | 32762 | 32768 |

三位小数的R未因此改变，但任务守恒断言不成立，不能继续把这些向量当作准确输入。S3最终build_vectors采用物理index，计划本身守恒；这不等于前面的标定数据也正确。[S0函数][s0code]、[标定函数][s3cal]

### 4.3 S1正确性不能直接覆盖S2及生产路径

S1的36个case支持其grouped链内不同分组输出一致；但1.248%的独立参考误差实际比较的是dense_chain与反量化参考，没有接到grouped副本链。S2又将gate/up从两次GMM改成一次融合GMM，不能直接继承S1 PASS。

最小补充是：同一小输入、相同量化状态下，连接dense/reference、S1 grouped与本轮实际S2／生产调用链；对实际使用专家检查offset、scale和布局。已有top-8测试验证的是合成专家输出的加权合并，尚未串起八个真实专家的计算和副本分流。无需全量重跑S1，但不能跳过本轮实际调用链的数值核验。[S1代码][s1code]

### 4.4 复用计时器需要落实块长

旧S2只有3次预热；用一次调用估计repeat后没有核验实际块时长。M=1的块约18.4 ms，M=8约22.9 ms，未达到声明的100 ms目标。五个块的p05/p95在该实现里就是最小／最大块均值，不是生产token尾延迟。

保留旧均值作为初步数据；新实验按20次预热、实际块长至少100 ms、5个块及交错顺序执行。时间不足就减少测点，不改变正确性与计时口径。

## 5. 仍然只用3小时，复用已有代码，只补缺口

| 时间 | 本轮动作 | 必须交付／停止条件 |
|---|---|---|
| 0–20分钟 | 查旧逐step dump；修复CPU余数分配、行数目标及空adapter契约 | 每专家及总任务守恒；行数求解与独立小枚举一致；第17槽在空adapter中为零任务 |
| 20–65分钟 | 复用加载，完成实际计时链的小样本数值桥接；检查offset；修正计时块长 | 未通过正确性则不采性能；不能把旧S1 PASS当生产整链通过 |
| 65–110分钟 | Online只做960／64热点及512／512对照，计planner、完整本地链和本地状态复制 | 两case组合正确性、配额和成本账本；实测与代理分列 |
| 110–155分钟 | Decode六格，4个开发step＋4个留出step；仅标定出现的工作类型 | 禁用未验证的大M模型外推；未覆盖格留空 |
| 155–180分钟 | 停采样，整理结果、原始记录和Git提交 | 180分钟结束，不自动续跑 |

CPU修正本轮只补最小守恒与求解验证；不增加全套S3重跑，不把“优于正确行数均衡”设为本轮必须产出的正结论。局部eager链可以保留为代理，但不能未经证据称为生产路径；若本轮要求的正确性门未通过，按执行卡退出。

人工路由可以验证机制，不能填补真实trace的证据身份。单卡本地copy可以提供量级，不能填补远端overhead。两条主任务完成到哪一步，由结果表中的证据类型决定。

**下一次汇报的两句话应当由数据填完：**

- Online：在什么逐step或人工case上，完整本地计算代理节省多少；planner、本地状态准备各花多少；全局交换、远端复制和真实净收益还缺什么。
- Decode：相同逻辑任务在EP16／32／64下，计算代理如何变化；新增通信最多允许花多少；少卡校准误差与跨节点参数哪些已经测过。

详细服务器操作见[3小时执行卡][card]，不再另开解法清单。

## 附录：复核范围与证据

本次读取了S0–S4/P2文档、原始JSON／CSV及相应代码；复算S3降幅、槽位任务守恒和行数配额。32份manifest所列镜像文件在统一Windows CRLF／LF换行后hash匹配。该检查确认文件内容对应，不能替代NPU复跑或证明实验设计正确。

S4拆分、shared expert和双stream等结果保留在原仓库，本次不作为两条主任务已经完成的证据。未改写原始实验数据，也未在本机实现或执行NPU实验。

[ultra]: https://arxiv.org/html/2606.04101v3
[card]: https://github.com/InternetMan-Wuju/Sync-Share/blob/main/handoff/2026-11-03_glm52_3h_online_decode.md
[s3data]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/S3_ALLOCATION.csv
[s3replay]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s3_replay_affected.py
[p2data]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/P2_LOCAL_CHECKS.json
[p2code]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/p2_local_checks.py
[s2data]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/S2_LOCAL_COST.csv
[s2code]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s2_local_cost.py
[s0data]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/evidence/s0_s4_p2/S0_EVIDENCE.json
[s0code]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s0_mapping_audit.py
[s3plan]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s3_plan.py
[s3cal]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s3_cost_calibration.py
[s1code]: https://github.com/InternetMan-Wuju/Sync-Share/blob/125423a6c43acb4aba632dd3346f2c0ce8b50be2/shared/current_exp/20261102_load_balance_e0/scripts/s1_npu_numerics.py
