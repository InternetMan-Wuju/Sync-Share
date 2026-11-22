# Prefill K=1：统一执行路径、格式与显存验收

## 结论先说

1. **统一功能路径：通过，但范围受限。** 正常分流、q=0、空输入、内部 padding、非法任务拒绝、未 ready、错层、过期 generation、真实 W8A8 和 49→50→49 三轮复用全部通过同一个 `UnifiedK1Adapter.execute()` 入口。
2. **格式与显存：** 实际 harness 中主库和槽 tensor 都是 base/ND format code 2；格式转换没有产生额外 storage。生产服务未运行，因此**生产格式配置仍未确认**。全组持久额外专家槽实际为 1 份 `37,810,176 bytes`；候选相对无副本独立进程在 rank1 的稳态 allocated 增量约 `37,810,688 bytes`、峰值 allocated 增量约 `21,171,712 bytes`；rank0 为 0。另有 rank0 源发送 flat buffer `37,810,176 bytes`。
3. **性能：NOT_RUN。** 本轮没有完成相同主专家分组 runner 的可比三条路径，因此没有生成 `TIMINGS.csv`，不报告提速或未获益。

最终自动判定：`GO_FUNCTIONAL_LIMITED`。

- 方案提交：`fa70244`
- 上一轮基线：`607121e3ca87c4514a234b99d36f69a89383b211`
- 主机：`npu1-23`
- 设备：两张 Ascend 910C，`davinci0`、`davinci1`
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`
- 真实层：layer49、layer50
- 真实副本：expert43，唯一外部槽

## 1. 单一执行入口

所有功能案例都调用：

```text
UnifiedK1Adapter.execute()
```

入口内部完成：

```text
逻辑 top8 校验
→ 逻辑专家到258通信编号映射
→ 可选 expert43 → comm257
→ 非法编号/槽 ready/层/专家/generation 检查
→ 两 rank 汇总拒绝或继续
→ 仅由 adapter 生成内部 padding
→ 原生 DeepEP NORMAL/alltoall dispatch
→ 真实 ModelSlim W8A8 主专家/副本计算
→ 原生 combine
→ 剥离 padding
→ 返回原始合法 token 顺序
```

没有把独立协议脚本、手工专家链和负例脚本分别拼成“统一通过”。

### 1.1 跨 rank 一致拒绝

预期错误发生时，每个 rank 先形成本地错误位，再参加统一状态汇总。只有所有 rank 都得到“允许”后才实际调用 dispatch。

负例：

- rank0 注入空 comm128；
- rank1 注入空 comm128；
- 未 ready 副本；
- 目标槽 layer 不匹配；
- 过期 generation。

结果：

- 5 个负例均返回 `REJECTED`；
- 两 rank 状态一致；
- 每个负例的 `dispatch_delta = 0`；
- 每个负例后立即在同一进程运行合法 q=0 用例，均再次 PASS，dispatch 正常发生。

合法 logical expert128 映射到 comm129；空 comm128 没有被误禁。

### 1.2 功能矩阵

两 rank 共 16 个同入口用例：

| 用例组 | 结果 |
|---|---|
| 双卡 mixed，q=0 且无槽 | PASS |
| 只有 rank0 有输入、全部任务在 rank0 | PASS，rank1 零专家任务 |
| 只有 rank1 有输入、全部任务在 rank1 | PASS，rank0 零专家任务 |
| 两 rank 都无输入 | PASS_EMPTY，双方一致早退，dispatch 0 |
| 部分 expert43 任务到副本 | PASS |
| 全部 expert43 任务到副本 | PASS |
| 非法 comm128 从 rank0/rank1 注入 | REJECTED，dispatch 0 |
| 未 ready / 错层 / stale generation | REJECTED，dispatch 0 |
| 每种负例后的恢复用例 | PASS |

真实任务检查：

- 每个合法用例 `task_multiset_equal=true`；
- 空编号合法任务始终为0；
- 内部 padding 的 `padding_expert_calls=0`；
- padding 不占副本槽；
- 实际 dispatch 调用计数：两 rank 各16次；
- 最终验证器从记录计算，不写死 dispatch 状态。

候选相对独立参考的最大绝对误差约 `0.00048`，预定义绝对容差 `0.02`，相对容差 `0.02`。

输入仍为合成 hidden/top8，不是真实 router trace。

### 1.3 内部 padding 身份

只有 adapter 能生成 padding：

- padding token 使用 `PADDING_TAG_BASE + rank` 的显式 token 身份；
- padding 使用零权重和空 comm128承载，但身份不是由“零权重”或“comm128”推断；
- 进入 dispatch 后按显式 tag 识别；
- 在专家计算前跳过；
- combine 后从返回结果中剥离；
- rank0-only 和 rank1-only 两个用例均验证了真实零专家任务 rank 继续参与通信。

## 2. 格式审计

实际隔离 harness 中：

- 主库 w13/w2/scale/offset：format code **2（base/ND）**；
- 唯一槽内所有权重 view：format code **2（base/ND）**；
- 槽内6个状态字段共享唯一槽 storage：`aliases_slot_storage=true`；
- 主库到外部槽的格式转换在本次实际路径中为 storage alias，未新增持久格式副本；
- 镜像仍打印 internal-format warning，不能据此推断生产一定使用 NZ。

生产格式结论：**UNKNOWN_NO_ACTIVE_SERVICE**。没有启动生产配置服务，因此不能声称生产格式已经匹配。

格式证据：`artifacts/FORMAT_AUDIT.json`。

## 3. 显存审计

### 3.1 主库和槽

- 每 rank 真实主专家集合：8个/层 × 2层；
- 每 rank 主库按 storage 去重：`604,962,816 bytes`；
- 唯一槽 storage 数量：1；
- 槽 bytes：`37,810,176`；
- 槽 owner：rank1；
- 空 comm128 持久专家状态：0；
- rank0 源发送 flat buffer：`37,810,176 bytes`。

### 3.2 独立进程 baseline vs candidate

同样的主库、输入和 adapter 路径，分别在新进程运行：

| rank | baseline final allocated | candidate final allocated | 增量 |
|---:|---:|---:|---:|
| 0 | 642,832,384 | 642,832,384 | 0 |
| 1 | 605,021,696 | 642,832,384 | 37,810,688 |

rank1 的稳态增量比槽 payload 多512 bytes，来自实际 tensor/allocator 开销，已如实记录。

峰值 allocated 增量：

- rank0：0；
- rank1：`21,171,712 bytes`。

allocated、reserved、峰值和输入/参考/工作区分类在 `artifacts/STORAGE_AUDIT.json`。

## 4. 跨层复用

统一入口完成：

```text
layer49 → layer50 → layer49
```

三层循环共：

- 7次安装；
- 6次切换后真实计算；
- generation：1,2,3,4,5,6,7；
- 层序列：49,50,49,50,49,50,49；
- 所有安装 source/target hash 相等；
- 所有安装使用同一槽 storage pointer `20622706999296`；
- 每次安装槽 bytes `37,810,176`；
- 6次复用输出均通过真实参考；
- rank1 每次 all-replica 测试有真实副本任务；
- 空编号任务始终0；
- 最终池 `active_bytes=0`、free=1。

验收器从原始安装/计算记录重新计算上述结论。

## 5. 自动验收器

`artifacts/VERDICT.json` 由 `build_validation.py` 从原始记录生成，不读取人工写的 PASS 字符串。

验收器回归：

- 删除一个 rank：拒绝；
- 删除一条安装记录：拒绝；
- 打乱 generation：拒绝；
- 将最终槽状态设为仍有 active bytes：拒绝。

验证器回归只改内存副本，不改实际原始记录。

## 6. 修改代码

- 新增统一 `UnifiedK1Adapter`，合并校验、跨 rank guard、padding、dispatch、真实计算和 combine。
- 增加显式 padding token 身份，不使用零权重/空 comm 推断。
- 增加槽身份/ready/layer/generation 检查且只在有 replica task 时要求槽。
- 真实主专家集合扩到每 rank 8个/层，覆盖全部任务只落在一张卡的场景。
- 保留 DeepEP NORMAL/alltoall 通信后端和真实 ModelSlim W8A8 GMM。
- 增加独立参考、格式/alias审计、独立进程 baseline/candidate 内存对照。
- 增加证据驱动验收器和缺失 rank/安装、乱序 generation、未释放状态的回归测试。

## 7. 未完成项

- 生产格式配置未确认；
- 完整 128/128 主库未加载；
- 原生分组 runner 未接入三条性能路径；
- 性能 NOT_RUN；
- 完整 MoE 层、shared expert、全模型服务、TTFT/吞吐、Decode、K=2、跨节点、多请求并发未测。

下一处具体阻塞是：**在生产格式配置下接入真实主专家分组 runner，并对齐 B0/B1/B2 的完整输出终点；在此之前性能数字不可比。**

## 8. 复现

```bash
cd shared/current_exp/20261122_global_k_unified_adapter

./reproduce.sh functional
./reproduce.sh format-memory
./reproduce.sh reuse
./reproduce.sh validate
./reproduce.sh cpu
./reproduce.sh all
```

主要证据：

- `artifacts/CASE_RESULTS.json`
- `artifacts/FORMAT_AUDIT.json`
- `artifacts/STORAGE_AUDIT.json`
- `artifacts/LAYER_REUSE.json`
- `artifacts/VERDICT.json`
- `artifacts/functional.log`
- `artifacts/memory_baseline.log`
- `artifacts/memory_candidate.log`
- `artifacts/runtime_sources.sha256`
- `artifacts/model_metadata.sha256`
- `artifacts/environment.json`
