# Global-K Adapter Repair Report

## 结论先说

本轮在节点 `npu1-23` 的两张 Ascend 910C 上，只修正并验证已有 K=1 Prefill 原型。四项结果均为 PASS：

| 项目 | 结果 | 一句话证据 |
| --- | --- | --- |
| INPUT_FIDELITY | **PASS** | hidden 未打标签、未改写；实际进入专家的每一行都与源 token 一致；259+259 token、4144 个 top8 任务通过 |
| OWNER_STATE | **PASS** | RESERVED 假 ready、跨层旧引用、旧 generation、单边错误、固定租约覆盖均被真实状态拦下，dispatch 增量为 0，随后合法请求通过 |
| MEMORY_ACCOUNTING | **PASS** | baseline 不创建 flat/pool；candidate 只有 rank0 一份源 flat、rank1 一个目标槽；槽 37,810,176 bytes，最终 allocated 增量 37,810,688 bytes/rank |
| VALIDATOR | **PASS** | campaign/subrun 身份强制核对；11 个定点改坏回归全部被拒绝 |

最终判定：**`GO_REPAIRED_HARNESS_LIMITED`**。

限定范围：两张卡、合成 hidden/top8、有限主专家库（每 rank 每层 8 个专家）、layer49/50、K=1、expert43、原生 DeepEP alltoall、真实 ModelSlim W8A8。不是完整 MoE 层、全模型、生产服务或 TTFT。

生产格式：**UNKNOWN_NO_ACTIVE_SERVICE**。运行镜像中的实际 tensor format code 为 `2`（base/ND），但未启动生产配置，不能把它当作生产格式。

性能：**NOT_RUN**。本轮没有做性能比较，不报告加速、回本或净收益。

- 方案提交：`0dfc891`
- 基线：`3efa0a6`
- Campaign：`bb77a86cd4a34714b729e744d381d307`
- 代码哈希：`8a7c079901af96ed27dffbb5ac63893ac88f31850b87820e6d8247d503dbfde1`
- 功能 subrun：`1eac534120c645a99c905d7468a04460`
- memory baseline subrun：`feda27c8da664eb4bdc0bbe06cb727cd`
- memory candidate subrun：`fc8582213ed84d839aa48b78bfea9389`

## 1. INPUT_FIDELITY

### 实现

任务身份完全放在 side metadata 中，字段包括 sender、token index、top-k slot、logical expert、communication id 和 padding 身份。hidden 的前两列只保留测试值，不再存储 token 编号或 padding 标记。

任务 side metadata 按原生 alltoall 的实际排列处理：

- sender 侧按 global comm id 排序；
- 目标侧按 local comm id 排序；
- 输入先用同一次 dispatch 的 handle 验证：`reversed_local_mapping`、`reversed_global_mapping`、`input_splits`、`output_splits`。

每次真实 dispatch 后、进入量化专家计算前，adapter 逐行比较 `recv_hidden[row]` 与源 rank 保存的原始 hidden 行。记录每行 `equal=true/false`，而不是只写一个总结布尔值。padding 由 adapter 显式生成，专家计算前跳过，不能由零值、零 weight 或空 comm 推断。

### 真实 NPU 验证

功能矩阵在两 rank 上各执行 20 个 case，共 19 次实际 dispatch/case 组。重点用例：

| 用例 | 输入 | 结果 |
| --- | --- | --- |
| `large_259_partial` | 每 rank 259 token，共 518 token；top8 共 4144 任务 | PASS |
| rank0 实际任务 | 1812 | 与期望任务多重集完全一致 |
| rank1 实际任务 | 2332 | 与期望任务多重集完全一致 |
| rank0 最大绝对/相对误差 | 0.0063536 / 0.0024190 | 容差 0.02 内 |
| rank1 最大绝对/相对误差 | 0.0076449 / 0.0028395 | 容差 0.02 内 |
| 调用者输入前后 | SHA-256 完全一致 | PASS |
| 每行专家输入审计 | 全部 `equal=true` | PASS |

259-token 用例包含合法值 `0、16、199、200、201、256、258`，并有两行 hidden 完全相同但任务身份不同。旧方法依赖的 16、200、BF16 整数标签边界均被跨过。dispatch 容量设为 512，没有把容量不足误归因于算法。

其他通过项：

- q=0 且无槽；
- rank1 无输入但接收另一 rank 的专家任务；
- rank1 有输入但本地专家任务为 0；
- 两 rank 全空，双方早退且 dispatch=0；
- partial/all replica 分流；
- 非法 comm128 从 rank0 或 rank1 单边注入，双方一致拒绝；
- 全部真实任务的 expected/actual identity 相等，空编号合法任务为 0。

证据：`INPUT_FIDELITY.json`、`artifacts/campaigns/bb77a86cd4a34714b729e744d381d307/functional/`。

## 2. OWNER_STATE

owner rank1 不再只检查调用者传入的字典。它读取目标池的真实 slot record，并核对：

- pool identity；
- owner rank；
- local slot id；
- layer/expert；
- generation；
- 真实状态必须为 READY 或 IN_USE；
- request identity 与真实 record、当前 lease 三方一致。

有 replica 任务时，adapter 在 dispatch 前进入固定 lease 的 use scope，并保持到 combine 返回。租约本身继续固定，直到下一次 install 或 release 由池的同步机制安全归还；这样在途设备工作不会被新 generation 覆盖。

### 实际状态反例

| 用例 | 真实状态 | 结果 |
| --- | --- | --- |
| RESERVED 但请求声称 ready | record=`reserved`，ready count=0，请求 `ready=true` | 两 rank REJECTED，dispatch=0 |
| install49 后 install50，重放旧49请求 | 当前 lease layer=50、generation=3；旧请求 layer=49、generation=2 | 两 rank REJECTED，dispatch=0 |
| 重装49后重放旧 generation | 当前 generation=4；旧请求 generation=2 | 两 rank REJECTED，dispatch=0 |
| 仅 rank0 持有错误 generation | rank0 请求 generation=3，owner 实际 generation=2 | 两 rank REJECTED，dispatch=0 |
| 固定租约期间尝试覆盖 | 真实 record=`in_use`，users=1；第二次 acquire 失败 | PASS，无覆盖 |

每个反例后都在同一进程执行合法请求，均再次 PASS。最终 slot record 为 `free`，`active_bytes=0`，没有遗留 lease。

生成的真实状态转换：

```text
reserved(generation 1)
→ install49(generation 2)
→ install50(generation 3)
→ install49(generation 4)
→ 50(generation 5) → 49(generation 6)
→ 50(generation 7) → 49(generation 8)
→ 50(generation 9) → 49(generation 10)
```

证据：`OWNER_STATE_CASES.json`、`LAYER_REUSE.json`。

## 3. MEMORY_ACCOUNTING

baseline 和 candidate 使用独立进程、相同主库和相同 q=0 输入。baseline 不创建目标池、不创建 rank0 源 flat、不做权重格式复制。candidate 只创建实际需要的对象：

- rank0：一份源发送 flat，37,810,176 bytes；
- rank1：全组唯一目标槽，37,810,176 bytes；
- 持久额外专家状态仍为 K=1，且只有一个 slot storage pointer。

### 阶段实测

采样发生在阶段开始和结束时，阶段峰值在阶段启动时重置；全过程峰值另存，不用运行结束后的 `after_bank` 补写。

| 阶段/对象 | rank0 bytes | rank1 bytes |
| --- | ---: | ---: |
| 主库 allocated（加载后） | 605,016,064 | 605,016,064 |
| baseline q=0 peak allocated | 638,714,880 | 638,714,880 |
| candidate resources created peak | 642,826,752 | 642,826,752 |
| candidate install peak | 642,826,752 | 642,826,752 |
| candidate q=0 peak | 676,525,568 | 676,525,568 |
| candidate split peak | 676,525,568 | 676,525,568 |
| candidate release 后 current | 642,831,872 | 642,831,872 |
| baseline final allocated | 605,021,184 | 605,021,184 |
| candidate final allocated | 642,831,872 | 642,831,872 |
| 最终 allocated 差值 | **37,810,688** | **37,810,688** |

全过程峰值：

- baseline：allocated 655,377,920；reserved 761,266,176。
- candidate：allocated 676,525,568；reserved 801,112,064。

解释：

- rank0 的最终增量是源发送 bufffer，不是可执行副本；
- rank1 的最终增量是目标专家槽；
- 37,810,688 比 payload 37,810,176 多 512 bytes，是实际 tensor/allocator 记账开销；
- baseline 中没有 flat、pool 或复制格式 buffer；
- candidate 没有第二份可执行持久副本。

格式审计：主库与槽内 6 个 view 在本次 harness 中均为 format code 2，所有槽 view 的 storage pointer 与槽 backing storage 一致。镜像仍提示 internal-format warning，不能据此推断生产为 NZ。生产格式继续标 UNKNOWN。

证据：`STORAGE_AUDIT.json`。

## 4. 跨层复用

同一池、同一槽完成：

```text
49 → 50 → 49 → 50 → 49 → 50 → 49 → 50 → 49
```

共 9 次安装、6 次复算：

- 9 次 source/target hash 相等；
- 9 次 ready=true；
- generation 严格递增为 2 到 10；
- 所有安装使用同一 slot storage pointer `20618734993408`；
- 每次 slot bytes 均为 37,810,176；
- 6 次 layer49/50 复算输出均通过独立参考；
- rank0 最大复用误差 0.0063536，rank1 最大 0.0076449；
- 最终状态 free、active_bytes=0、backing_reallocations=0。

证据：`LAYER_REUSE.json`。

## 5. 可靠验收

每次 `./reproduce.sh all` 生成新的 campaign UUID 和三个独立 subrun UUID，并在启动时写 `campaign_manifest.json`。每个 case、install、memory 记录携带 campaign、subrun、rank、code hash 和场景身份。

validator 显式接收 campaign 目录，不推测“最新一次”。它从原始误差、实际任务集合、逐行输入审计、owner record、format/storage 和最终池状态重新计算 PASS，不读取人工 `case_pass` 作为结论。

11 个证据副本变异全部被拒绝：

1. 过期 campaign/run id；
2. 输入哈希改变；
3. owner 槽状态改坏；
4. 误差超限；
5. format ERROR；
6. 任务不守恒；
7. baseline 出现 flat；
8. 缺 rank；
9. 缺 install；
10. generation 乱序；
11. 最终槽未释放。

证据：`VALIDATOR_REGRESSIONS.json`、`VERDICT.json`。

CPU 回归同时通过：slot identity 12 项、expert search 12 项，原始输出见 `CPU_TESTS.txt`。

## 6. 修改内容

- 移除 hidden 标签写入；新增 side metadata 和真实 dispatch 排列适配。
- 新增逐行 expert-input 审计与实际任务身份。
- 新增 owner 真实 pool record 检查、固定 lease 覆盖保护和跨 rank 一致拒绝。
- control collective 使用独立 Gloo 组，避免与 NPU dispatch/combine 流混用。
- 修正 memory baseline，使 no-copy 对照不创建 flat/pool/复制格式对象。
- 新增阶段峰值采样和全过程峰值。
- 新增 campaign/subrun/code hash 身份、严格 validator 和 11 个变异回归。
- 修复 control collective 非对称导致的部分 rank 等待，以及 install/case 文件可见性竞态。

## 7. 未完成与阻塞

- 生产格式配置未确认，继续标 UNKNOWN。
- 合成 hidden/top8，不是真实 AgentX 或人类文本路由。
- 主专家库只有每 rank 每层 8 个专家，不是完整 128/128。
- 未接完整 MoE 层、shared expert、全模型服务、TTFT、Decode、K=2、并发或跨节点。
- 本轮性能始终 NOT_RUN。
- 下一处具体阻塞：在生产格式和完整主专家分组 runner 下对齐同一边界的计算/通信终点；完成前不能做性能结论。

## 8. 复现

```bash
cd shared/current_exp/20261123_global_k_adapter_repair

# 重新生成真实 NPU campaign、运行 CPU 测试并验收
./reproduce.sh all

# 对已有 campaign 重新运行 validator，必须显式给目录
./reproduce.sh validate artifacts/campaigns/bb77a86cd4a34714b729e744d381d307

# 单独入口
./reproduce.sh functional
./reproduce.sh owner-state
./reproduce.sh memory
./reproduce.sh cpu
```

`all` 的退出码为 0 当且仅当四项全部 PASS。预期拒绝由 case 捕获；意外失败不会记为成功。
