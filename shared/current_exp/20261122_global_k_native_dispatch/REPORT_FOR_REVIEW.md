# Prefill 全组 K=1：原生分发可行性与已有副本诊断

**结论摘要：原生流程当前不能直接接入全组 K=1；持续驻留副本条件下也没有稳定回本；最大确定阻塞是安装镜像要求物理专家数按 EP rank 均匀整除，无法表达 256 逻辑专家 + 1 个全组唯一额外实例（257 个物理实例）。**

- 方案提交：`6ac4875`
- 实验基线：`d0c4a3f482bd96471b01daf76b104e4b21c3b5ac`
- 分支：`main`
- 主机：`npu1-23`
- 设备：两张 Ascend 910C，`davinci0`、`davinci1`
- 镜像：`quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3`
- 模型：`/data_lib/data/models/glm-q`，layer49 / expert43，ModelSlim W8A8
- 槽：全 EP 组 K=1，固定 owner rank1；单槽 `37,810,176 bytes`
- 范围：手工配额的两卡专家子链；不是完整 MoE 层、服务 TTFT 或 Decode

本轮没有启动完整模型服务、Decode、K=2、跨节点、多请求并发或自动搜索器。

## 1. 原生分发接口核查

没有发现正在运行的 SGLang 服务进程，因此不能从活动进程直接读取本次 server 参数；接口证据来自本轮实际安装镜像源码，以及 GLM/ModelSlim 实际调用链。

实际代码路径：

```text
glm4_moe.Glm4MoeSparseMoeBlock
→ NPU TopK + logical/physical expert mapping
→ DeepEPMoE/FusedMoE.dispatch
→ FusedMoE.run_moe_core
→ quant_method.apply
→ ModelSlimFusedMoEMethod / AscendRunnerCore
→ GroupedMatmul / npu_grouped_matmul
→ DeepEP/FusedMoE.combine
```

S/P/D 定义：

- S：原 token 发送卡，也是结果返回位置；
- P：原版布局中 logical expert43 的权重卡；
- D：额外槽所在卡。

当前两卡实验是 **S=P=rank0，D=rank1**。因此当前手工路径仍然是 `S→D→S`，不能声称已经消除了真实三卡路径里的中间通信。

### 1.1 五项接口结论

| 合同 | 结论 | 精确证据 |
|---|---|---|
| 256 逻辑专家 + 1 全组额外实例，非等长 128/129 | **当前不支持** | `fused_moe_triton/layer.py:373` 直接断言 `num_global_routed % storage_ep_size == 0`；257 % 2 = 1 |
| 保持 logical top-8 和权重不变 | **部分支持，但受全局物理布局限制** | `expert_location_dispatch.py` 有 logical→physical replica 映射，TopK 可保留原 top-k 权重；但物理专家总数仍须均匀分到 rank |
| 外部槽计算 | **需要小适配，不是原生入口** | `modelslim.py:593-604` 从 `layer.w13_weight/w2_weight/scale` 构造 QuantInfo；`matmul.py:35-43` 直接从 layer 属性读权重；没有 slot handle 参数 |
| 正确回填/合并 | **协议字段存在，但 257 实例未证实** | `DeepEPNormalDispatchOutput` 携带 topk IDs/weights；`DeepEPNormalCombineInput` 沿同一 metadata 返回；布局构造在进入该协议前已失败 |
| 所有 rank 协议一致 | **当前不均匀布局不支持** | `FusedMoE` 给每个 rank 相同 `num_local_experts`；`ExpertLocationMetadata.num_local_physical_experts` 也要求物理专家数整除 EP size |

完整接口审计：`artifacts/INTERFACE_AUDIT.json`；源码片段：`artifacts/interface_source_snippets.txt`；源码哈希：`artifacts/runtime_sources.sha256`。

### 1.2 为什么没有做 C1 原生接入

C1 要求真实 256 逻辑专家 / top-8 / 两 rank / 全组 K=1 的 metadata 合同。当前 `FusedMoE` 在构造阶段就要求 257 个物理专家均匀分给 2 rank，无法形成合法的 128/129 布局。继续做只能：

- 偷偷改成每卡一个槽，违反 K=1；或
- 绕过 FusedMoE，自写一套 P2P，不能称为原生分发；或
- 改框架支持非均匀 per-rank expert count，超出本轮允许范围。

因此本轮在具体断言处停止原生接入，保留 `classify=current_not_supported`，没有把手工专家链命名为原生路径。

## 2. 生命周期对照设计

固定主点 `n=8192, q=2048`，复核点 `n=4096, q=1024`。

### 2.1 旧语义

- 每次 split 前安装槽并 ready；
- 单次 split 使用完后 release/归还；
- 原路径不使用槽；
- 安装发生在计时外，但安装成本单独记录；
- 每个模式前都重新建立外部槽视图/专家链对象。

主点共 12 对，两个输入种子，每个种子 3 对 original-first、3 对 split-first。

### 2.2 持续驻留

- 同一 `(n,q)` 测量块开始前安装一次；
- 租约保持 READY，块内所有 12 对共享同一槽 generation、同一权重视图和专家链对象；
- 每次 split 通过 `use` 使用引用；输出返回且 use 引用退出后才 ack；
- 原路径和 split 都面对同一份已驻留槽；
- 块结束后才 release/逐出；
- 安装和逐出成本不计入单次 pair 时间，但单独记录。

每轮两个点、每点 12 对；两个独立进程轮次。共 **96 次 resident 正式执行**。旧语义另有 **24 次正式执行**。

## 3. 正式结果

### 3.1 旧语义：8192/2048

| 项 | 结果 |
|---|---:|
| 原路径中位数 | 2.1763 ms |
| 已有副本中位数 | 2.2379 ms |
| 逐对净节省中位数 | **-0.0550 ms** |
| 逐对范围 | -0.1079 .. +0.0079 ms |
| 正收益组 | 1/12 |
| 原路径先组中位数 | -0.0550 ms |
| 分流先组中位数 | -0.0523 ms |

结果与上轮一致：旧生命周期下已有副本仍亏。

### 3.2 持续驻留：8192/2048

| 轮次 | 原路径中位数 | 驻留副本中位数 | 逐对净节省中位数 | 正收益组 | original-first | split-first |
|---|---:|---:|---:|---:|---:|---:|
| round1 | 2.2389 ms | 2.2253 ms | **+0.0111 ms** | 9/12 | +0.0152 ms | +0.0111 ms |
| round2 | 2.1704 ms | 2.1897 ms | **-0.0098 ms** | 4/12 | -0.0189 ms | +0.0079 ms |

round1 有 9/12 正收益，但 round2 只有 4/12；没有达到文档要求的“两轮均 12/12 且两个顺序组均正”。

### 3.3 持续驻留：4096/1024

| 轮次 | 原路径中位数 | 驻留副本中位数 | 逐对净节省中位数 | 正收益组 |
|---|---:|---:|---:|---:|
| round1 | 1.3387 ms | 1.5323 ms | **-0.1943 ms** | 1/12 |
| round2 | 1.3301 ms | 1.5176 ms | **-0.1871 ms** | 0/12 |

复核点持续驻留仍明显亏，说明问题不只是每次重新安装。

### 3.4 安装/逐出成本

旧语义每次 split 安装：

- 源卡转换/打包/发送中位数：`0.4210 ms`；
- 目标卡接收/ready 中位数：`0.8303 ms`；
- 目标槽 release 中位数：`1.2083 ms`。

持续驻留块级安装：

- 源卡实际安装中位数：`0.5165 ms`；
- 目标卡实际安装中位数：`0.9275 ms`；
- 块末逐出中位数：`0.3861 ms`。

首次 warm install 的 80–755 ms 级开销来自首轮 HCCL/格式准备，不计入正式 pair；正式块级安装数字来自已预热路径。

### 3.5 正确性与槽状态

- 旧语义 24/24 样本 `max_abs = 0.0`；
- resident 96/96 样本 `max_abs = 0.0`；
- 无非有限值；
- 所有样本任务守恒；
- resident 每个点仅有 2 次池 acquisition（warm + resident block），块内 12 对不重复安装；
- generation、owner、local slot、layer、expert 均写入原始样本；
- 每轮结束槽恢复 free。

原始配对数据：`artifacts/PAIRED_TIMINGS.csv`。

## 4. 设备时间线

对 `8192/2048` 的 original 和 existing 各采 2 个 profiler 短窗口，两个 rank 都记录。原始 trace 留在服务器：

```text
shared/current_exp/20261122_global_k_native_dispatch/artifacts/profiler_raw/
```

约 4.7 MB，未上传到 git；摘要和 trace 文件哈希在 `artifacts/DEVICE_TIMELINE_SUMMARY.json`、`artifacts/device_timeline_rank{0,1}.json`。

设备证据显示：

- original 两 rank 只有一组公共控制/确认 HCOM；
- existing 在源卡增加 activation/result HCOM 收发；
- existing 在目标卡同时出现 activation receive、result send 和更多 HCOM wait；
- 目标卡 existing 的 GroupedMatmul 计算确实出现，但 HCOM 等待比计算更显著；
- profiler 本身显著放大绝对时长，因此这些数字不能替代正式 pair 时间；
- 两卡时钟未校准，不计算“设备核重叠百分比”。

原文关键结论：**额外时间不只是目标卡 GMM；目标端 HCOM 接收/发送及等待仍是主点 resident 路径的重要组成。**

## 5. CPU 回归

沿用上一轮 24 项：

- 12 个槽安全/释放竞争测试通过；
- 12 个 expert-search/独立 oracle 测试通过。

本轮没有新增生命周期协议之外的生产算法，也没有把 CPU 模拟作为 NPU 通过证据。

## 6. 未完成项和边界

- 原生 257 实例 dispatch/combine 未实现；
- 原生 expert43 子链接入未尝试，因为 A 阶段布局合同已失败；
- 完整 MoE 输出、shared expert、全 expert combine 未测；
- 完整服务 TTFT/吞吐未测；
- Decode、K=2、跨节点、多请求并发未测；
- 真实跨请求自然命中未证明；
- 持续驻留正收益只代表“同一进程、同一专家、同一测量块”的条件成本。

## 7. 复现

```bash
cd shared/current_exp/20261122_global_k_native_dispatch

./reproduce.sh interface
./reproduce.sh cpu
./reproduce.sh old
./reproduce.sh resident
./reproduce.sh profile
./reproduce.sh analyze

# 全流程
./reproduce.sh all
```

主要交付：

- `artifacts/INTERFACE_AUDIT.json`
- `artifacts/DEVICE_TIMELINE_SUMMARY.json`
- `artifacts/PAIRED_TIMINGS.csv`
- `artifacts/SUMMARY.json`
- `artifacts/interface_source_snippets.txt`
- `artifacts/runtime_sources.sha256`
- `artifacts/model_metadata.sha256`
- `artifacts/environment.json`
- `reproduce.sh`
