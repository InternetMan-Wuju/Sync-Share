# 全组 K 槽服务修复与 DeepEP 真实改道

基线 `db1ce2d304709f717b86bc5b7ca1a5c3c7a30002`。本轮先跑通、算对。原版、原始 SGLang EPLB 加冗余、我方可配置 K 仍是三组正式对照。小规模按 die 验证，全量用两台机器的全部 32 个 die 组成同一个 EP32 组，三组各跑服务器 AgentX 的 8192、16384、32768 token。耗时和显存是后续优化的基线，验收不设提速门槛。[上一轮报告](../shared/current_exp/20261123_global_k_service_ep32/REPORT_FOR_REVIEW.md)

## 1. 先修各 rank 对装槽的决定

上一轮 `ensure_slots()` 中只有 owner 更新 `slot_meta`，收集到的 owner 状态只写入日志。同一层再次命中时，owner 跳过安装，源 rank 仍可能发送，其他 rank 仍进入 barrier。CPU 模拟已复现分支分歧。32K 的现场日志记录了三个 8192-token 批次，最后还剩 8192 token。要用同请求和各 rank 的进度来确认卡点。[装槽代码](../shared/current_exp/20261123_global_k_service_ep32/plugin/k_service.py) · [请求证据](../shared/current_exp/20261123_global_k_service_ep32/artifacts/EP32_SERVICE_EVIDENCE.json)

1. 在任何设备收发之前，由真实 owner 提供实际槽身份与状态，对本次用到的所有槽形成全组一致的决定。按固定顺序核对层、逻辑专家、slot ID、owner、generation、ready 和实际绑定存储，得到复用、重装或回退。命中时所有 rank 一起跳过对应安装，未命中时各方按同一计划加入所需通信。失败或无有效槽时在发起 dispatch 前作共同决定。安装完成后再发布新代际，避免拿日志里的状态当执行依据。
2. 槽的身份至少用 `(owner_rank, local_slot_id)` 表示。K 仍由配置指定，实际槽数之和等于 K。现有按 owner 索引会让同一 owner 的两个配置项共用一块存储。支持这种配置时分别验证存储和任务，首版若限定一 owner 一槽，在服务启动时明确校验并拒绝重复 owner。测试关闭路径、K=1、K=2 双槽同时使用和跨层复用。
3. 增加能抓住旧错误的回归。覆盖同层连续命中、一次跨层请求 q=0 后重回已驻留层、不同槽冷热混合、源与 owner 同 rank、零任务 rank、换层过期引用。检查各 rank 的通信顺序和状态，确认复制完成才 ready，持有当前租约直至相关设备工作完成。明确 Prefill、Decode 和混合 batch 的入口处理。同一调度步上的各 rank 必须一致，纯 Decode 走原路径。
4. 用上一轮同一份 K=2、32K 输入指纹重跑，先做隔离单请求，再复测原来的 8K→16K→32K 顺序。做 `verify=false` 的真实路径和独立 `verify=true` 的诊断对照。用有界超时记录请求、chunk、层、rank、槽代际和通信序号。覆盖装槽/复用、dispatch、计算、combine、最后一个 Prefill chunk、首个 Decode 和响应，能指出最早缺哪个完成事件。心跳和 HTTP 200 单独记录。跑完四个 chunk 并收到完整响应才记成功。

## 2. 随后把真实副本任务接进 DeepEP

复用上一轮 272 通信编号配一个真实槽的回放逻辑。保留 256 个模型逻辑专家和原主库，给 adapter 独立定义通信编号和真实槽映射。EP32 的一个可行起点是每 rank 8 个主专家、至多 1 个槽的 owner 布局，对应等宽 9 个通信位置、全组 288 个编号。只有配置的 K 个额外位置关联真实权重，逐 rank 核对占位、实际张量和分配字节。先对照目标镜像的 DeepEP NORMAL dispatch/combine 与 ModelSlim 接口，再决定最小接线。[已跑通的回放](../shared/current_exp/20261123_prefill_eplb_three_way/scripts/replay_ep16.py) · [服务原型](../shared/current_exp/20261123_global_k_service_ep32/plugin/k_service.py)

- 保留逻辑 top-k、每个分支的位置与 router 权重，在 dispatch 前把选中的任务改为副本通信编号。接收端用本次实际排序和分组，让主专家走主路径，副本使用当前有效的 W8A8 槽及量化参数，拼回同一次 dispatch 的返回顺序后走 combine。检查 shared expert 路径仍正确。
- 原位置要真正少算这些任务。用实际 dispatch 接收记录和计算计数，按请求、层、发送 rank、token、top-k 位置核对一对一的任务去向。至少验证一个发送 rank、主权重 rank、槽 owner 三者分开的用例，EP32 记录实际跨节点 activation 改道。预先选定能让两个槽同层都收到真实任务的 K=2 用例。显式 padding 和合法任务分开核对，真实任务不能落空编号。
- 小规模先跑关闭入口的原版对照，再测 K=1、K=2、q=0、q>0 和同槽跨层复用。数值验证用同输入的原版输出，记录每层误差与有限值。若首 token 有差异，核对最终累计 token ID、两个首选 logits 的差距和原版重复波动，说明差异落在何处。修正流式客户端对累计 `output_ids` 的处理，并与服务报告的完成 token 数核对。[旧客户端](../shared/current_exp/20261123_global_k_service_ep32/scripts/request_agentx.py)

## 3. 原始 EPLB 和 EP32 全量

- 原版可先复核健康和三档输入。原始 EPLB 按目标镜像的原生算法和更新器，用独立校准输入重排，EP32 从每层 +32 个原生冗余位置起步。先把 `storage_offset must be 0` 缩小到实际 send/recv 张量、内部格式和切片布局。若需要兼容修正，独立记录并保留原生算法，验证目标张量格式、权重和临时峰值。在线组保存完成事件、生效映射和实际执行位置。若在线仍失败，保留日志并继续其余组，静态布局只作为额外诊断数据。[在线故障证据](../shared/current_exp/20261123_global_k_service_ep32/artifacts/EP32_SERVICE_EVIDENCE.json)
- 全量三组共用模型、拓扑、输入指纹、chunk 设置、并发、缓存和输出配置。服务器已有 AgentX 数据优先。保留上一轮 32K 输入作故障回归，其 32768 token 来自同 trace 的两个 16K 片段；有足够长的单条源请求时，优先用于正式 32K 对照。只有长度与块哈希时沿用确定性生成，标明 AgentX 派生方式。记录 32 rank 实际参与每次验证、各档真实 token 数和启用的层。
- 三组各覆盖 8K、16K、32K。验收依据是任务真正改道、无遗漏或重复、输出正确且完整响应。K=2 可作为我方主对照，报告实际 K 和 owner 分配。先做正确性，再给同边界的 TTFT、装槽/重排耗时和显存。正式测量关闭额外参考 MoE 和大张量同步审计。正负性能结果都如实交付。

## 4. 复现与交付

新产物放在 `shared/current_exp/20261123_global_k_native_service_dispatch/`，提交补丁、复现入口和一份 `REPORT_FOR_REVIEW.md`。每次复现使用独立 run ID。验收从显式指定的原始请求、各 rank 阶段和实际槽/任务日志重新计算，旧矩阵仅供历史对照。报告给出 32K 挂起是否复现或修复、哪一段有完成证据、DeepEP 改道是否实际发生、在线 EPLB 是否完成，以及三组各档状态。受阻项附具体失败配置、日志和复现命令。完成后清理本轮资源，提交、推送并回报 commit 和报告路径。
