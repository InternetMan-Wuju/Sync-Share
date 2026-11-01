# MTP / 推测解码：双节点三场景 45 分钟对照

本轮使用节点 23（MTP OFF）和 25（MTP ON），分别为单机 TP16/EP16，并行比较。主变量为 `--speculative-algorithm NEXTN`（草稿 3 token，验证最多推进 4 token）；两端固定相同镜像、模型、只读权重加载外挂、`--chunked-prefill-size 16384`、请求输入数据、`temperature=0` 和 `ignore_eos=true`。服务脚本请求默认 Decode Graph full/bs 8,16,24,32 与 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`，以两端实际日志判定是否生效。启动前原有服务和设备状态归档在 `results/<run_id>/before/`。

三组实际长度：输入/目标输出分别 `1024/32768`、`8192/1024`、`32768/300`。第三组先清理缓存，再预热共用的 19661-token 前缀；32K 是**总输入**，目标缓存命中约 60%，以服务器 `cached_tokens / prompt_tokens` 为准。其余两组每次清缓存以固定缓存状态。固定随机 token ID 数据集及种子见 `results/<run_id>/dataset.json`；随机 ID 只用于测量，真实语言回复另行预检。

每个场景同时在两台机器上运行 **45 分钟**，每台顺序提交请求、最大在途 1 条，分别保存逐请求 raw SSE、双时钟逐事件时间、完整 input IDs、最终/截断时 output IDs、服务器 `meta_info`、完整 `/metrics` 和 NPU 采样。`1K/32K` 完整输出可能远超过 45 分钟，截止时主动结束正在生成的请求并保存已有输出和接受统计，标记 `partial_at_45m_deadline=true`，不把部分输出视为完成的 32768 token。任何非正常错误单独记录，不能算进成功样本。接受长度口径为该请求**实际输出 token 数 / `spec_verify_ct`**，包含一次验证额外产生的 token；接受率为 `spec_num_correct_drafts / spec_num_proposed_drafts`。

启动与只读外挂：`serve_verified.sh`、`serve.sh`；真实语义检查：`chat_once.py`；小规模 1K/64 统计预检：`probe.py`；完整 45 分钟采集：`bench.py`。启动脚本、镜像及代码哈希、模型配置、只读权重来源、真实回复、原始事件、Prometheus 抓取和分析结果均归档在 `results/<run_id>/`。当前 run_id 保存在 `run_id.txt`。两机器的工作目录不实时共享：25 节点服务日志需在运行结束后同步回 23 节点同路径。
