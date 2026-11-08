# 启动与修正记录

- attempt1：加载期间主动停止，加入同一真实层输入的16rank交错复测入口与更完整的padding检查；未产生有效性能样本。
- attempt2：权重加载后初始化算子时失败，`ModuleNotFoundError: No module named tbe`。原因是server.sh覆盖了CANN set_env.sh提供的PYTHONPATH。保留失败证据startup_failure_tbe.txt；原始日志在本地local/server_attempt2_tbe_failure.log。
- attempt3：外挂路径改为前置并保留CANN原路径；新增在完整加载前执行的真实NPU算子预检，已输出NPU_PREFLIGHT_PASS。后续结果见RESULTS.md。

这些失败/加载时间不算性能样本；不能只提交成功数据而隐去启动问题。所有失败保留原日志，未修改系统安装。

- 正式A/B前，revision2把同输入基线输出显式clone为独立存储，避免后端可能复用输出buffer造成伪等价。停止了尚处于预热阶段的pilot suite，保留全部pilot日志；随后三输入重新执行独立输出快照验证。未覆盖任何正式样本。
- 温度0的off重复请求也出现完整32token生成不一致。目标层同一hidden/top-k的独立快照对比与整模型跨请求输出一致性是不同验收，后者保留为未解决问题。
