# 外挂实现与复现

原模型和SGLang安装保持只读。复用前一run的layout、审计权重加载器及plugin/runtime.py；新增decision.py和decision_experiment.py，经experiment.execute的显式decision_*模式进入。默认control=off，所有实验模式必须test_only。撤销时停止隔离容器，取消外挂挂载即可；实际槽布局需要与启动配置一致。

## 执行入口

1. 先运行`python3 tests/test_decision.py`及`python3 scripts/simulate.py`，后者只做旧8K锚定的CPU外推。
2. 复用本机rootless Docker及既有镜像，不升级依赖。必要时启动独立daemon：`XDG_RUNTIME_DIR=/tmp/chengyetai-rootless-docker dockerd-rootless.sh --host unix:///tmp/chengyetai-rootless-docker/docker.sock --data-root /data_lib/cyt/.rootless-docker/data --exec-root /data_lib/cyt/.rootless-docker/exec`。
3. `bash scripts/launch.sh`只使用本机0–15设备。保持已有审计manifest与只读权重挂载，见server.sh。完整模型加载约十几分钟。
4. 容器内运行request_decision.py。首次复现应先以case0、split=train、scales=1,2,4,8完成开发标定，再在宿主运行fit_model.py冻结模型，然后以case1/2、split=test执行留出验证。
5. 宿主运行analyze_decision.py整理原始记录。run_study.py是**本轮失败后续接入口**：已有8K/16K开发记录时仅补case0 scales=4,8，再冻结和测试。不要在全新目录误把它当完整首跑流程；START.json截止时间也属于本次运行，重现必须另建run和新时间盒。

输入ID保存在被忽略的local/inputs.json，来源摘要为INPUTS.json。可用prepare_inputs.py和同revision本地Arrow重建；不可把真实权重、hidden、大型dump或凭据提交远端。

## 动作与模型

模型参数只从开发输入标定，Frozen JSON在测试前写定。决策候选至多两个热点、三个已标定目的rank、一次动作，所有代价以ms计。复制peer仅允许本轮已测2→0/1/9；未知成本和超出标定规模拒绝。

当前单次dispatch仅验证到每rank1024行。更大负载由相同16K块顺序原生调用构成；各对照都计clone/concat，完整批次路由在复制前已存在。为了不把不同块瓶颈误合并，分块决策只评分全迁出候选，模型按块累计激活槽/固定成本。半配额仍做数值验证，但不纳入分块成本模型拟合。

完整状态仍通过ND打包、HCCL发送/接收、目的NZ bank安装、设备同步和全组ready。通信资源在验证和预热中初始化，首次peer建立不进入稳态cold结果。计数、精确来源配额及分流复用已有实现；off的备用槽任务为零。

## 版本与失败记录

最初8K/16K开发数据由artifacts/measured_decision_experiment_v1.py执行；此版本32K原生dispatch失败。最终plugin/decision_experiment.py增加相同分块和日志修正，完成其余开发及全部留出测试。旧decision.py也按实测SHA还原保存为artifacts/measured_decision_pre_chunks.py。各rank保存源码SHA；不要把未执行的后续编辑冒充已测版本。

初版half测试的plan.new_loads沿用了全迁出元数据；实际执行使用的是half的q，slots与quotas正确并通过输出验证。原始记录未覆盖；最终版已修正此元数据，模型只使用实际slots且排除half，不受该字段影响。

初始默认/显式nofile及并行提前ACL初始化都有失败；artifacts/STARTUP_FAILURE.txt保存摘要，完整日志在local/server_failed_*.log。最终为65535文件句柄上限、每rank依次提前初始化ACL、TE_PARALLEL_COMPILER=1。串行初始化只在模型构造阶段执行，不在性能路径内；根因仍未完全定位。

初始化改动也仅在外挂wrapper中。此前8K/16K执行的runtime快照为artifacts/measured_runtime_parallel_init.py；最终版增加CPU组同步的依次初始化。它不改变量化权重、原专家计算或后续复制协议。

## 当前未完成

生产开关、graph/overlap支持、真实长文本/多请求分布、完整请求加速、通信增量的独立模型、跨层共享备用空间、32K原生dispatch错误根因，都未完成。本次交付的是经过本机16NPU验证的实验外挂与决策证据。
