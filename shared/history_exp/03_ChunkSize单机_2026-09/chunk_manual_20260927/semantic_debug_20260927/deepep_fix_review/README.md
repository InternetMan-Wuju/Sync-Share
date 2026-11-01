# DeepEP 修复包只读检查

日期：2026-09-27。检查对象：`/mnt/raid/user/data/tools/bench3/kfix-build/deepep_fix_pkg/deep_ep`；比较对象为当前 `sglang-glm52-pd:fe694986` 镜像内 DeepEP，并核对 23 活跃容器的扩展和算子库摘要。未换包、未重启服务。

## 结论

该目录包含可作为外挂候选的完整 DeepEP 包，包含 Python 包、CPython 3.11 / AArch64 扩展、`libcust_opapi.so`、Ascend910_93 编译内核 `.o` 及对应清单。具有实际部署价值，但尚未验证当前服务中的运行效果。

## 实际修复内容

- 新增 `window_layout.h` / `A3WindowLayout`，为 normal 与 V2 low-latency 共用 HCCL 窗口时分离 notify、combine、selector、同步状态与数据区域。
- `CamMoeDispatchNormal`、`CamMoeCombineNormal`、multi-round combine、V2 dispatch/combine 均引入 `isHybridDeployment` 分支，混合模式采用新地址布局。
- 非混合分支仍保留 legacy 布局：normal 数据偏移为 102 MB + 4 MB = 106 MB；normal combine 状态区仍为 4 MB。因此应优先视为 hybrid/auto 场景的稳定性修复，不能仅凭存在 normal 算子改动就保证当前纯 normal 输出恢复。
- 候选还包含量化接口变化：Buffer.dispatch 的 `quant_mode` 入口改为 `use_fp8/use_mxfp4/use_mxfp8`；`utils.py` 仍保留 `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` 到 INT8 的回退。当前 SGLang 的 normal dispatch 调用没有传 `quant_mode`，现有环境变量因此初步可兼容。

## 外挂部署条件

- 原目录只读，部署时复制包到 `workspace/current_exp` 下，记录全包摘要；将整个 `deep_ep` 目录只读挂载到原安装位置，Python、C++ 扩展、算子库、内核和清单一并替换。
- 当前顶层 `deep_ep_cpp...so` 是指向 `deep_ep/deep_ep_cpp...so` 的符号链接，已核实整目录挂载可以覆盖实际扩展来源；上线后仍应记录实际 import 路径和摘要。
- 当前镜像的 distribution 元数据报告 `1.0.0+146153e5.cann.9.0.0.b250`。候选目录没有 dist-info，不能从目录自身确认 `2026.9.0.post2` 标签或精确提交；源码的 A3WindowLayout 改动与 PDF 描述相符。
- 候选 NPU 二进制、CANN 运行及语义效果尚未验证。

## 与当前重复输出的关系

DeepEP 包不会修复模型文件里已有的 NaN/Inf。当前量化副本的异常数值与通信问题需要分别归因。若做换包对照，应先在 25 保持同一模型挂载、TP16/EP16、Chunk8192、MTP OFF、normal 及其他环境参数，仅更换 DeepEP，发送同一短中文问题。该对照用于判断本次语义异常是否也受通信路径影响；真实回复未通过时，不能宣称换包已解决问题。

原始全包比较见 `comparison.json` 和 `source.diff`；候选相关源码见 `buffer.py`、`utils.py`、`device_info.py`、`window_layout.h`；摘要与核实状态见 `assessment.json`。
