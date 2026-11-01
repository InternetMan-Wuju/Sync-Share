# 测试4 结果：PD 分离部署（GLM-5.2 W8A8）

## 结论速览

- **PD 分离在本集群跑通了**：4 机 2P2D（2 prefill + 2 decode + 1 router），端到端 `/generate` 与 `/v1/chat/completions` 都能正常出结果。
- **前缀缓存（radix cache）在 prefill 侧有效**：压测中 `cached-token` 持续命中，首请求 prefill 完共享前缀后，后续请求只 prefill 尾部。
- **但离参考测试4 的目标差距很大**，三个硬限制（见下）导致 64K/128K 长上下文 + TPOT≤30ms 的参考成绩在本集群当前拓扑下无法复现。

## 三个硬限制（关键发现）

### 1. 单机 TP16 的 KV 池只有 8 万 token，撑不起 64K/128K 高并发

- GLM-5.2 W8A8 权重约 721 GiB（181 个分片），16 卡 × 64GB 上每卡权重占 47.2GB。
- `--mem-fraction-static 0.90` 下 KV 池 `max_total_num_tokens=80000`（8.2GB/卡），`context_len=32768`。
- 参考测试4 要 64K/128K 输入、并发 10/20/30。即使前缀命中 90%，单 prefill 也要缓存约 `0.9L + N*0.1L` 个 token：
  - 64K 输入、并发 30：约 57.6K + 30×6.4K ≈ 250K token，远超 80K。
  - 所以本轮只能跑到 **8K 输入** 这一档（参考的一半以下）。
- 要贴近参考，需要把 prefill 从「单机 TP16」改成「2 机 TP32」甚至更多，让单侧 KV 池翻倍以上。

### 2. Ascend PD 的 KV 传输后端 `ascend` 只有 sdma 可用，device_rdma 初始化失败

- `--disaggregation-transfer-backend ascend` 走 `memfabric_hybrid.TransferEngine`，传输协议由 `ASCEND_MF_TRANSFER_PROTOCOL` 决定。
- `sdma`（默认）：短请求 OK，但跨机长 KV 传输非常慢（观察日志里 prefill 进入 inflight 后长时间 128 token/批 地推进）。
- `device_rdma`：本机有 RoCE 设备 `rocep162s0f0` / `rocep162s0f1`（10.120.73.x），但设置 `device_rdma` 后
  `Ascend Transfer Engine initialization failed`（transfer_engine.py:84，RDMA 路径的 HCCL all_gather 初始化失败）。
- 参考测试4 的 64K/128K + TPOT≤30ms 依赖 device RDMA；本集群当前无法直接开。

### 3. Decode 开启 CUDA Graph 会卡死，只能禁用，导致 decode 很慢

- decode 加 `--cuda-graph-max-bs 32` 后，capture 阶段卡在
  `NPUGraph.cpp:223 Waiting for pending NCCL work to finish before starting graph capture`（数分钟不结束）。
- 所以只能 `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`。
- 禁用 graph 后 decode 明显变慢：5 并发、8K 上下文时约 **18 token/s/instance**（2 个 decode 合计约 36 token/s），
  每 token 约 270ms，远高于参考的 TPOT≤30ms。

## 部署拓扑与端口

| 角色 | 节点 | TP | 端口 |
|---|---|---|---|
| MemFabric store | 10.120.72.23 | - | 24727 |
| prefill 0 | 10.120.72.23 | 16 | 18423（bootstrap 18998） |
| prefill 1 | 10.120.72.25 | 16 | 18425（bootstrap 18998） |
| decode 0 | 10.120.72.31 | 16 | 18431 |
| decode 1 | 10.120.72.32 | 16 | 18432 |
| router | 10.120.72.23 | - | 18623 |

- 模型：`/mnt/raid/user/data/models/glm-q`（GLM-5.2，GlmMoeDsaForCausalLM，256 routed experts + 1 shared，78 层）。
- 镜像：`quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16`（sglang 0.5.17.dev386）。
- 传输：`--disaggregation-transfer-backend ascend`，`ASCEND_MF_TRANSFER_PROTOCOL=sdma`。

## 启动脚本（本轮新增，均在 `fixed_tests/test4_pd/`）

- `serve_pd.sh`：单机 TP16 prefill/decode 启动（参数化 CTX / MAX_REQ / 传输协议 / CUDA graph）。
- `start_pd.sh`：4 机编排（store + 2 prefill + 2 decode + router）。
- `stop_pd.sh`：一键停止。
- `bench_pd.py`：随机 token-id + 共享前缀 + 流式计时（TTFT / TPOT / e2e / tokens/s）。
- `run_test4.sh`：并发 10/20/30 三档，`input=8192 / output=1024 / prefix_ratio=0.9 / requests=并发×4`。

## 压测结果（input=8192, output=1024, prefix=0.9, requests=并发×4）

结果文件在 `results/test4_in8192_conc*.json`。decode 无 CUDA graph，跑得较慢。

### 并发 10（已出首批 10 条）

| 指标 | 值 |
|---|---|
| e2e | 287 ~ 292 s/请求 |
| TTFT | 15.1 ~ 19.6 s |
| 每请求输出 | 1024 token |
| 真实 TPOT（推导） | (e2e - TTFT) / (1023) ≈ **265 ms/token** |
| 两 decode 合计生成吞吐 | ≈ 38 token/s（19 token/s × 2） |

> 说明：`bench_pd.py` 里基于 SSE 事件间隔算的 `tpot=0.52ms` 是错的（`/generate` 在 `ignore_eos` 下把 token 攒批回传），
> 正确 TPOT 用 `(e2e - TTFT)/(out-1)` 推导。后续会把脚本里的 TPOT 改为这个口径。

（并发 20 / 30 跑完后回填。）

## 与参考测试4 的对照

| 项 | 参考测试4 | 本轮 |
|---|---|---|
| 模型 | GLM-5.1 W8A8 | GLM-5.2 W8A8 |
| 拓扑 | 4 节点 A3，2 prefill + 2 decode | 4 节点 910C，2 prefill + 2 decode |
| 输入 | 64K / 128K | 8K（受 KV 池限制） |
| 输出 | 1K | 1K |
| 前缀缓存 | 90% | 90%（已验证命中） |
| 传输 | device RDMA | sdma（device_rdma 初始化失败） |
| TPOT | ≤30ms | 约 270ms（无 CUDA graph） |
| 结论 | 可复现框架，数值不可比 | 框架跑通，长上下文目标未达成 |

## 复现步骤

```bash
cd /home/chengyetai/workspace/glm52_ep64_chengyetai/fixed_tests/test4_pd
# 1) 启动（会自动先停掉旧的 EP64）
bash start_pd.sh
# 2) 等 4 个实例 "fired up and ready to roll" 后，重启 router（见 start_pd.sh 第 6 步）
# 3) 压测
INPUT=8192 bash run_test4.sh
# 4) 停止
bash stop_pd.sh
```
