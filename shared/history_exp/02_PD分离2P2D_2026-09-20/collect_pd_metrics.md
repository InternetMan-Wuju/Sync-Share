# 测试4 服务端指标采集流程

压测脚本（`bench_pd_prefix.py`）只测客户端的 TTFT/TPOT/E2E/token。这里补上
服务端三类指标：**四段分解**、**EPLB cost**、**Expert×Layer 热力图**。

## 指标与来源

| 指标 | 来源 | 工具 |
|---|---|---|
| 四段分解（queue/prefill/中间/decode） | Prometheus `/metrics` | `./breakdown_ttft.py`（本目录） |
| EPLB cost（rebalance 次数/每次耗时） | decode 服务日志 | `../glm52_ep64_chengyetai/ep_lab/parse_rebalance_timing.py` |
| Expert×Layer 热力图 | expert_distribution.pt | `../glm52_ep64_chengyetai/ep_lab/analyze_expert_heat.py`（容器内跑） |

> PD 分离下，EP / EPLB / expert recorder 都在 **decode 侧**；prefill 的指标
> （queue、prefill_forward）在 **prefill 侧**。具体哪个 metric 落在哪个容器需首次实测确认，
> 采集时两个容器都采一份最稳。

## 1. 四段分解

每个 cell 在压测前后各采一次。两个 prefill 节点合并后参与统计：

```bash
curl -s http://10.120.72.23:18423/metrics -o prefill0_before.prom
curl -s http://10.120.72.25:18423/metrics -o prefill1_before.prom
cat prefill0_before.prom prefill1_before.prom > prefill_before.prom
curl -s http://10.120.72.31:18423/metrics -o decode_before.prom

# ... 跑压测 ...

curl -s http://10.120.72.23:18423/metrics -o prefill0_after.prom
curl -s http://10.120.72.25:18423/metrics -o prefill1_after.prom
cat prefill0_after.prom prefill1_after.prom > prefill_after.prom
curl -s http://10.120.72.31:18423/metrics -o decode_after.prom

python3 ./breakdown_ttft.py \
  --prefill-before prefill_before.prom --prefill-metrics prefill_after.prom \
  --decode-before decode_before.prom --decode-metrics decode_after.prom \
  --out breakdown.json
```

必须使用 before/after histogram 差分；直接读取 after 累计值会混入启动、预热和前序 cell。

分解口径（对齐 `TTFT残差分析.md`）：

- queue = `sglang:queue_time_seconds`
- prefill = `sglang:per_stage_req_latency_seconds{stage="prefill_forward"}`
- 中间 = TTFT − queue − prefill（tokenizer↔scheduler 的 IPC）
- decode = e2e − TTFT

## 2. EPLB cost

从 decode 容器日志抓 rebalance 时序：

```bash
sudo docker logs pd-decode-glm52 2>&1 | grep -E "rebalance (start|end)" > rebalance.log
python3 ../glm52_ep64_chengyetai/ep_lab/parse_rebalance_timing.py --log rebalance.log --out rebalance_cost.json
```

产出：rebalance 次数、每次耗时、total/mean/max/min。**日志千万别删**。

## 3. Expert×Layer 热力图

```bash
# 压测前，在 decode 侧开启 expert distribution recorder
curl -s -X POST http://10.120.72.31:18423/start_expert_distribution_record

# ... 跑 bench_pd_prefix.py ...

curl -s -X POST http://10.120.72.31:18423/stop_expert_distribution_record
curl -s -X POST http://10.120.72.31:18423/dump_expert_distribution_record

# 把 recorder 的 .pt 从 decode 容器拷出来
sudo docker cp pd-decode-glm52:/tmp/expert_distribution_recorder_<最新>.pt ./expert_distribution.pt

# 在容器内跑分析（宿主机无 torch）
sudo docker cp ../glm52_ep64_chengyetai/ep_lab/analyze_expert_heat.py pd-decode-glm52:/tmp/
sudo docker exec pd-decode-glm52 python3 /tmp/analyze_expert_heat.py \
  --pt /tmp/expert_distribution_recorder_<最新>.pt --out /tmp/heat.json
sudo docker cp pd-decode-glm52:/tmp/heat.json ./heat.json
```

产出：全局专家热度、逐层热度、时刻快照、rank 负载、不均衡三件套（CV/peak/mean/max/min）。

## 完整流程（每档实验）

```text
1. 起 2P2D 服务 + router
2. 健康检查（router /health == 200）
3. decode 侧 start_expert_distribution_record
4. 跑 bench_pd_prefix.py（预热 + 并发压测）
5. decode 侧 stop + dump expert_distribution_record
6. 采 metrics（prefill + decode）
7. 抓 decode 日志里的 rebalance start/end
8. 分析：breakdown_ttft.py / parse_rebalance_timing.py / analyze_expert_heat.py
9. 产物齐全检查，缺一停下
```
