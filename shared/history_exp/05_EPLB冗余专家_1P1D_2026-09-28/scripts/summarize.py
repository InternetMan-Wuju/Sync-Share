#!/usr/bin/env python3
"""Consolidate only observed metrics and log events, preserving raw references."""
import argparse,collections,datetime,json,math,re,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('session',type=Path);args=ap.parse_args()

def quantile(values,p):
    if not values:return None
    xs=sorted(values);return round(xs[int((len(xs)-1)*p)],4)
def moments(values):
    if not values:return None
    return {'mean':round(statistics.mean(values),4),'p50':quantile(values,.5),'p95':quantile(values,.95),'p99':quantile(values,.99)}

rows=[]
for group in sorted(args.session.iterdir()):
    f=group/'bench/summary.json'
    if not f.exists():continue
    bench=json.loads(f.read_text()); requests=[json.loads(line) for line in (group/'bench/requests.jsonl').read_text().splitlines()]
    statuses=collections.Counter(x['status'] for x in requests)
    aud={}
    for node in (23,25):
        nodepath=group/f'node_{node}'
        logfile=nodepath/'docker.log';data=logfile.read_text(errors='replace') if logfile.exists() else ''
        aud[str(node)]={'log_bytes':len(data),'oom_events':len(re.findall(r'(?i)torch\.OutOfMemoryError:',data)),
            'rebalance_start_lines':len(re.findall(r'(?i)rebalance start',data)),
            'rebalance_end_lines':len(re.findall(r'(?i)rebalance end',data)),
            'graph_capture_end':len(re.findall(r'Capture target decode NPU graph end',data)),
            'expert_window_pt_count':len(list((nodepath/'eplb_windows').glob('*.pt'))),
            'expert_window_total_bytes':sum(p.stat().st_size for p in (nodepath/'eplb_windows').glob('*.pt'))}
    events=group/'metrics/events.jsonl';eventrows=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    metrics=collections.Counter((e.get('name'),e.get('endpoint'),str(e.get('status','error'))) for e in eventrows)
    args_file=group/'config.txt';config=args_file.read_text() if args_file.exists() else ''
    snapshots_file=group/'snapshots/snapshots.jsonl'
    snapshots=[json.loads(line) for line in snapshots_file.read_text().splitlines()] if snapshots_file.exists() else []
    valid_heat=[x for x in snapshots if x.get('status')=='ok' and x.get('raw_bytes',0)>0]
    ok=[r for r in requests if r['status']=='ok']
    result={'group':group.name,'config':config,'bench':bench,'request_statuses':dict(statuses),'actual_prompt_tokens':collections.Counter(str(x.get('prompt_tokens')) for x in ok),
        'actual_output_token_counts':dict(collections.Counter(x.get('output_tokens') for x in ok)),
        'cached_tokens':moments([x['cached_tokens'] for x in ok if isinstance(x.get('cached_tokens'),(int,float))]),
        'per_node_evidence':aud,'heat_snapshots':valid_heat,'heat_snapshot_errors':[x for x in snapshots if x.get('status')=='error'],'metrics_endpoints':{'|'.join(str(x) for x in k):n for k,n in metrics.items()}}
    rows.append(result)
(args.session/'consolidated.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
header=['# EPLB 1P1D 观测汇总','',f'生成时间：{datetime.datetime.now().astimezone().isoformat()}。本页仅汇总 `results` 的实际观测，解释以各组服务器日志为准。','',
 '| 组 | 状态 | EPLB | 冗余 | 输入/输出 | 实际墙钟 s | 输出 tokens/s | TTFT 均值/P95 s | TPOT 均值/P95 ms | E2E 均值/P95 s |','|---|---|---|---|---|---:|---:|---|---|---|']
for row in rows:
    b=row['bench'];c=dict(line.split('=',1) for line in row['config'].splitlines() if '=' in line)
    def fmtstat(field,mult=1):
        x=b.get(field)
        return f'{x["mean"]*mult:.2f} / {x["p95"]*mult:.2f}' if x else 'NA'
    header.append(f'| {row["group"]} | {b["completed"]}成功/{b["errors"]}错误/{b["incomplete"]}未满 | {c.get("eplb","?")} | {c.get("redundant","?")} | {b["input_tokens"]}/{b["output_tokens_target"]} | {b["actual_wall_s"]:.1f} | {b["output_tokens_per_s"]:.2f} | {fmtstat("ttft_s")} | {fmtstat("tpot_s",1000)} | {fmtstat("e2e_s")} |')
header += ['','统计口径：请求的 TTFT 是请求发出至首次含 token SSE；TPOT 为（E2E−TTFT）/(输出 token 数−1)，没有将它当成逐 token ITL。吞吐以完整窗口墙钟计算。',
 '每组原始请求 `bench/requests.jsonl`，每请求原始 SSE 为 `bench/sse/*.jsonl.gz`，后端 Prometheus 抓取及 recorder API 为 `metrics/`；`node_23`/`node_25` 中包含启动配置、权重补丁审计、专家窗口与服务器日志。','']
for row in rows:
    header += [f'## {row["group"]} 的后端证据','']
    for node,info in row['per_node_evidence'].items():
        header += [f'- 节点 {node}：OOM 堆栈 {info["oom_events"]}；rebalance start/end {info["rebalance_start_lines"]}/{info["rebalance_end_lines"]}；Decode Graph 捕获结束记录 {info["graph_capture_end"]}；专家窗口 .pt {info["expert_window_pt_count"]} 个（{info["expert_window_total_bytes"]} 字节）。']
    header+= [f'- 已抓取 metrics/API 响应：`consolidated.json` 中 `metrics_endpoints`；缓存命中 token 摘要：{row["cached_tokens"]}。','']
header += ['## 运行中专家热度快照', '', '每张图是该时点最近一个统计窗口的 MoE 层 × 逻辑专家路由数，色标为 log(1+count)；`.pt` 保留原始计数和专家放置映射。','']
for row in rows:
    header += [f'### {row["group"]}', '', '| 分钟 | Prefill 23 | Decode 25 |', '|---:|---|---|']
    valid = {(x['mark_seconds'], x['node']): x for x in row['heat_snapshots']}
    for t in (60, 150, 240, 330, 420, 510, 570):
        cells=[]
        for node in (23,25):
            x=valid.get((t,node))
            if x:
                image=Path(x['image']).relative_to(args.session)
                raw=Path(x['raw']).relative_to(args.session)
                cells.append(f'[PNG]({image}) / [原始 PT]({raw})，{x.get("steps", "?")} forwards')
            else:cells.append('未采到有效窗口')
        header.append(f'| {t/60:g} | {cells[0]} | {cells[1]} |')
    header += [f'有效快照 {len(row["heat_snapshots"])} 张；采集错误 {len(row["heat_snapshot_errors"])} 次；详见该组 `snapshots/snapshots.jsonl`。', '']
(args.session/'RESULTS.md').write_text('\n'.join(header))
print(args.session/'RESULTS.md')
