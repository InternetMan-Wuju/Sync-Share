#!/usr/bin/env python3
"""Summarize the four-request staggered Chunk comparison from archived raw SSE only."""
import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path


def events(path):
    frames=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            entry=json.loads(line)
            raw=entry.get('raw_line','')
            if not raw.startswith('data:') or raw[5:].strip() == '[DONE]':
                continue
            obj=json.loads(raw[5:].strip())
            count=(obj.get('meta_info') or {}).get('completion_tokens')
            if isinstance(count,int) and count>0:
                frames.append((count,entry['recv_perf_ns'],entry['recv_wall_ns']))
    return frames


def quantile_nearest(vals,quantile):
    return sorted(vals)[math.ceil(len(vals)*quantile)-1] if vals else None


def fmt(v,digits=3):
    return f'{v:.{digits}f}' if isinstance(v,(float,int)) else '—'


def load_group(path):
    data={name:json.loads((path/'requests'/name/'result.json').read_text()) for name in 'ABCD'}
    ticks={name:events(path/'requests'/name/'sse_raw.jsonl.gz') for name in 'ABCD'}
    dispatch={r['name']:r for r in json.loads((path/'dispatches.json').read_text()) if r['submitted']}
    return data,ticks,dispatch


def gap_table(data,ticks,dispatch):
    details=[]
    for original,arrival in [('A','B'),('B','C'),('C','D')]:
        first=dispatch[arrival]['dispatch_perf_ns']
        last=data[arrival]['first_token_perf_ns']
        points=ticks[original]
        gaps=[(points[i-1][1],points[i][1],(points[i][1]-points[i-1][1])/1e6)
              for i in range(1,len(points)) if points[i][0]-points[i-1][0]==1]
        before=[g[2] for g in gaps if g[1]<=first]
        overlap=[g[2] for g in gaps if g[0]<last and g[1]>first] if last is not None else []
        after=[g[2] for g in gaps if last is not None and g[0]>=last]
        details.append({'decode_request':original,'incoming_prefill':arrival,
            'incoming_dispatch_perf_ns':first,'incoming_first_token_perf_ns':last,
            'before_count':len(before),'before_p50_ms':quantile_nearest(before,.5),
            'before_max_ms':max(before) if before else None,
            'overlap_count':len(overlap),'overlap_max_ms':max(overlap) if overlap else None,
            'after_count':len(after),'after_p50_ms':quantile_nearest(after,.5),
            'after_max_ms':max(after) if after else None,
            'incoming_server_prefill_s':(
               data[arrival]['meta_info']['prefill_finished_time']-
               data[arrival]['meta_info']['forward_entry_time']) if
               'prefill_finished_time' in data[arrival]['meta_info'] and
               'forward_entry_time' in data[arrival]['meta_info'] else None})
    return details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    a=ap.parse_args()
    root=a.root
    dataset=json.loads((root/'dataset.json').read_text())
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    rows=[]
    effects=[]
    traces={}
    for chunk in (8192,1024):
        path=root/f'node_23/chunk_{chunk}/bench'
        data,ticks,dispatch=load_group(path)
        for name,r in data.items():
            m=r['meta_info'];t=ticks[name]
            pre=m.get('prefill_finished_time',0)-m.get('forward_entry_time',0)
            gaps=[(t[i][1]-t[i-1][1])/1e6 for i in range(1,len(t))]
            rows.append({'chunk':chunk,'name':name,'prompt':r['prompt_tokens'],
              'cached':r['cached_tokens'],'hit_ratio':r['cached_tokens']/r['prompt_tokens'],
              'output':r['output_tokens'],'finish':r['finish_reason'],'error':r['error'],
              'queue_ms':m.get('queue_time',0)*1000,'server_prefill_s':pre,
              'ttft_s':r['ttft_s'],'first_residual_ms':
                 (r['ttft_s']-m.get('queue_time',0)-pre)*1000,
              'after_first_s':r['e2e_s']-r['ttft_s'],'tpot_ms':r['tpot_ms'],
              'e2e_s':r['e2e_s'],'itl_max_ms':max(gaps),
              'actual_output_ids_sha256':json.loads((path/'requests'/name/'output_integrity.json').read_text())['actual_output_sha256'],
              'num_retractions':m.get('num_retractions'),'decode_throughput':m.get('decode_throughput')})
            traces[f'{chunk}-{name}']={'starts_at':t[0][1],
                'gap_ms':gaps,'gap_right_perf_ns':[e[1] for e in t[1:]]}
        for effect in gap_table(data,ticks,dispatch):
            effects.append({'chunk':chunk,**effect})
    with (out/'requests.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    with (out/'interference.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=effects[0].keys());wr.writeheader();wr.writerows(effects)
    (out/'traces.json').write_text(json.dumps(traces,ensure_ascii=False)+'\n')
    doc=['# Chunk Size 8K / 1K：四条 32K 请求错开发送','',
      f'原始数据：`{root}`。数据集种子 {dataset["seed"]}；每条输入 32768、目标输出 300，固定公共前缀 {dataset["prefix_tokens"]} 个 ID，四个不同后缀；两组复用相同输入。两组各最多四条请求、发请求时间上限 30 分钟，已发请求等待完成。','',
      '本轮 A 输出至少 20 token 后发 B；B 输出至少 20 token 后发 C；A 完成、C 输出至少 20 token 后发 D。A/B/C 分别在 B/C/D 到来时作为 Decode 中的请求。','',
      '## 每请求记录','',
      '| Chunk | 请求 | 输入/命中/输出 | Queue(ms) | 服务端 Prefill(s) | TTFT(s) | 首 token 前剩余(ms) | 首 token 后(s) | TPOT(ms) | E2E(s) | 最长 ITL(ms) |',
      '|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        doc.append(f'| {r["chunk"]} | {r["name"]} | {r["prompt"]}/{r["cached"]}/{r["output"]} | {fmt(r["queue_ms"],2)} | {fmt(r["server_prefill_s"])} | {fmt(r["ttft_s"])} | {fmt(r["first_residual_ms"],1)} | {fmt(r["after_first_s"])} | {fmt(r["tpot_ms"],1)} | {fmt(r["e2e_s"])} | {fmt(r["itl_max_ms"],1)} |')
    doc+=['','## 输出序列与服务配置核对','',
      '| 输入 | 8K/1K 最终 300 个输出 ID 是否相同 |',
      '|:---:|:---:|']
    for name in 'ABCD':
        hashes=[r['actual_output_ids_sha256'] for r in rows if r['name']==name]
        doc.append(f'| {name} | {"是" if len(hashes)==2 and hashes[0]==hashes[1] else "否；不能假定 Decode 内容相同"} |')
    doc+=['', '启动命令请求 Decode CUDA graph `full`（bs 8/16/24/32），但固定镜像在 DeepEP normal 模式日志中报告 graph 被禁用；须以各组 `server.log` 的实际记录和 `npu graph` 运行日志为准，不能将“已传参数”写成“实际开启”。', '']
    doc+=['','## 每次注入的 ITL（客户端流式事件）','',
      '| Chunk | 正在 Decode | 后来 Prefill | 注入前 ITL 中位(ms) | 注入前最大(ms) | 注入窗口最大(ms) | 窗口之后 ITL 中位(ms) | 窗口内间隔数 |',
      '|---:|:---:|:---:|---:|---:|---:|---:|---:|']
    for r in effects:
        doc.append(f'| {r["chunk"]} | {r["decode_request"]} | {r["incoming_prefill"]} | {fmt(r["before_p50_ms"],1)} | {fmt(r["before_max_ms"],1)} | {fmt(r["overlap_max_ms"],1)} | {fmt(r["after_p50_ms"],1)} | {r["overlap_count"]} |')
    doc+=['','**口径**：窗口从后来请求发起至客户端收到其首 token，含发送、排队、Prefill、首 token 交付，不是纯 Prefill 阶段。ITL 间隔只要与该窗口相交便计入“注入窗口”；同一条边界间隔可能跨窗口。服务端 Prefill 来自 SSE `meta_info` 的两个时间戳；Queue 来自同一 meta；TTFT/E2E/ITL 来自客户端 SSE 时间。剩余时间是差值，不等于 IPC；首 token 后不是纯 NPU Decode。每个 Chunk 仅四条，P95/P99 没有可靠尾延迟意义。','',
      '各请求的完整 input IDs、raw SSE 与 output IDs、服务器日志、`launch.sh`、显存和 `/metrics` 采样都在原始目录，CSV 只用于快速查看。', '']
    (out/'README.md').write_text('\n'.join(doc))
    print(out/'README.md')

if __name__=='__main__':
    main()
