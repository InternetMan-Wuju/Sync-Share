#!/usr/bin/env python3
"""Offline ChunkSize evidence without counting duplicated TP metrics 16 times."""
from report_labels import humanize_report
import json
import re
import statistics
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK=ROOT.parent/'Qchunksize'
LINE=re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)')
LABEL=re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def metrics(path):
    out=[]
    for line in path.read_text().splitlines():
        m=LINE.match(line)
        if m:
            try:v=float(m[3])
            except ValueError:continue
            out.append((m[1],dict(LABEL.findall(m[2] or '')),v))
    return out


def main():
    cells=[]
    for p in sorted((CHUNK/'results/autonomous_20261012').glob('*/result.json')):
        data=json.loads(p.read_text());directory=p.parent
        info=json.loads((directory/'server_info.json').read_text())
        timeline=[]
        for f in sorted((directory/'monitor').glob('*.prom')):
            values=metrics(f);gauges={}
            for name in ('sglang:num_running_reqs','sglang:num_queue_reqs','sglang:token_usage','sglang:full_token_usage'):
                matched=[v for n,labels,v in values if n==name]
                gauges[name]=max(matched) if matched else None
            timeline.append({'unix':int(f.stem),**gauges})
        peaks={name:max((v[name] for v in timeline if v[name] is not None),default=None)
               for name in ('sglang:num_running_reqs','sglang:num_queue_reqs','sglang:token_usage','sglang:full_token_usage')}
        cells.append({'name':directory.name,'scene':data['name'],'concurrency':data['concurrency'],
                      'chunk_requested':data['chunk'],'chunk_effective':info.get('chunked_prefill_size'),
                      'source':str(p),'dataset_sha256':data['dataset_sha256'],
                      'server':{k:info.get(k) for k in ('version','tp_size','dp_size','ep_size','mem_fraction_static',
                       'max_total_num_tokens','max_running_requests','context_length','deepep_mode','enable_mixed_chunk',
                       'disable_cuda_graph','speculative_algorithm','disable_shared_experts_fusion')},
                      'aggregate':data['aggregate'],'sampled_peaks':peaks,'timeline':timeline,
                      'request_retractions':sum(r['meta'].get('num_retractions',0) for r in data['requests'])})
    groups=[]
    for scene,conc in sorted({(c['scene'],c['concurrency']) for c in cells}):
        rows=[c for c in cells if (c['scene'],c['concurrency'])==(scene,conc) and
              c['aggregate']['complete'] and c['aggregate']['cache_condition_valid']]
        by_chunk={}
        for c in rows:by_chunk.setdefault(c['chunk_requested'],[]).append(c)
        candidates=[]
        for chunk,rs in by_chunk.items():
            speeds=[c['aggregate']['total_output_tok_s'] for c in rs]
            candidates.append({'chunk':chunk,'repeats':len(rs),'throughput_mean':statistics.mean(speeds),
                               'throughput_min':min(speeds),'throughput_max':max(speeds),
                               'ttft_p50_mean_s':statistics.mean(c['aggregate']['ttft_s_p50'] for c in rs),
                               'tpot_p50_mean_ms':statistics.mean(c['aggregate']['tpot_ms_p50'] for c in rs)})
        candidates.sort(key=lambda c:c['throughput_mean'],reverse=True)
        groups.append({'scene':scene,'concurrency':conc,'candidates':candidates,
                       'same_dataset_sha256':len({c['dataset_sha256'] for c in rows})==1})
    report={'cells':cells,'groups':groups,
            'scope':'Gauge peaks use max across replicated TP ranks, not sum. 10-second observations can miss shorter peaks. Stage wall spans contain scheduler gaps; counters do not imply pure kernel timing.'}
    (CHUNK/'ChunkSize负载与阶段分析.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# ChunkSize负载与阶段证据','',report['scope'],'',
           '| 单元 | 有效Chunk | KV容量token | 采样峰值running | 采样峰值queue | 采样峰值KV比例 | 请求回退次数 |',
           '|---|---|---|---|---|---|---|']
    for c in cells:
        g=c['sampled_peaks'];lines.append(f"| {c['name']} | {c['chunk_effective']} | {c['server']['max_total_num_tokens']} | {g['sglang:num_running_reqs']} | {g['sglang:num_queue_reqs']} | {g['sglang:full_token_usage']} | {c['request_retractions']} |")
    lines+=['','## E2E平均时间分解','',
            '下表使用同组请求的平均值，可相加核验。prefill与decode均是含调度间隙的墙钟跨度；queue_time单列，不再重复加进E2E。','',
            '| 单元 | 客户端E2E均值s | 接收到首次forward s | Prefill墙钟s | Decode至完成s | 客户端残差s | 独立queue_time s |',
            '|---|---|---|---|---|---|---|']
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    for c in cells:
        a=c['aggregate'];b=a.get('timestamp_breakdown_mean_s') or {}
        lines.append('| '+c['name']+' | '+fmt(a.get('e2e_s_mean'))+' | '+' | '.join(fmt(b.get(k)) for k in
                     ('server_before_forward_s','server_prefill_span_s','server_decode_and_finish_s',
                      'client_minus_server_e2e_s','reported_queue_s'))+' |')
    lines+=['','不同chunk的输入、实际服务开关、缓存比例和E2E阶段跨度在JSON中逐项保留。候选按重复轮次的平均总吞吐比较；没有重复的单元只作初测。']
    (CHUNK/'ChunkSize负载与阶段分析.md').write_text(humanize_report('\n'.join(lines)+'\n'))
    print('chunk evidence cells',len(cells),'groups',len(groups))


if __name__=='__main__':main()
