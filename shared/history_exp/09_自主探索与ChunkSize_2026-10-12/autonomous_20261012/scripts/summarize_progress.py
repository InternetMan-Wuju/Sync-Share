#!/usr/bin/env python3
"""CPU reports of completed autonomous cells; missing results remain missing."""
from report_labels import humanize_report
import csv
import hashlib
import json
import statistics
from pathlib import Path
from audit_mtp_workload_coverage import main as audit_mtp_workload_coverage
from audit_chunk_requirement_coverage import main as audit_chunk_requirement_coverage

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
Q1=ROOT.parent/'Q1MTPExp';CHUNK=ROOT.parent/'Qchunksize'


def read(p):return json.loads(p.read_text())
def n(v,d=3):return '—' if v is None else f'{v:.{d}f}'


def summarize(rows):
    ok=[r for r in rows if not r.get('error') and r.get('meta',{}).get('spec_verify_ct')]
    v=sum(r['meta']['spec_verify_ct'] for r in ok)
    if not v:return {}
    h=[0]*4
    for r in ok:
        hist=r['meta'].get('spec_correct_drafts_histogram') or r['meta'].get('spec_accept_histogram') or []
        for i,x in enumerate(hist):h[i]+=x
    correct=sum(r['meta']['spec_num_correct_drafts'] for r in ok)
    proposed=sum(r['meta']['spec_num_proposed_drafts'] for r in ok)
    assert sum(h)==v and sum(i*x for i,x in enumerate(h))==correct and proposed==3*v
    return {'AL':sum(r['completion_tokens'] for r in ok)/v,'rate':correct/proposed,
            'positions':[sum(h[i:])/v for i in (1,2,3)],'accepted_stat_requests':len(ok),
            'periodic_requests':sum(bool(r.get('output_shape',{}).get('periodic_suffix')) for r in ok)}


def main():
    rows=[]; invalid_groups=[]
    for p in sorted((ROOT/'results').glob('*/*/result.json')):
        info_path=p.parent.parent/'server_info.json'
        if info_path.exists() and not json.loads(info_path.read_text()).get('speculative_algorithm'):continue
        d=read(p)
        validation=d.get('validation',[])
        if any(v.get('failures') for v in validation) or any(r.get('error') or not r.get('stream_done') for r in d['requests']):
            invalid_groups.append({'source':str(p),'validation':validation,'reason':'incomplete or failed request; not an acceptance result'})
            continue
        a=summarize(d['requests']);rows.append({'run':p.parent.parent.name,'group':p.parent.name,'source':str(p),**a})
    # Exact same-input paired records; never infer prefix-window acceptance from totals.
    comparisons=[]
    latest=read(Q1/'results/latest_native_full.json');before=Path(latest['out'])
    for group in ('w1_8k_1k','w2_32k_1k','w4_prefix32k_300','native_exact_1k256'):
        after=ROOT/'results/publisher_v19'/group/'result.json'
        bp=(Q1/'results/native_randomids_20261012_1844/native_exact_ids.json') if group.startswith('native_exact') else before/group/'result.json'
        if not after.exists():continue
        a=read(after);b=read(bp);sa=summarize(a['requests']);sb=summarize(b['requests'])
        old={r['index']:r for r in b['requests']};paired=[]
        for r in a['requests']:
            o=old[r['index']];ai=r['output_ids'];bi=o.get('output_ids') or []
            mismatch=next((i for i,(x,y) in enumerate(zip(ai,bi)) if x!=y),None)
            old_sha=o.get('input_sha256') or o.get('input_ids_sha256')
            if not old_sha and o.get('payload_file'):
                payload_path=Path(o['payload_file'])
                if not payload_path.is_absolute():payload_path=Path('/home/chengyetai')/payload_path
                if payload_path.exists():
                    payload=read(payload_path)
                    old_sha=hashlib.sha256(json.dumps(payload['input_ids'],separators=(',',':')).encode()).hexdigest()
            paired.append({'index':r['index'],'first_output_difference':mismatch,
                           'same_input_sha256':old_sha==r.get('input_ids_sha256') if old_sha else None,
                           'before_input_sha256':old_sha,'after_input_sha256':r.get('input_ids_sha256'),
                           'same_output_ids':ai==bi,'before_distinct':len(set(bi)),'after_distinct':len(set(ai)),
                           'before_periodic_suffix':o.get('output_shape',{}).get('periodic_suffix'),
                           'after_periodic_suffix':r.get('output_shape',{}).get('periodic_suffix')})
        comparisons.append({'group':group,'before':sb,'after':sa,'before_source':str(bp),'after_source':str(after),'paired_outputs':paired})
    lines=['# 自主MTP探索：阶段实测结果','',
           '场景和服务配置已用中文说明，长度单位均为token。原代号及缩写见[名称说明](负载与配置名称说明.md)。','',
           '权重恢复依据公开量化源SHA256。以下仅纳入已完成请求，失败启动另列；不是同条件速度收益榜单。','',
           '| 服务/轮次 | 负载 | AL | 草稿接受率 | P1/P2/P3 | 主要计数请求 | 检出周期尾段请求 | 来源 |',
           '|---|---|---|---|---|---|---|---|']
    for r in rows:
        pos='/'.join(n(v,4) for v in r.get('positions',[])) or '—'
        lines.append(f"| {r['run']} | {r['group']} | {n(r.get('AL'))} | {n(r.get('rate'),4)} | {pos} | {r.get('accepted_stat_requests','—')} | {r.get('periodic_requests','—')} | [原始结果]({r['source']}) |")
    lines+=['','不完整/校验失败组不纳入上表：'+str(len(invalid_groups))+'组；现场路径与失败校验保存在JSON中。']
    lines+=['','## 权重对照与输入证据','',
            '| 场景 | 原始副本AL | 公开源校验版AL | 原始副本草稿率 | 校验版草稿率 | 逐请求输入SHA一致 |',
            '|---|---|---|---|---|---|']
    for r in comparisons:
        matched=sum(x['same_input_sha256'] is True for x in r['paired_outputs'])
        total=len(r['paired_outputs'])
        evidence=f'{matched}/{total}' if matched==total else f'{matched}/{total}；其余未留足证据'
        lines.append(f"| {r['group']} | {n(r['before'].get('AL'))} | {n(r['after'].get('AL'))} | {n(r['before'].get('rate'),4)} | {n(r['after'].get('rate'),4)} | {evidence} |")
    lines+=['','1K/32K旧结果与新1K/1024短探针长度不同，不用总AL作同条件对照。逐请求完整输出、EOS和周期形状均保存。',
            '正式短负载对照保持镜像/TP/EP/DP/MEM/chunk/max-prefill/MRR/context/MTP参数一致。旧启动声明Graph但normal下实际graph显存为0，新启动显式关闭；两者均无图执行。草稿路径视图是加载组织差异，不是另一套草稿权重。逐请求输入SHA核验结果见JSON。',
            'native_exact旧短探针没有保存逐请求payload或输入SHA，故该行仅列历史数字，不作为严格同输入权重因果证据。前三组正式短负载的输入SHA全部一致。',
            '允许EOS请求有时在prefill即结束，没有verify，位置率留空；未将缺计数当成0。',
            '草稿使用同一checkpoint第78层的7个原始分片视图，仅减少加载扫描量；原NextN加载和计算实现未修改。','']
    state=read(ROOT/'overnight_status.json') if (ROOT/'overnight_status.json').exists() else {}
    lines+=['## 服务配置与失败记录','', '| 配置 | 健康 | 请求结果 | 参数 | 日志 |','|---|---|---|---|---|']
    for r in state.get('runs',[]):
        logfile=ROOT/'logs'/r['container']/'server.log'
        lines.append(f"| {r['tag']} | {r.get('healthy')} | {r.get('suite_rc',r.get('failure','待运行'))} | {' '.join(r.get('flags',[]))} | [日志]({logfile}) |")
    final_states=[read(ROOT/name) if (ROOT/name).exists() else {}
                  for name in ('overnight_status.json','chunk_repeat_status.json','final_mtp_isolation_status.json')]
    completion_text=('当时编排的基础网格、候选复测与最终MTP单变量诊断均已结束；这不表示AboutModels四组要求已完成。具体覆盖见文首核对。'
                     if all(s.get('completed') for s in final_states) else
                     '仍有阶段在推进；尚无结果的配置不能写成已验证。')
    lines+=['','基础阶段：'+state.get('phase','未知')+'。'+completion_text]
    (ROOT/'MTP自主探索实测.md').write_text(humanize_report('\n'.join(lines)+'\n'))
    (ROOT/'MTP自主探索实测.json').write_text(json.dumps({'rows':rows,'invalid_groups':invalid_groups,'weight_comparisons':comparisons,'state':state},ensure_ascii=False,indent=2))
    audit_mtp_workload_coverage()
    cells=[]
    for p in sorted((CHUNK/'results/autonomous_20261012').glob('*/result.json')):
        d=read(p);a=d['aggregate'];cells.append({'scene':d['name'],'chunk':d['chunk'],'concurrency':d['concurrency'],
                                               'source':str(p),'mtp_on':d['mtp_on'],'repeat':d.get('repeat',1),**a})
    lines=['# ChunkSize固定并发实验：实测表','',
           '场景按输入/输出token数命名，并发为客户端同时在途的请求上限。原代号见[名称说明](../autonomous_20261012/负载与配置名称说明.md)。','',
           '23单机、公开源校验权重、原生random-ids；同场景/并发下输入固定。MTP关闭；同一矩阵服务参数除chunk外固定。缓存命中和原始日志单列。','',
           '| 场景 | 并发 | Chunk | 轮次 | 成功/请求 | 总输出tok/s | TTFT p50 s | TTFT p95 s | TPOT p50 ms | TPOT p95 ms | E2E p50 s | E2E p95 s | 命中率 | 来源 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in cells:
        lines.append(f"| {r['scene']} | {r['concurrency']} | {r['chunk']} | {r['repeat']} | {r['requests_ok']}/{r['requests']} | {n(r.get('total_output_tok_s'))} | {n(r.get('ttft_s_p50'))} | {n(r.get('ttft_s_p95'))} | {n(r.get('tpot_ms_p50'))} | {n(r.get('tpot_ms_p95'))} | {n(r.get('e2e_s_p50'))} | {n(r.get('e2e_s_p95'))} | {n(r.get('cache_hit_ratio'),4)} | [原始结果]({r['source']}) |")
    lines+=['','## 各并发当前最佳选择','', '| 场景 | 并发 | 已完成chunk数 | 当前平均吞吐最高Chunk | 平均总输出tok/s |','|---|---|---|---|---|']
    for key in sorted({(r['scene'],r['concurrency']) for r in cells}):
        valid=[r for r in cells if (r['scene'],r['concurrency'])==key and r['complete'] and r['cache_condition_valid']]
        if valid:
            means={chunk:statistics.mean(r['total_output_tok_s'] for r in valid if r['chunk']==chunk) for chunk in {r['chunk'] for r in valid}}
            best=max(means,key=means.get)
            lines.append(f"| {key[0]} | {key[1]} | {len(means)} | {best} | {n(means[best])} |")
    lines+=['','最佳仅在已完成的网格内成立。接近的候选需复测；不同输入和并发不能混算速度倍率。',
            'E2E分解保存服务器前向前、prefill墙钟跨度、decode至结束跨度及客户端残差；这些跨度包含调度间隙，不伪称纯算子耗时。',
            '每组metrics前后快照、按10秒采样、全部请求payload/ID/全文/SSE均已保存，可离线继续分析。']
    (CHUNK/'ChunkSize自主实验实测.md').write_text(humanize_report('\n'.join(lines)+'\n'))
    (CHUNK/'ChunkSize自主实验实测.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2))
    if cells:
        fields=sorted({k for r in cells for k in r})
        with (CHUNK/'ChunkSize自主实验实测.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in cells)
    audit_chunk_requirement_coverage()
    print('completed MTP groups',len(rows),'weight pairs',len(comparisons),'chunk cells',len(cells),'phase',state.get('phase'))


if __name__=='__main__':main()
