#!/usr/bin/env python3
"""Summarize archived ON/OFF MTP runs without treating partial 32K output as complete."""
import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

SCENES=['decode','mixed','prefix']

def nfmt(v,n=2):return f'{v:.{n}f}' if isinstance(v,(int,float)) else '—'
def avg(rows,key):
    x=[r[key] for r in rows if isinstance(r.get(key),(float,int))]
    return statistics.mean(x) if x else None

def window_observation(path,r,stop_ns):
    count=0;first=None;last=None
    with gzip.open(path,'rt',encoding='utf-8') as source:
        for line in source:
            entry=json.loads(line)
            if entry['recv_perf_ns']>stop_ns:continue
            raw=entry.get('raw_line','')
            if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':continue
            meta=json.loads(raw[5:]).get('meta_info') or {}
            c=meta.get('completion_tokens')
            if isinstance(c,int) and c>count:
                count=c
                if first is None:first=entry['recv_perf_ns']
                last=entry['recv_perf_ns']
    return {**r,'window_output_tokens':count,'window_first_perf_ns':first,
            'window_last_perf_ns':last,
            'window_tpot_ms':(last-first)/1e6/(count-1) if first and last and count>1 else None}


def qtile(values,q):
    return sorted(values)[math.ceil(len(values)*q)-1] if len(values)>=3 else None


def main(root):
    out=root/'analysis';out.mkdir(exist_ok=True)
    doc=['# MTP ON/OFF 双节点 45 分钟实验：实际观察','',
         '23 节点 MTP OFF；25 节点 MTP ON。`decode=1024/32768`、`mixed=8192/1024`、`prefix=32768/300`。两机每场景同一 45 分钟窗口；数据集、种子和两机原始请求见上级目录。',
         '', '| 场景 | 模式 | 已完成请求 | 45 分钟截断请求 | 非正常错误 | 实际生成总数 | 窗口输出吞吐 (token/s，分母含预热时间) | 观测生成速率 (token/s) | 单请求 E2E 输出速率 (token/s) | TTFT 均值(s) | 观察到的 TPOT 均值(ms/token) | 接受长度 (含 bonus) | Draft 接受率 |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    records=[]
    percentiles=[]
    for scene in SCENES:
        summaries={}
        for mode in ('off','on'):
            folder=root/scene/mode
            if not (folder/'completed_or_partial_results.json').exists():continue
            rows=json.loads((folder/'completed_or_partial_results.json').read_text())
            schedule=json.loads((root/scene/'schedule.json').read_text())
            dur=schedule['duration_s']
            rows=[window_observation(folder/f'request_{r["case_index"]:04d}/measurement/sse_raw.jsonl.gz',r,int(schedule['stop_perf_s']*1e9)) for r in rows]
            ok=[r for r in rows if r['http_status']==200 and r['error'] is None]
            completed=[r for r in ok if r['actual_output']==r['requested_output'] and r['done_events']==1]
            cutoff_path=root/scene/'terminal_counter_cutoff.json'
            boundary_ids={x.get('active',{}).get('rid') for x in json.loads(cutoff_path.read_text())['responses']
                         if x.get('active')} if cutoff_path.exists() else set()
            partial=[r for r in ok if r['partial_at_45m_deadline'] or
                     (r['actual_output']<r['requested_output'] and r['meta_info_last'].get('id') in boundary_ids)]
            acceptance_rows=[r for r in ok if (r['meta_info_last'].get('spec_verify_ct') or 0)>0]
            accepted_output=sum(r['actual_output'] for r in acceptance_rows)
            correct=sum((r['meta_info_last'].get('spec_num_correct_drafts') or 0) for r in acceptance_rows)
            proposed=sum((r['meta_info_last'].get('spec_num_proposed_drafts') or 0) for r in acceptance_rows)
            verify=sum((r['meta_info_last'].get('spec_verify_ct') or 0) for r in acceptance_rows)
            recorded_tokens=sum(r['actual_output'] for r in ok)
            tok=sum(r['window_output_tokens'] for r in ok)
            boundary_tail_rows=[r for r in ok if r['started_perf_ns']>=int((schedule['stop_perf_s']-3)*1e9) and r['actual_output']<r['requested_output']]
            boundary_tail_indexes={r['case_index'] for r in boundary_tail_rows}
            latency_rows=[r for r in ok if r['window_first_perf_ns'] is not None and r['case_index'] not in boundary_tail_indexes]
            generated=sum(max(0,r['window_output_tokens']-1) for r in latency_rows)
            generation_seconds=sum((r['window_last_perf_ns']-r['window_first_perf_ns'])/1e9 for r in latency_rows
                if r['window_output_tokens']>1 and r['window_first_perf_ns'] and r['window_last_perf_ns'])
            e2e_seconds=sum(r['e2e_s'] for r in latency_rows)
            e2e_tokens=sum(r['actual_output'] for r in latency_rows)
            acceptance_covered_window_tokens=sum(r['window_output_tokens'] for r in acceptance_rows)
            stats={'scene':scene,'mode':mode,'window_seconds':dur,'complete_requests':len(completed),
                   'partial_requests':len(partial),'error_requests':len(rows)-len(ok),
                   'observed_tokens':tok,'collected_tokens_including_postdeadline':recorded_tokens,
                   'postdeadline_collected_tokens':recorded_tokens-tok,'requests_without_output_in_window':sum(r['window_first_perf_ns'] is None for r in ok),'boundary_tail_requests':len(boundary_tail_rows),'latency_sample_count':len(latency_rows),
                   'window_output_tokens_per_s':tok/dur,
                   'observed_generation_tokens_per_s':generated/generation_seconds if generation_seconds else None,
                   'single_request_e2e_output_tokens_per_s':e2e_tokens/e2e_seconds if e2e_seconds else None,
                   'mean_ttft_s':avg(latency_rows,'ttft_s'),'mean_observed_tpot_ms':avg(latency_rows,'window_tpot_ms'),
                   'spec_verify_ct':verify,'spec_proposed_drafts':proposed,'spec_accepted_drafts':correct,
                   'acceptance_covered_output_tokens':accepted_output,
                   'acceptance_covered_window_tokens':acceptance_covered_window_tokens,
                   'acceptance_output_coverage_ratio':acceptance_covered_window_tokens/tok if tok and mode=='on' else None,
                   'weighted_accept_length':accepted_output/verify if verify else None,
                   'weighted_draft_accept_rate':correct/proposed if proposed else None,
                   'actual_cached_ratios':[r['cached_tokens']/r['prompt_tokens'] for r in ok
                       if r['cached_tokens'] is not None and r['prompt_tokens']]}
            for metric,key in [('ttft_s','ttft_s'),('tpot_ms','window_tpot_ms')]:
                values=[r[key] for r in latency_rows if isinstance(r.get(key),(int,float))]
                percentiles.append({'scenario':scene,'mode':mode,'metric':metric,'n':len(values),
                    'mean':statistics.mean(values) if values else None,'p95':qtile(values,.95),
                    'p99':qtile(values,.99),'quantile_method':'nearest_rank',
                    'interpretation':'single-request observation; no meaningful tail estimate' if len(values)<3
                      else ('small sample; exploratory tails' if len(values)<20 else 'window workload sample')})
            summaries[mode]=stats
            doc.append(f'| {scene} | {mode.upper()} | {len(completed)} | {len(partial)} | {len(rows)-len(ok)} | {tok} | {nfmt(tok/dur,3)} | {nfmt(stats["observed_generation_tokens_per_s"],3)} | {nfmt(stats["single_request_e2e_output_tokens_per_s"],3)} | {nfmt(stats["mean_ttft_s"],3)} | {nfmt(stats["mean_observed_tpot_ms"],2)} | {nfmt(stats["weighted_accept_length"],3)} | {nfmt(stats["weighted_draft_accept_rate"],3)} |')
            for row in rows:
                meta=row['meta_info_last'];records.append({'scenario':scene,'mode':mode,'index':row['case_index'],
                  'input_sha256':row['input_sha256'],'prompt_tokens':row['prompt_tokens'],'cached_tokens':row['cached_tokens'],
                  'target_output':row['requested_output'],'actual_output':row['actual_output'],'output_tokens_in_45m':row['window_output_tokens'],'window_tpot_ms':row['window_tpot_ms'],
                  'boundary_tail':row['case_index'] in boundary_tail_indexes,
                  'completed':row['done_events']==1 and row['actual_output']==row['requested_output'],
                  'partial_deadline':row['partial_at_45m_deadline'] or (row['actual_output']<row['requested_output'] and meta.get('id') in boundary_ids),'http_status':row['http_status'],
                  'error':str(row['error'] or ''),'ttft_s':row['ttft_s'],'observed_tpot_ms':row['observed_tpot_ms'],
                  'e2e_s':row['e2e_s'],'spec_verify_ct':meta.get('spec_verify_ct'),
                  'spec_accept_length':meta.get('spec_accept_length'),'spec_accept_rate':meta.get('spec_accept_rate'),
                  'spec_num_correct_drafts':meta.get('spec_num_correct_drafts'),
                  'spec_num_proposed_drafts':meta.get('spec_num_proposed_drafts')})
        (out/(scene+'_summary.json')).write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    if records:
        with (out/'all_requests.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=records[0].keys());wr.writeheader();wr.writerows(records)
    if percentiles:
        with (out/'latency_percentiles.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=percentiles[0].keys());wr.writeheader();wr.writerows(percentiles)
    doc+=['','**口径**：接受长度只按有最终累计计数的 MTP ON 请求的实际输出 token 数 / `spec_verify_ct` 求加权值；覆盖 token 数及比例另存各场景 summary JSON，包含 bonus token；接受率为已接受 draft / 已提出 draft。该固定镜像中间 SSE 不带接受计数；截止附近 helper 对指定 ID 先发 abort，让原流收到最终计数。边界若还提交极短新请求且无最终计数，不纳入接受长度分母，吞吐则仍记录其实际输出。服务返回的逐请求 `spec_accept_length` 和原始计数仍在 `measurement/result.json` 的 `meta_info_last`。OFF 没有接受统计。','',
          '“45 分钟截断”表示这条请求未输出满目标长度；其 TTFT 和“观察到的 TPOT”仅对应已收到的输出，不代表完整 32K 输出。窗口输出吞吐按原始 SSE 到达时间严格只统计截止前实际输出 token 数，再除以各场景 2700 秒；截止后才到达的 token 与没有窗口内输出的极短边界请求单独记录，不纳入窗口 TTFT/TPOT 均值；截止前最后3秒才开始且未完成的极短补发请求标为 boundary_tail，保留吞吐贡献，但也不纳入延迟均值；包含缓存清理、预热、输入处理和间隙。观测生成速率以 `(实际输出 - 1)/首末输出到达间隔` 跨请求汇总；单请求 E2E 输出速率以实际输出数除以对应请求总耗时，均为客户端可观察量。实际完成请求数、错误、是否在窗口结束前完成见各组 `summary.json`。','',
          '缓存前缀工作负载的实际命中率逐请求记录在 summary JSON；其它场景每条请求均尝试清除缓存。`/flush_cache` 只有在没有活跃请求时才真正执行；若截止后仍有遗留请求，应检查下一场景的缓存记录和服务日志。','',
          '以上是单请求在途（每台独立节点），**服务整体吞吐**不能直接推广到更高并发；随机 token ID 不能代表真实对话的接受率。客户端 TPOT 用 `(最后输出事件到达 - 首个输出事件到达)/(实际输出 token 数-1)`，若 MTP 每事件返回多个 token，不把事件间隔误当作单 token ITL。','']
    (out/'README.md').write_text('\n'.join(doc))
    print(out/'README.md')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);main(ap.parse_args().root)
