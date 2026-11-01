#!/usr/bin/env python3
"""Preserve the old41-record audit and append clearly labelled new probes."""
from report_labels import humanize_report, combined_row_label, combined_lengths, combined_counts
import csv,json,statistics
from pathlib import Path
ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
Q1=ROOT.parent/'Q1MTPExp'
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def main():
 history=json.loads((Q1/'MTP全部实验汇总_20261012.json').read_text())
 rows=[dict(x) for x in history['rows']];failed=[]
 for p in sorted((ROOT/'results').glob('*/*/result.json')):
  if p.parent.name=='sanity_chat':continue
  d=json.loads(p.read_text());records=d['requests']
  if any(v.get('failures') for v in d.get('validation',[])) or any(r.get('error') or not r.get('stream_done') for r in records):
   failed.append({'source_file':str(p),'validation':d.get('validation')});continue
  agg=d['aggregate'];verified=[r for r in records if r['meta'].get('spec_verify_ct')]
  verify=sum(r['meta']['spec_verify_ct'] for r in verified)
  info=json.loads((p.parent.parent/'server_info.json').read_text())
  if not info.get('speculative_algorithm'):continue
  pos=agg.get('pos_rates_weighted')
  # Acceptance excludes requests ending in prefill without a verify step;
  # performance includes every completed request, including those requests.
  performance={key:statistics.mean(r[key] for r in records if r.get(key) is not None)
               if any(r.get(key) is not None for r in records) else None
               for key in ('ttft_s','tpot_ms','e2e_s')}
  total_output=sum(r['completion_tokens'] for r in records)
  total_prompt=sum(r['meta']['prompt_tokens'] for r in records)
  total_cached=sum(r['meta'].get('cached_tokens',0) for r in records)
  rows.append({'category':'公开源校验自主探索','date':'2026-10-12/13','label':p.parent.name,'run':p.parent.parent.name,
   'mode':'ON','node':'23','dataset':'完整ShareGPT聊天' if p.parent.name.startswith('sharegpt') else '保存的原生random-ids及明确标注的诊断变体',
   'concurrency':d['concurrency'],'requests_measured':len(records),'requests_with_acceptance':len(verified),
   'input_lengths':sorted({r['input_tokens_sent'] for r in records}),'output_target':d['target_output_tokens'],
   'AL':sum(r['completion_tokens'] for r in verified)/verify if verify else None,
   'draft_rate':agg.get('accept_rate_weighted'),'P1_P2_P3':pos,
   'TTFT_s':performance['ttft_s'],'TPOT_ms':performance['tpot_ms'],'E2E_s':performance['e2e_s'],
   'output_tok_s':total_output/agg['wall_s'],'throughput_method':'所有完成请求的总输出/整组墙钟；包含prefill即结束的请求；短诊断，不作跨配置收益榜单',
   'performance_requests':len(records),'performance_output_tokens':total_output,
   'acceptance_count_scope':'仅spec_verify_ct>0的请求；prefill即结束的请求位置率缺值，不补0',
   'cache_hit_ratio':total_cached/total_prompt if total_prompt else None,'source_file':str(p),'status':'completed',
   'ignore_eos':d['ignore_eos'],'server_config':{k:info.get(k) for k in ['version','tp_size','dp_size','ep_size','deepep_mode','mem_fraction_static','chunked_prefill_size','max_total_num_tokens','disable_cuda_graph']},
   'notes':'完整EOS回复与长度截断窗口分开；原始payload/输出/计数/验证保留。不重填旧记录缺失字段。'})
 assert len({(r.get('source_file'),r['label'],r['mode']) for r in rows})==len(rows),'duplicate record identities'
 dest=Q1/'MTP全部实验汇总_含自主探索'
 scope='历史41记录保持原口径。新增公开源恢复后、在23进行的有效MTP请求组；不含sanity、预热、target单token核对、失败组或MTP OFF Chunk矩阵。不是同条件收益榜单。'
 dest.with_suffix('.json').write_text(json.dumps({'scope':scope,'rows':rows,'invalid_autonomous_groups':failed},ensure_ascii=False,indent=2))
 keys=sorted({k for r in rows for k in r})
 with dest.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as fobj:
  w=csv.DictWriter(fobj,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
 lines=['# MTP全部实测：历史与自主探索合并','',scope,'',
 '场景名已改为中文，长度单位均为token。配置代号与缩写见[名称说明](../autonomous_20261012/负载与配置名称说明.md)。','',
 '| 轮次 | 场景 | 输入/目标输出 | 并发/请求 | AL | 草稿率 | P1/P2/P3 | 总输出tok/s | TTFT均值s | TPOT均值ms | E2E均值s | 来源 |',
 '|---|---|---|---|---|---|---|---|---|---|---|---|']
 for r in rows:
  pos='/'.join(f(v,4) for v in r.get('P1_P2_P3') or []) or '—'
  mode = 'MTP开启' if r['mode'] == 'ON' else 'MTP关闭'
  lines.append(f"| {r.get('run',r['category'])} | {combined_row_label(r)}（{mode}） | {combined_lengths(r)} | {combined_counts(r)} | {f(r.get('AL'))} | {f(r.get('draft_rate'),4)} | {pos} | {f(r.get('output_tok_s'))} | {f(r.get('TTFT_s'))} | {f(r.get('TPOT_ms'))} | {f(r.get('E2E_s'))} | [原始结果]({r.get('source_file')}) |")
 lines+=['',f'共{len(rows)}条历史/有效新增记录；{len(failed)}个自主探索失败请求组单列在JSON。历史原文的缺失值、窗口截断和不同吞吐口径仍有效，不将新P位置率回填历史。','',
  '诊断变体的输入格式、EOS、实际长度与服务开关见JSON及原始payload；部分允许EOS的256窗口仍达长度上限，并非完整回复。sharegpt_chat_complete的finish_reason应逐条检查。']
 dest.with_suffix('.md').write_text(humanize_report('\n'.join(lines)+'\n'));print('combined MTP records',len(rows),'invalid new groups',len(failed))
if __name__=='__main__':main()
