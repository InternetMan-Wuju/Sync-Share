#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

def read_csv(path):
 with open(path,newline='',encoding='utf-8-sig') as f:
  return list(csv.DictReader(f))

def num(x):
 try: return float(str(x).strip())
 except: return 0.0

def main():
 p=argparse.ArgumentParser(); p.add_argument('--profile-dir',required=True); p.add_argument('--out',required=True); a=p.parse_args()
 roots=sorted(Path(a.profile_dir).glob('npu1-23_*_ascend_pt'))
 per_rank=[]; op_values=defaultdict(list)
 for root in roots:
  out=root/'ASCEND_PROFILER_OUTPUT'; step_file=out/'step_trace_time.csv'; op_file=out/'op_statistic.csv'
  if not step_file.exists() or not op_file.exists(): continue
  step=read_csv(step_file)[0]
  step_total=num(step.get('Stage'))
  if step_total<=0:
   step_total=num(step.get('Computing'))+num(step.get('Communication'))+num(step.get('Free'))
  ops=read_csv(op_file)
  rank_ops={}
  for row in ops:
   name=row.get('OP Type') or row.get('Op Type') or row.get('Name')
   if not name: continue
   rank_ops[name]=rank_ops.get(name,0.0)+num(row.get('Total Time(us)'))
  for name,value in rank_ops.items(): op_values[name].append(value)
  per_rank.append({
   'rank_dir':root.name,'step_total_us':step_total,
   'computing_us':num(step.get('Computing')),
   'communication_not_overlapped_us':num(step.get('Communication(Not Overlapped)')),
   'communication_us':num(step.get('Communication')),
   'overlapped_us':num(step.get('Overlapped')),
   'free_us':num(step.get('Free')),
   'top_ops':sorted(rank_ops.items(),key=lambda x:-x[1])[:12],
  })
 steps=[r['step_total_us'] for r in per_rank]
 op_summary=[]
 for name,vals in op_values.items():
  op_summary.append({'op':name,'rank_count':len(vals),'median_us':statistics.median(vals),'mean_us':statistics.fmean(vals),'max_us':max(vals),'min_us':min(vals)})
 op_summary.sort(key=lambda x:-x['median_us'])
 def total_ops(names):
  return sum(x['median_us'] for x in op_summary if x['op'] in names)
 attention_names={'SparseFlashAttention','LightningIndexer','fused_rope_qk_mqa_kernel_opt','fused_split_qk_norm_kernel'}
 moe_names={'CamMoeCombineNormal','CamMoeDispatchNormal','GroupedMatmul','MoeLowLatencyCombineV2','MoeLowLatencyDispatchV2','MoeGatingTopK','DispatchLayout'}
 collective_names={x['op'] for x in op_summary if any(k in x['op'].lower() for k in ('allreduce','allgather','reduce_scatter','alltoall','broadcast'))}
 median_step=statistics.median(steps) if steps else 0
 categories={
  'attention_us':total_ops(attention_names),
  'moe_us':total_ops(moe_names),
  'collective_us':total_ops(collective_names),
  'other_us':max(0,median_step-total_ops(attention_names)-total_ops(moe_names)-total_ops(collective_names)),
 }
 result={'rank_count':len(per_rank),'step_total_us':{'median':median_step,'mean':statistics.fmean(steps) if steps else 0,'min':min(steps) if steps else 0,'max':max(steps) if steps else 0},'top_ops':op_summary[:30],'categories_us':categories,'category_fraction_of_median_step':{k:(v/median_step if median_step else 0) for k,v in categories.items()},'per_rank':per_rank}
 Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'out':a.out,'rank_count':len(per_rank),'step_median_us':median_step,'step_min_us':min(steps),'step_max_us':max(steps),'categories_us':categories,'category_fraction':result['category_fraction_of_median_step'],'top_ops':[(x['op'],round(x['median_us'],1)) for x in op_summary[:15]]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
