#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN_ID='20261103_051202'
START=ROOT.joinpath('CLOCK.txt').read_text().splitlines()[0].split('=',1)[1]
DEADLINE=ROOT.joinpath('CLOCK.txt').read_text().splitlines()[1].split('=',1)[1]
NOW=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).astimezone().isoformat(timespec='seconds')

def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def write_timings():
 rows=[]
 with open(ROOT/'TIMINGS_ONLINE.csv') as f:
  for r in csv.DictReader(f):
   rows.append({'section':'online','case':r['case'],'ep':'','batch':'','work_type':'','block':r['block'],'calls':r['calls'],
                'total_ms':r['total_ms'],'ms_per_call':r['ms_per_call'],'host_ms_per_call':r['host_ms_per_call'],'order':r['order'],'mode':'eager_local_chain','evidence':'this_run'})
 with open(ROOT/'TIMINGS_DECODE.csv') as f:
  for r in csv.DictReader(f):
   rows.append({'section':'decode','case':'','ep':r['ep'],'batch':'','work_type':r['work_type'],'block':r['block'],'calls':r['calls'],
                'total_ms':r['total_ms'],'ms_per_call':r['ms_per_call'],'host_ms_per_call':r['host_ms_per_call'],'order':'','mode':'eager_local_chain','evidence':'this_run'})
 online=json.loads((ROOT/'ONLINE_RESULT.json').read_text())
 for kind in ('copy','format_prep'):
  for r in online[kind]['timing_blocks']:
   rows.append({'section':'online_overhead','case':kind,'ep':'','batch':'','work_type':'','block':r['block'],'calls':r['calls'],
                'total_ms':r['total_ms'],'ms_per_call':r['ms_per_call'],'host_ms_per_call':r['host_ms_per_call'],'order':'','mode':'eager_local_operation','evidence':'this_run'})
 for case in ('hot','uniform'):
  samples=online['planner'][case]['bench']['samples_us']
  for block,value in enumerate(samples):
   rows.append({'section':'planner','case':case,'ep':'','batch':'','work_type':'','block':block,'calls':20000,
                'total_ms':value*20000/1000.0,'ms_per_call':value/1000.0,'host_ms_per_call':value/1000.0,'order':'','mode':'cpu_reference_planner','evidence':'this_run'})
 with open(ROOT/'TIMINGS.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 return rows

def main():
 cpu=json.loads((ROOT/'CPU_AUDIT.json').read_text()); online=json.loads((ROOT/'ONLINE_RESULT.json').read_text()); decode=json.loads((ROOT/'DECODE_RESULT.json').read_text())
 timing_rows=write_timings()
 plans={'start':START,'deadline':DEADLINE,'package_prepared_at':NOW,'weight_manifest_sha256':'7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5',
   'online':{'method':'UltraEP-style current-count -> copy decision -> integer quota; two experts/two ranks, one temp slot per rank',
     'examples':online['cases'],'planner':online['planner'],'copy':online['copy'],'format_prep':online['format_prep'],'correctness':online['correctness']},
   'decode':{'identity':decode['identity'],'mapping':'contiguous logical experts to EP ranks; 16/8/4 slots for EP16/32/64',
     'real_trace_contract':decode['plans']['real_trace_contract'],'steps':decode['plans']['steps'],'grid':decode['grid'],
     'work_types_measured':list(decode['work_type_timing'].keys()),'permutation_validation':decode['permutation_validation']},
   'cpu_corrections':cpu}
 (ROOT/'INPUTS_PLANS.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
 artifacts=[]
 for p in [ROOT/'CPU_AUDIT.json',ROOT/'ONLINE_RESULT.json',ROOT/'DECODE_RESULT.json',ROOT/'INPUTS_PLANS.json',ROOT/'TIMINGS.csv']+[ROOT/f for f in ['cpu_audit.py','online_npu.py','decode_npu.py','package_results.py','RESULTS.md','CORRECTIONS.md','REPRODUCE.md','CLOCK.txt']]:
  if p.exists(): artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest={'run_id':RUN_ID,'start_time':START,'deadline':DEADLINE,'package_prepared_at':NOW,
   'host':platform.node(),'device':{'count':1,'name':'Ascend910_9382','binding':'only /dev/davinci0'},
   'software':{'torch':'2.10.0+cpu','torch_npu':'2.10.0','CANN':'9.0.0','image_id':'sha256:26a020117564da514f33c4e72353d07020bc4bfe176f4e61858b4abeac6d3e9b','image_digest':'quay.io/ascend/sglang@sha256:f492ff6a50b779db301bb5a8f80e33c98981c5076bd0feb6225b63d7971c6385'},
   'base_git_commit':'3eeb16b','weight_manifest_sha256':'7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5',
   'input_identity':{'online':'fixed 960/64 and 512/512 mechanisms; artificial hidden','decode':'fixed seed 20261103 artificial top8; real trace audit batch16 only'},
   'correctness':{'cpu_conservation':all(v['corrected_by_expert_conserved'] and v['empty_adapter_extra_slots_zero'] for v in cpu['splits'].values()),
     'online_bridge':online['correctness']['pass'],'online_split_bitwise':online['correctness']['split']['fixed_quant_9_8_vs_full_17']['bitwise_equal'],
     'copy_bitwise':all(v['bitwise_equal'] for v in online['copy']['check']['source_vs_destination'].values()),
     'decode_permutation_validated':all(v['pass_at_5pct'] for v in decode['permutation_validation'])},
   'timing_protocol':{'warmup_calls':20,'formal_blocks':5,'min_calls_per_block':20,'min_block_ms':100,
     'block_floor_verified':all(v['all_blocks_ge_100ms'] for v in decode['work_type_timing'].values()) and all(v['all_blocks_ge_100ms'] for v in online['timings'].values()),
     'online_copy_format_floor_verified':all(x['total_ms']>=100 for k in ('copy','format_prep') for x in online[k]['timing_blocks']),
     'planner_samples':15,'planner_iterations_per_sample':20000},
   'artifacts':artifacts,'raw_local_only':['/home/chengyetai/workspace/current_exp/20261102_load_balance_e0/logs/20261102_081322/eplb_windows/000000_1793579172.815737.pt','/home/chengyetai/workspace/current_exp/20261103_3h_online_decode/weight_audit/','/home/chengyetai/workspace/current_exp/20261103_3h_online_decode/*.stdout']}
 (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'manifest_correctness':manifest['correctness'],'timing_rows':len(timing_rows),'artifacts':len(artifacts)},indent=2))
if __name__=='__main__':main()
