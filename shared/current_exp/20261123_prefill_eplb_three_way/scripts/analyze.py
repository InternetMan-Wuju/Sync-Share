#!/usr/bin/env python3
from __future__ import annotations
import csv, glob, hashlib, json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPLAY={
 '8k':ROOT/'artifacts/replay16_8k_final_8k',
 '16k':ROOT/'artifacts/replay16_16k_final_16k',
 '32k':ROOT/'artifacts/replay16_32k_final_32k',
}
SERVICE={
 ('8k','off'):ROOT/'artifacts/off_8k_ttft_10.json',
 ('16k','off'):ROOT/'artifacts/off_16k_ttft_10.json',
 ('32k','off'):ROOT/'artifacts/off_32k_ttft_10.json',
 ('8k','eplb_static'):ROOT/'artifacts/eplb_static_8k_ttft.json',
 ('16k','eplb_static'):ROOT/'artifacts/eplb_static_16k_ttft.json',
 ('32k','eplb_static'):ROOT/'artifacts/eplb_static_32k_ttft.json',
}

def pct(v,p):
 v=sorted(v);return v[int(p*(len(v)-1))]
summary={'topology':'EP16, one node npu1-23, 16 Ascend 910C','scope':'layer49 routed-MoE replay using real captured EP16 hidden/top-k plus separate service diagnostics','replay':{},'service':{},'memory':{},'actions':{},'blockers':{}}
replay_csv=[]
for size,dirp in REPLAY.items():
 rows=[json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(dirp/'replay_rank*.json')))]
 assert len(rows)==16
 arms={}
 for arm in ('original','eplb','ours'):
  vals=[];rep_replica=[];rep_rank_med=[];errors=[]
  for rep in range(10):
   ms=[r['records'][rep]['arms'][arm]['total_wall_ms'] for r in rows]
   vals.append(max(ms));rep_rank_med.append(statistics.median(ms));rep_replica.append(sum(r['records'][rep]['arms'][arm]['replica_tasks'] for r in rows))
   for r in rows: errors.extend(r['records'][rep]['arms'][arm].get('error_vs_original',[]))
   replay_csv.append({'size':size,'arm':arm,'rep':rep,'group_max_ms':max(ms),'rank_median_ms':statistics.median(ms),'replica_tasks':rep_replica[-1]})
  arms[arm]={'median_group_max_ms':statistics.median(vals),'p95_group_max_ms':pct(vals,.95),'min_group_max_ms':min(vals),'max_group_max_ms':max(vals),'replica_tasks':rep_replica[0],'max_abs_error_vs_original':max((e['max_abs'] for e in errors),default=0.0),'max_nrmse_vs_original':max((e['nrmse'] for e in errors),default=0.0),'raw_group_max_ms':vals}
 for target in ('eplb','ours'):
  deltas=[arms['original']['raw_group_max_ms'][i]-arms[target]['raw_group_max_ms'][i] for i in range(10)]
  arms[f'original_minus_{target}']={'median_ms':statistics.median(deltas),'p95_ms':pct(deltas,.95),'samples_ms':deltas}
 summary['replay'][size]={'rows_per_rank':rows[0]['hidden_rows'],'arms':arms}

# Service diagnostics: separate processes, not a formal paired three-way result.
service_csv=[]
for (size,arm),path in SERVICE.items():
 if not path.exists():continue
 d=json.loads(path.read_text())
 measured=[r for r in d['rows'] if r.get('phase')=='measured']
 good=[r for r in measured if r.get('ttft_s') is not None]
 tts=[r['ttft_s'] for r in good]; totals=[r['total_s'] for r in good]
 summary['service'].setdefault(size,{})[arm]={'requests':len(measured),'successful':len(good),'ttft_median_s':statistics.median(tts) if tts else None,'ttft_p95_s':pct(tts,.95) if tts else None,'ttft_min_s':min(tts) if tts else None,'ttft_max_s':max(tts) if tts else None,'total_median_s':statistics.median(totals) if totals else None,'status':'PASS' if len(good)==10 else 'FAIL_OR_LIMITED'}
 for i,r in enumerate(measured):service_csv.append({'size':size,'arm':arm,'rep':i,'ttft_s':r.get('ttft_s'),'total_s':r.get('total_s'),'status':'ok' if r.get('ttft_s') is not None else 'no_first_token'})
for size in ('8k','16k'):
 o=summary['service'][size]['off']['ttft_median_s'];e=summary['service'][size]['eplb_static']['ttft_median_s']
 summary['service'][size]['eplb_static']['delta_ttft_vs_off_ms']=(e-o)*1000
summary['service']['32k']['eplb_static']={'status':'BLOCKED_CONTEXT_CAPACITY','max_allowed_tokens':27386,'request_tokens':32741,'reason':'static EPLB +16 raises per-rank weight by about 2.66 GiB and shrinks max_total_tokens from 53632 to 27392'}


# Per-rank load and timing summary for the final 10-replicate runs.
rank_details={}
raw_sha={}
for size,dirp in REPLAY.items():
 rank_details[size]={}
 rows=[json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(dirp/'replay_rank*.json')))]
 for arm in ('original','eplb','ours'):
  arm_rows=[]
  for r in rows:
   vals=[]; tasks0=None; replicas0=None
   for rep in range(10):
    acc=0.0
    for ci,x in enumerate(r['records'][rep]['arms'][arm]['chunks']):
     acc+=x['wall_ms']
     if rep==0:
      own=next(y for y in x['rank_tasks'] if y['rank']==r['rank'])
      if tasks0 is None: tasks0=[]
      tasks0.append({'chunk':ci,'local_rows':x['rows'],'local_task_count':own['tasks'],'local_counts':own['counts']})
    vals.append(acc)
    if rep==0: replicas0=r['records'][rep]['arms'][arm]['replica_tasks']
   arm_rows.append({'rank':r['rank'],'rank_median_ms':statistics.median(vals),'rank_max_ms':max(vals),'replica_tasks':replicas0,'chunks':tasks0})
  rank_details[size][arm]=arm_rows
 for path in sorted(Path(dirp).glob('*')):
  if path.is_file(): raw_sha[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'artifacts/REPLAY_RANK_DETAILS.json').write_text(json.dumps(rank_details,indent=2)+'\n')
(ROOT/'artifacts/RAW_REPLAY_HASHES.json').write_text(json.dumps(raw_sha,indent=2)+'\n')
layout=json.loads((ROOT/'layouts/eplb16_static.json').read_text())['physical_to_logical_map'][49]
cal=json.loads((ROOT/'artifacts/calibration/eplb16_layer49_summary.json').read_text())
observed=json.loads((ROOT/'artifacts/calibration/eplb_static_service_replica_tasks.json').read_text())
summary['actions']['eplb']={'algorithm':cal['algorithm'],'algorithm_sha256':cal['algorithm_sha256'],'physical_experts':272,'extra_experts':16,'hot_logical':[x['logical'] for x in cal['top_hot']],'service_layout_active':True,'service_layout_evidence':'server log: init_expert_location from init_by_mapping','service_actual_replica_tasks_8k':observed['replica_tasks'],'service_total_tasks_8k':observed['total_tasks'],'replay_replica_tasks':{s:summary['replay'][s]['arms']['eplb']['replica_tasks'] for s in REPLAY}}
k1_install=json.loads((ROOT/'artifacts/K1_INSTALL.json').read_text())
summary['actions']['ours']={'global_extra_slots':1,'slot_owner':0,'hot_logical':cal['top_hot'][0]['logical'],'slot_bytes':37810176,'replay_replica_tasks':{s:summary['replay'][s]['arms']['ours']['replica_tasks'] for s in REPLAY},'service_status':'BLOCKED_UNIFORM_PHYSICAL_LAYOUT','steady_install_ms_median':k1_install['total_ms_median'],'steady_source_prepare_ms_median':k1_install['source_prepare_ms_median'],'steady_transfer_ms_median':k1_install['transfer_ms_median']}
# Memory and source hashes.
summary['memory']={
 'ep16_main_bank_bytes_per_rank':604962816,
 'ep16_eplb17_bank_bytes_per_rank':642772992,
 'eplb_persistent_extra_bytes_per_rank':642772992-604962816,
 'ours_persistent_slot_bytes_group':37810176,
 'service_off_weight_mem_gib_per_rank':47.97,
 'service_eplb_static_weight_mem_gib_per_rank':50.63,
 'service_eplb_weight_delta_gib_per_rank':50.63-47.97,
 'service_off_max_total_tokens':53632,
 'service_eplb_static_max_total_tokens':27392,
 'production_format_unknown':True,
}
summary['blockers']={
 'ours_service':{'status':'FAIL','location':'sglang.srt.eplb.expert_location._init_common:assert num_physical_experts % ep_size == 0','config':'EP16, redundant=1 gives 257 physical experts','evidence':str(ROOT/'logs/k1_invalid_k1_invalid_probe/server.log')},
 'native_eplb_online_rebalance':{'status':'FAIL','location':'sglang.srt.eplb.eplb_manager.rebalance','errors':['For a tensor of internal format, its storage_offset must be 0','NPU out of memory: tried to allocate 410.00 MiB, 216.30 MiB free'],'evidence':str(ROOT/'logs/eplb_online_eplb_online_8k/server.log')},
 'ep32':{'status':'NOT_RUN','reason':'EP16 already confirmed the uniform-layout blocker for K1; formal three-group EP32 cannot exist without changing the fixed physical-layout contract.'}
}
summary['conclusion']='No stable three-way latency win at 8K/16K/32K. EPLB and K1 are within noise after excluding isolated large outliers; medians differ by sub-millisecond to low-millisecond and signs vary. Service TTFT diagnostic is off vs native-layout EPLB only: EPLB is 15.8 ms faster at 8K and 7.1 ms slower at 16K, both small and not a formal three-way result. K1 service and native online rebalance are blocked.'
(ROOT/'artifacts/RESULTS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'artifacts/REPLAY_TIMINGS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=replay_csv[0].keys());w.writeheader();w.writerows(replay_csv)
with (ROOT/'artifacts/SERVICE_TTFT.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=service_csv[0].keys());w.writeheader();w.writerows(service_csv)
print(json.dumps(summary,indent=2))
