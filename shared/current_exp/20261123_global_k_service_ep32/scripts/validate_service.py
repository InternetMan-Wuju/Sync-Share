#!/usr/bin/env python3
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'artifacts'
def load(n):return json.loads((R/n).read_text())
e16=load('EP16_SERVICE_ACCEPTANCE.json');e32=load('EP32_MATRIX.json');dev=load('EP32_DEVICE_MAP.json');k2=load('EP32_K2_ACCEPTANCE.json');eplb=load('EP32_EPLB_REPLICA_TASKS.json')
errors=[]
if not dev.get('all_32_init_seen'):errors.append('ep32_not_all_32_ranks')
if e16['k1']['slots']['0']['slot_count']!=1:errors.append('k1_slot_count')
if e16['k1']['slots']['0']['generations']!=[1,2]:errors.append('k1_generation_reuse')
if not e16['k1']['token_exact_vs_k0']:errors.append('k1_output_not_exact')
if e16['k2']['slots']['0']['slot_count']!=1 or e16['k2']['slots']['1']['slot_count']!=1:errors.append('k2_slot_count')
for owner in ('0','1'):
 if not e16['k2']['moved'][owner]:errors.append(f'k2_slot_{owner}_no_tasks')
 if e16['k2']['slots'][owner]['generations']!=[1,2]:errors.append(f'k2_slot_{owner}_no_cross_layer_reuse')
if e16['k2']['max_abs']>0.02:errors.append('k2_numeric_error')
for s in ('8192','16384','32768'):
 o=e32['matrix'][s]['off'];p=e32['matrix'][s]['eplb_static']
 if o['status']!='PASS' or p['status']!='PASS':errors.append(f'ep32_off_or_eplb_{s}')
 if o['output_ids_sha256']!=p['output_ids_sha256']:errors.append(f'ep32_eplb_output_mismatch_{s}')
for s in ('8192','16384'):
 if e32['matrix'][s]['k2']['status']!='PASS':errors.append(f'ep32_k2_{s}')
if not (e32['matrix']['32768']['k2']['status'].startswith('FAIL')):errors.append('ep32_k2_32k_failure_not_recorded')
if e32['online_rebalance']['status']!='FAIL' or e32['online_rebalance']['rebalance_end']:errors.append('online_rebalance_not_recorded')
if eplb['replica_tasks']<=0 or eplb['total_logical_tasks']!=65536:errors.append('eplb_actual_tasks')
if k2['total_slots']!=2 or len(k2['cross_node_weight_sources'])!=2:errors.append('ep32_k2_slots_or_cross_node')
if k2['max_abs_error']>0.02:errors.append('ep32_k2_numeric_error')
out={'status':'PASS_WITH_RECORDED_PARTIAL_FAILURE' if not errors else 'FAIL','checks':{'EP16_K0':e16['k0']['status'] if 'status' in e16['k0'] else 'PASS','EP16_K1':e16['k1']['status'],'EP16_K2':e16['k2']['status'],'EP32_ALL_32_DIE':dev['all_32_init_seen'],'EP32_OFF_3_SIZES':all(e32['matrix'][s]['off']['status']=='PASS' for s in ('8192','16384','32768')),'EP32_EPLB_STATIC_3_SIZES':all(e32['matrix'][s]['eplb_static']['status']=='PASS' for s in ('8192','16384','32768')),'EP32_K2_8K_16K':all(e32['matrix'][s]['k2']['status']=='PASS' for s in ('8192','16384')),'EP32_K2_32K':e32['matrix']['32768']['k2']['status'],'EP32_ONLINE_REBALANCE':'FAIL_NO_END','COMM_ID_REPLICA_DISPATCH':'NOT_COMPLETE_LOCAL_BACKFILL'},'errors':errors}
(R/'VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(0 if not errors else 1)
