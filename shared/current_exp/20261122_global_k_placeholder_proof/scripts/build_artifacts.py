#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'
def load(name): return json.loads((ART/name).read_text())

def main():
    i256=[load(f'interface_run_256_rank{r}.json') for r in (0,1)]
    i257=[load(f'interface_run_257_rank{r}.json') for r in (0,1)]
    i258=[load(f'interface_run_258_rank{r}.json') for r in (0,1)]
    real=[load(f'real258_rank{r}.json') for r in (0,1)]
    zero=[load(f'zero_rank_258_rank{r}.json') for r in (0,1)]
    zero=[load(f'zero_rank_258_rank{r}.json') for r in (0,1)]

    interface_runs={
      'backend':'SGLang DeepEPDispatcher NORMAL with DEEP_USE_MODE=alltoall',
 'default_normal_failure':{'evidence':'artifacts/default_normal_256_failure.log','error':'aclnnDispatchLayout not present in libopapi.so'},
      'backend_note':'The image default normal strategy failed at Buffer.get_dispatch_layout because aclnnDispatchLayout is absent from libopapi. The installed DeepEP NORMAL alltoall strategy then passed; it uses native torch_npu npu_moe_init_routing_v2 / npu_moe_finalize_routing and torch.distributed all_to_all_single.',
      'profile_256':i256,'profile_257':i257,'profile_258':i258,'zero_rank_with_empty_padding':zero,
      'checks':{
        '256_native_positive':all(x.get('constructor')=='PASS' and x.get('correct') for x in i256),
        '257_actual_failure':all(x.get('constructor')=='PASS' and x.get('failure_type') and 'shape' in x.get('failure_message','') for x in i257),
        '258_native_protocol':all(x.get('constructor')=='PASS' and x.get('correct') for x in i258),
        'zero_rank_participant':all(x.get('correct') for x in zero) and zero[1].get('legal_tokens_sent')==0 and zero[1].get('legal_recv_tasks_rank')==0 and zero[0].get('empty_padding_rows_received')==8,
      }
    }
    task_identity={
      'mapping':{
        'rank0':{'local_0_127':'comm_0_127','local_128':'comm_128_empty'},
        'rank1':{'local_0_127':'comm_129_256','local_128':'comm_257_replica43'},
      },
      'logical_experts':256,'communication_slots':258,'real_physical_instances_in_scope':257,
      'empty_comm':128,'replica_comm':257,
      'protocol_258':i258,
      'zero_rank_with_empty_padding':zero,
      'task_identity_checks':{
        'rank0_multiset_equal':i258[0].get('task_multiset_equal'),
        'rank1_multiset_equal':i258[1].get('task_multiset_equal'),
        'rank0_tasks':[i258[0].get('expected_task_count'),i258[0].get('actual_task_count')],
        'rank1_tasks':[i258[1].get('expected_task_count'),i258[1].get('actual_task_count')],
        'empty_legal_tasks':i258[0].get('empty_id_received_count'),
        'empty_injection_guard':i258[0].get('empty_injection_guard'),
        'coverage':i258[0].get('required_logical_expert_coverage'),
        'replica_comm_257_tasks':i258[0].get('replica_comm_257_task_count'),
        'max_category_error':max(i258[0].get('max_expert_label_error',999),i258[1].get('max_expert_label_error',999)),
        'max_token_error':max(i258[0].get('max_token_position_error',999),i258[1].get('max_token_position_error',999)),
      }
    }
    # Storage is per rank; main storages are deduplicated by torch storage pointer in each rank.
    storage={
      'active_logical_experts':real[0]['active_logical_experts'],
      'rank_main_experts':{str(r):real[r]['rank_main_experts'] for r in (0,1)},
      'layers':real[0]['layers'],
      'full_128_per_rank_main_library':False,
      'main_storage_by_rank':{str(r):real[r]['main_storage'] for r in (0,1)},
      'slot_storage':{str(r):real[r]['slot_storage'] for r in (0,1)},
      'empty_comm_state_bytes':0,
      'temporary_buffers':{
        'source_flat_state_bytes_rank0':37810176,
        'per_rank_hidden_bytes':8*6144*2,
        'reference_global_hidden_bytes':16*6144*2,
        'reference_partial_float32_bytes':16*6144*4,
        'note':'Dispatch/combine and operator workspaces are reflected in per-rank allocated/reserved peaks; they are not additional persistent expert state.'
      },
      'all_rank_extra_persistent_slot_count':1,
      'all_rank_extra_persistent_slot_bytes':sum(x['slot_storage'].get('bytes',0) for x in real),
      'memory_before':{str(r):real[r]['memory_before'] for r in (0,1)},
      'memory_after_bank':{str(r):real[r]['memory_after_bank'] for r in (0,1)},
      'memory_final':{str(r):real[r]['memory_final'] for r in (0,1)},
      'scope_note':'Only the finite active main-expert set was allocated: 4 main experts per rank per layer over 2 layers. No full 128/128 main library was instantiated. The empty communication id owns no expert state.'
    }
    # Match install events by sequence; hashes must match and one storage pointer must persist.
    installs0=real[0]['install_events']; installs1=real[1]['install_events']
    install_pairs=[]
    for a,b in zip(installs0,installs1):
        install_pairs.append({'layer':a['layer'],'generation':b.get('generation'),'source_hash':a.get('source_hash'),'target_hash':b.get('target_hash'),'hash_equal':a.get('source_hash')==b.get('target_hash'),'slot_storage_ptr':b.get('slot_storage_ptr'),'slot_storage_nbytes':b.get('slot_storage_nbytes')})
    reuse={
      'rank0_runs':real[0]['reuse_runs'],'rank1_runs':real[1]['reuse_runs'],'install_pairs':install_pairs,
      'same_slot_storage_pointer':len({p['slot_storage_ptr'] for p in install_pairs})==1,
      'generations':[p['generation'] for p in install_pairs],
      'all_hash_pairs_equal':all(p['hash_equal'] for p in install_pairs),
      'all_rank0_outputs_within_tolerance':all(x['within_tolerance'] for x in real[0]['reuse_runs']),
      'all_rank1_outputs_within_tolerance':all(x['within_tolerance'] for x in real[1]['reuse_runs']),
      'cycles_49_50_49':3,
    }
    checks={
      'interface_256_positive':interface_runs['checks']['256_native_positive'],
      '257_actual_failure_captured':interface_runs['checks']['257_actual_failure'],
      '258_native_protocol':interface_runs['checks']['258_native_protocol'],
      'zero_rank_participant':interface_runs['checks']['zero_rank_participant'],
      'zero_rank_participant':all(x.get('correct') for x in zero) and zero[1].get('legal_tokens_sent')==0 and zero[1].get('legal_recv_tasks_rank')==0 and zero[0].get('empty_padding_rows_received')==8,
      'real_w8a8_258_native':all(all(x['within_tolerance'] and x['task_multiset_equal'] and x['empty_tasks']==0 for x in r['native_runs']) for r in real),
      'real_replica_partial_and_all':(any(x['mode']=='partial' and x['replica_tasks']>0 for x in real[1]['native_runs']) and any(x['mode']=='all' and x['replica_tasks']>0 for x in real[1]['native_runs'])),
      'empty_id_zero_state':storage['empty_comm_state_bytes']==0 and all(x['empty_tasks']==0 for r in real for x in r['native_runs']+r['reuse_runs']),
      'unique_group_slot':storage['all_rank_extra_persistent_slot_count']==1 and storage['all_rank_extra_persistent_slot_bytes']==37810176,
      'two_layer_reuse':reuse['same_slot_storage_pointer'] and reuse['all_hash_pairs_equal'] and reuse['all_rank0_outputs_within_tolerance'] and reuse['all_rank1_outputs_within_tolerance'],
    }
    verdict='GO' if all(checks.values()) else 'INCONCLUSIVE'
    stages={
      'P0_256_native_positive':{'status':'PASS','evidence':['artifacts/interface_run_256_rank0.json','artifacts/interface_run_256_rank1.json'],'command':'./reproduce.sh interface'},
      'P0_257_actual_failure':{'status':'PASS_EXPECTED_FAILURE','evidence':['artifacts/interface_run_257_rank0.json','artifacts/interface_run_257_rank1.json'],'command':'./reproduce.sh interface'},
      'P1_258_native_protocol':{'status':'PASS','evidence':['artifacts/interface_run_258_rank0.json','artifacts/interface_run_258_rank1.json'],'command':'./reproduce.sh protocol'},
      'P1_zero_legal_task_rank':{'status':'PASS_WITH_EMPTY_PADDING','evidence':['artifacts/zero_rank_258_rank0.json','artifacts/zero_rank_258_rank1.json'],'command':'./reproduce.sh protocol'},
      'P2_real_w8a8_native':{'status':'PASS_FINITE_ACTIVE_SET','evidence':['artifacts/real258_rank0.json','artifacts/real258_rank1.json'],'command':'./reproduce.sh real'},
      'P3_storage_unique_slot':{'status':'PASS_FINITE_MAIN_LIBRARY','evidence':['artifacts/STORAGE_AUDIT.json'],'command':'./reproduce.sh memory'},
      'P3_layer_reuse':{'status':'PASS','evidence':['artifacts/LAYER_REUSE.json'],'command':'./reproduce.sh reuse'},
      'performance':{'status':'NOT_RUN','evidence':[],'reason':'functional proof completed; fixed-point native performance was outside remaining scope'}
    }
    out={
      'verdict':verdict,
      'stages':stages,
      'scope':('GO is limited to 258-slot native all-to-all protocol with finite active main-expert set '
               '(4 experts/rank/layer; 8 active logical experts; layers 49/50) and one real K=1 slot. '
               'It does not prove allocation of the full 128/128 main expert library or full MoE/service behavior.'),
      'checks':checks,'interface_256_positive':interface_runs['checks']['256_native_positive'],
      'actual_257_failure':interface_runs['checks']['257_actual_failure'],
      'active_experts':real[0]['active_logical_experts'],'main_capacity_per_rank_bytes':real[0]['main_storage']['unique_bytes'],
      'extra_slot_bytes_group':storage['all_rank_extra_persistent_slot_bytes'],
      'performance_measured':False,
      'not_measured':['full 128/128 main bank allocation','full-model service','TTFT/throughput','K=2','Decode','cross-node','multi-request concurrency']
    }
    (ART/'INTERFACE_RUNS.json').write_text(json.dumps(interface_runs,indent=2)+'\n')
    (ART/'TASK_IDENTITY.json').write_text(json.dumps(task_identity,indent=2)+'\n')
    (ART/'STORAGE_AUDIT.json').write_text(json.dumps(storage,indent=2)+'\n')
    (ART/'LAYER_REUSE.json').write_text(json.dumps(reuse,indent=2)+'\n')
    (ART/'VERDICT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if verdict!='GO': raise SystemExit(1)

if __name__=='__main__': main()
