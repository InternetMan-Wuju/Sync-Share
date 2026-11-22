#!/usr/bin/env python3
from __future__ import annotations
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'
def load(n): return json.loads((ART/n).read_text())

def collect_installs():
    pairs=[]
    tags=['unified_install49']+[f'unified_reuse{c}_l{l}' for c in range(3) for l in (50,49)]
    for tag in tags:
        r0=load(f'{tag}_install_rank0.json'); r1=load(f'{tag}_install_rank1.json')
        pairs.append({'tag':tag,'layer':r1['layer'],'generation':r1['generation'],'ready':r1['ready'],'source_hash':r0.get('source_hash'),'target_hash':r1.get('target_hash'),'hash_equal':r0.get('source_hash')==r1.get('target_hash'),'slot_storage_ptr':r1.get('slot_storage_ptr'),'slot_storage_bytes':r1.get('slot_storage_bytes'),'slot_formats':r1.get('slot_formats'),'install_ms_source':r0.get('install_ms'),'install_ms_target':r1.get('install_ms')})
    return pairs

def validate(dataset):
    errors=[]; f=dataset.get('functional',{})
    if set(f)!={0,1}: errors.append('missing_rank_functional')
    if errors: return errors
    c0={x['case_id']:x for x in f[0]['matrix']}; c1={x['case_id']:x for x in f[1]['matrix']}
    expected=['both_mixed_q0_noslot','rank0_only','rank1_only','both_empty','invalid_rank0','valid_after_invalid_rank0','invalid_rank1','valid_after_invalid_rank1','unready_slot','valid_after_unready','partial_slot','all_slot','wrong_layer','valid_after_wrong_layer','stale_generation','valid_after_stale']
    if list(c0)!=expected or list(c1)!=expected: errors.append('case_list_or_order')
    for name in expected:
        if name not in c0 or name not in c1: errors.append(f'missing_case:{name}'); continue
        a,b=c0[name],c1[name]
        negative=name in ('invalid_rank0','invalid_rank1','unready_slot','wrong_layer','stale_generation')
        if negative:
            if a['status']!='REJECTED' or b['status']!='REJECTED': errors.append(f'negative_not_rejected:{name}')
            if a.get('dispatch_delta')!=0 or b.get('dispatch_delta')!=0: errors.append(f'negative_dispatched:{name}')
        elif name=='both_empty':
            if a['status']!='PASS_EMPTY' or b['status']!='PASS_EMPTY': errors.append('empty_not_early_exit')
            if a.get('dispatch_delta')!=0 or b.get('dispatch_delta')!=0: errors.append('empty_dispatched')
        else:
            if a['status']!='PASS' or b['status']!='PASS': errors.append(f'case_not_pass:{name}')
            if not a.get('within_tolerance',False) or not b.get('within_tolerance',False): errors.append(f'tolerance:{name}')
            if not a.get('task_multiset_equal',False) or not b.get('task_multiset_equal',False): errors.append(f'tasks:{name}')
            if a.get('empty_legal_tasks',1)!=0 or b.get('empty_legal_tasks',1)!=0: errors.append(f'empty_legal_task:{name}')
    if c0['rank0_only']['padding_sent']!=0 or c1['rank0_only']['padding_sent']!=1: errors.append('rank0_only_padding')
    if c1['rank0_only'].get('padding_expert_calls')!=0 or c0['rank0_only'].get('padding_received',0)==0: errors.append('rank0_only_padding_compute')
    if c1['rank1_only']['padding_sent']!=0 or c0['rank1_only']['padding_sent']!=1: errors.append('rank1_only_padding')
    if c0['rank1_only'].get('padding_expert_calls')!=0 or c0['rank1_only'].get('padding_received',0)==0 or c1['rank1_only'].get('padding_sent')!=0: errors.append('rank1_only_padding_compute')
    if sum(x.get('dispatch_delta',0) for x in f[0]['matrix'])+sum(1 for _ in f[0].get('reuse',[])) != f[0].get('dispatch_calls'): errors.append('dispatch_call_accounting')
    slots=dataset.get('installs',[])
    if len(slots)!=7: errors.append(f'install_count:{len(slots)}')
    if [x['layer'] for x in slots] != [49,50,49,50,49,50,49]: errors.append('install_layer_sequence')
    if [x['generation'] for x in slots] != list(range(1,8)): errors.append('install_generations')
    if not all(x.get('ready') and x.get('hash_equal') for x in slots): errors.append('install_ready_or_hash')
    ptrs={x.get('slot_storage_ptr') for x in slots}
    if len(ptrs)!=1: errors.append('slot_storage_identity')
    if any(x.get('slot_storage_bytes')!=37810176 for x in slots): errors.append('slot_bytes')
    if f[1].get('final_slot',{}).get('pool',{}).get('active_bytes')!=0 or f[1].get('final_slot',{}).get('pool',{}).get('local',{}).get('1',{}).get('free')!=1: errors.append('final_slot_not_released')
    reuse0=f[0].get('reuse',[]); reuse1=f[1].get('reuse',[])
    if len(reuse0)!=6 or len(reuse1)!=6: errors.append('reuse_count')
    if [x['layer'] for x in reuse0] != [50,49,50,49,50,49]: errors.append('reuse_sequence')
    if not all(x['result'].get('within_tolerance') for x in reuse0+reuse1): errors.append('reuse_tolerance')
    if not all(x['result'].get('replica_tasks',0)>0 for x in reuse1): errors.append('reuse_replica_tasks')
    mem=dataset.get('memory',{})
    if set(mem)!={'baseline','candidate'} or set(mem['baseline'])!={0,1} or set(mem['candidate'])!={0,1}: errors.append('memory_missing')
    else:
        if mem['baseline'][0]['main_storage_bytes']!=mem['candidate'][0]['main_storage_bytes'] or mem['baseline'][1]['main_storage_bytes']!=mem['candidate'][1]['main_storage_bytes']: errors.append('memory_main_mismatch')
        if mem['candidate'][1].get('slot_storage',{}).get('ready') is not True: errors.append('memory_candidate_slot')
    fmt=dataset.get('format',{})
    if fmt.get('production_format_config')!='UNKNOWN_NO_ACTIVE_SERVICE': errors.append('production_format_unknown')
    if not fmt.get('bank_formats') or not fmt.get('slot_formats'): errors.append('format_missing')
    return errors

def main():
    f={r:load(f'functional_{r}.json') for r in (0,1)}
    mem={s:{r:load(f'memory_{s}_rank{r}.json') for r in (0,1)} for s in ('baseline','candidate')}
    installs=collect_installs()
    slot_formats=[]
    for x in installs: slot_formats.extend(x.get('slot_formats',[]))
    format_audit={'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','runtime_observed_format':'base/ND code 2 for all audited bank and slot tensors','bank_formats':{str(r):f[r]['bank_formats'] for r in (0,1)},'conversion_alias':f[0]['conversion_alias'],'slot_formats':slot_formats,'internal_format_warning_observed':True,'note':'Actual isolated harness used base-format tensors. Production format cannot be confirmed without a production service/configuration; NZ was not inferred.'}
    flat=f[0].get('flat_buffer')
    storage={'main_storage_by_rank':{str(r):f[r]['main_storage'] for r in (0,1)},'unique_slot_storage_pointer':sorted({x['slot_storage_ptr'] for x in installs}),'slot_storage_bytes':37810176,'extra_persistent_slot_count':len({x['slot_storage_ptr'] for x in installs}),'empty_comm_state_bytes':0,'source_flat_buffer_rank0':flat,'memory_baseline':{str(r):mem['baseline'][r] for r in (0,1)},'memory_candidate':{str(r):mem['candidate'][r] for r in (0,1)},'candidate_minus_baseline_final_allocated':{str(r):mem['candidate'][r]['memory_final']['allocated']-mem['baseline'][r]['memory_final']['allocated'] for r in (0,1)},'candidate_minus_baseline_peak_allocated':{str(r):mem['candidate'][r]['memory_final']['max_allocated']-mem['baseline'][r]['memory_final']['max_allocated'] for r in (0,1)},'note':'Reference-only tensors and allocator reserved bytes remain classified separately from persistent expert weights.'}
    reuse={'installs':installs,'rank0_runs':f[0]['reuse'],'rank1_runs':f[1]['reuse'],'final_slot':f[1]['final_slot'],'cycles':3}
    dataset={'functional':f,'installs':installs,'memory':mem,'format':format_audit}
    errors=validate(dataset)
    # Accepted-evidence regression tests: mutate copies only.
    mutations={'missing_rank':{'functional':{0:f[0]},'installs':installs,'memory':mem,'format':format_audit},'missing_install':{'functional':f,'installs':installs[:-1],'memory':mem,'format':format_audit},'bad_generation':{'functional':f,'installs':copy.deepcopy(installs),'memory':mem,'format':format_audit},'unready_final':{'functional':copy.deepcopy(f),'installs':installs,'memory':mem,'format':format_audit}}
    mutations['bad_generation']['installs'][3]['generation']=99
    mutations['unready_final']['functional'][1]['final_slot']['pool']['active_bytes']=37810176
    regression={k:validate(v) for k,v in mutations.items()}
    regression_ok=all(bool(v) for v in regression.values())
    cases={'run_id':f[0]['run_id'],'rank0':f[0]['matrix'],'rank1':f[1]['matrix'],'dispatch_calls_rank0':f[0]['dispatch_calls'],'dispatch_calls_rank1':f[1]['dispatch_calls']}
    stages={
      'functional':'PASS' if not errors and regression_ok else 'FAIL',
      'format_resource':'PASS_HARNESS_BASE_FORMAT_PRODUCTION_UNKNOWN' if not errors and regression_ok else 'FAIL',
      'reuse':'PASS' if not errors else 'FAIL',
      'performance':'NOT_RUN'
    }
    verdict={'verdict':'GO_FUNCTIONAL_LIMITED' if stages['functional']=='PASS' else 'INCONCLUSIVE','stages':stages,'validation_errors':errors,'validator_regressions':regression,'functional_scope':'finite 8+8 main experts per layer, synthetic hidden/top8, 258 native all-to-all, real ModelSlim W8A8','format_scope':'actual harness base/ND; production unknown','performance_status':'NOT_RUN','next_blocker':'production-format/config confirmation and full grouped/complete-layer integration'}
    (ART/'CASE_RESULTS.json').write_text(json.dumps(cases,indent=2)+'\n')
    (ART/'FORMAT_AUDIT.json').write_text(json.dumps(format_audit,indent=2)+'\n')
    (ART/'STORAGE_AUDIT.json').write_text(json.dumps(storage,indent=2)+'\n')
    (ART/'LAYER_REUSE.json').write_text(json.dumps(reuse,indent=2)+'\n')
    (ART/'VERDICT.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict,indent=2))
    if stages['functional']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
