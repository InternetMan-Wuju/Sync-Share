#!/usr/bin/env python3
"""Evidence-driven validator for the repaired K=1 adapter campaign.

The validator accepts one explicit campaign directory, checks campaign/subrun
identity, reconstructs expected task identities from raw manifests, and
re-derives every PASS from raw errors, owner state, storage/format records and
input hashes.  A missing record is never treated as a pass.
"""
from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
ATOL=0.02
EXPECTED_CASES=[
 ('both_mixed_q0_noslot','PASS',49,2,2),
 ('no_input_receives_tasks','PASS',49,4,0),
 ('input_but_zero_expert_tasks','PASS',49,4,4),
 ('both_empty','EMPTY',49,0,0),
 ('invalid_rank0','REJECT',49,2,2),
 ('valid_after_invalid_rank0','PASS',49,2,2),
 ('invalid_rank1','REJECT',49,2,2),
 ('valid_after_invalid_rank1','PASS',49,2,2),
 ('reserved_but_claims_ready','REJECT',49,2,2),
 ('valid_after_reserved','PASS',49,2,2),
 ('partial_slot','PASS',49,2,2),
 ('all_slot','PASS',49,2,2),
 ('large_259_partial','PASS',49,259,259),
 ('one_rank_bad_slot','REJECT',49,2,2),
 ('valid_after_one_rank_bad','PASS',49,2,2),
 ('replay_old_49_after_install50','REJECT',49,2,2),
 ('current_50_after_reinstall','PASS',50,2,2),
 ('replay_old_generation_49','REJECT',49,2,2),
 ('current_generation_49','PASS',49,2,2),
 ('fixed_lease_overwrite_blocked','PASS',49,2,2),
]
EXPECTED_INSTALLS=['repaired_install49','repaired_transition50','repaired_reinstall49','repaired_reuse0_l50','repaired_reuse0_l49','repaired_reuse1_l50','repaired_reuse1_l49','repaired_reuse2_l50','repaired_reuse2_l49']
EXPECTED_INSTALL_LAYERS=[49,50,49,50,49,50,49,50,49]
NEGATIVES={'invalid_rank0','invalid_rank1','reserved_but_claims_ready','one_rank_bad_slot','replay_old_49_after_install50','replay_old_generation_49'}


def load(path):
    return json.loads(Path(path).read_text())


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def hidden_sha(path):
    return sha_tensor(torch.load(path,map_location='cpu'))


def plan_task_count(manifests,rank):
    total=sum(int(m['tokens']) for m in manifests)
    plans=[[],[]]
    for sender,m in enumerate(manifests):
        tasks=[]
        if int(m['tokens'])>0:
            tasks=[{'sender':sender,'token':t,'slot':s,'logical':int(m['logical_rows'][t][s]),'comm':int(c),'padding':False} for t,row in enumerate(m['comm_rows']) for s,c in enumerate(row)]
        elif total>0:
            tasks=[{'sender':sender,'token':-1,'slot':s,'logical':None,'comm':128,'padding':True} for s in range(8)]
        tasks.sort(key=lambda x:x['comm'])
        for target in (0,1):
            lo,hi=(0,129) if target==0 else (129,258)
            plans[target].extend([x for x in tasks if lo<=x['comm']<hi])
    for p in plans: p.sort(key=lambda x:x['comm']-((0) if p is plans[0] else 129))
    return len(plans[rank])


def expected_tasks(manifests,rank):
    lo,hi=(0,129) if rank==0 else (129,258)
    return sorted((m['rank'],t,s,int(m['logical_rows'][t][s]),int(c)) for m in manifests for t,row in enumerate(m['comm_rows']) for s,c in enumerate(row) if lo<=c<hi)


def actual_tasks(rec):
    return sorted((t['sender'],t['token'],t['slot'],t['logical'],t['comm']) for t in rec.get('actual_tasks',[]) if not t.get('padding'))


def collect(campaign_dir):
    campaign_dir=Path(campaign_dir)
    ds={'campaign_dir':str(campaign_dir),'manifest':load(campaign_dir/'campaign_manifest.json'),'functional_dir':campaign_dir/'functional','memory':{},'installs':[],'hidden_sha':{}}
    for r in (0,1):
        ds.setdefault('functional',{})[r]=load(campaign_dir/'functional'/f'functional_{r}.json')
    for scen,dirname,prefix in [('baseline','memory-baseline','baseline'),('candidate','memory-candidate','candidate')]:
        ds['memory'][scen]={r:load(campaign_dir/dirname/f'memory_{prefix}_rank{r}.json') for r in (0,1)}
    for tag in EXPECTED_INSTALLS:
        p=campaign_dir/'functional'
        ds['installs'].append({'tag':tag,'rank0':load(p/f'{tag}_install_rank0.json'),'rank1':load(p/f'{tag}_install_rank1.json')})
    case_records={}
    for cid,_,layer,_,_ in EXPECTED_CASES:
        for r in (0,1):
            case_records[(cid,r)]=load(campaign_dir/'functional'/f'{cid}_l{layer}_rank{r}.json')
            hp=campaign_dir/'functional'/f'{cid}_l{layer}_hidden_rank{r}.pt'
            if hp.exists(): ds['hidden_sha'][(cid,r)]=hidden_sha(hp)
    for cid,_,layer,_,_ in EXPECTED_CASES:
        recs=[ds['functional'][r] for r in (0,1)]
        for r in (0,1):
            row=next((x for x in recs[r].get('matrix',[]) if x.get('case_id')==cid),None)
            if row is None: raise KeyError(f'missing case {cid} rank{r}')
            ds.setdefault('cases',{})[(cid,r)]={'record':row,'manifests':[case_records[(cid,i)] for i in (0,1)]}
    ds['reuse_manifest']=campaign_dir/'functional'
    for r in (0,1):
        for item in ds['functional'][r].get('reuse',[]):
            cid=item['result']['case_id']; layer=item['layer']
            hp=campaign_dir/'functional'/f'{cid}_l{layer}_hidden_rank{r}.pt'
            if hp.exists(): ds['hidden_sha'][(cid,r)]=hidden_sha(hp)
    return ds


def validate(ds):
    e={k:[] for k in ('CAMPAIGN','INPUT_FIDELITY','OWNER_STATE','MEMORY_ACCOUNTING')}
    def add(k,m): e[k].append(m)
    mf=ds.get('manifest',{}); cid=mf.get('campaign_id')
    if not cid: add('CAMPAIGN','missing_campaign_id')
    if mf.get('expected_install_tags')!=EXPECTED_INSTALLS: add('CAMPAIGN','manifest_install_tags')
    if mf.get('expected_cases')!=[x[0] for x in EXPECTED_CASES]: add('CAMPAIGN','manifest_case_list')
    code=mf.get('code_hash'); subruns=mf.get('subruns',{})
    required_subruns={'functional','memory-baseline','memory-candidate'}
    if not required_subruns.issubset(subruns): add('CAMPAIGN','missing_subrun_identity')
    for r in (0,1):
        f=ds.get('functional',{}).get(r,{})
        if f.get('campaign_id')!=cid: add('CAMPAIGN',f'functional_campaign_rank{r}')
        if f.get('subrun_id')!=subruns.get('functional'): add('CAMPAIGN',f'functional_subrun_rank{r}')
        if f.get('code_hash')!=code: add('CAMPAIGN',f'functional_code_hash_rank{r}')
        if f.get('rank')!=r: add('CAMPAIGN',f'functional_rank_identity_{r}')
        if int(f.get('dispatch_capacity',0))<512: add('CAMPAIGN',f'dispatch_capacity_rank{r}')
    # Installs: real layer/generation/storage/format sequence.
    installs=ds.get('installs',[])
    if len(installs)!=len(EXPECTED_INSTALLS): add('MEMORY_ACCOUNTING',f'install_count:{len(installs)}')
    layers=[]; gens=[]; ptrs=set(); bytes_seen=set(); pool_ids=set()
    for item in installs:
        a,b=item['rank0'],item['rank1']; layers.append(b.get('layer')); gens.append(b.get('generation'))
        for rec in (a,b):
            if rec.get('campaign_id')!=cid or rec.get('code_hash')!=code or rec.get('subrun_id')!=subruns.get('functional'): add('CAMPAIGN',f'install_identity:{item["tag"]}')
        if b.get('ready') is not True: add('OWNER_STATE',f'install_not_ready:{item["tag"]}')
        if not a.get('source_hash') or a.get('source_hash')!=b.get('target_hash') or not a.get('hash_equal') or not b.get('hash_equal'): add('MEMORY_ACCOUNTING',f'install_hash:{item["tag"]}')
        ptrs.add(b.get('slot_storage_ptr')); bytes_seen.add(b.get('slot_storage_bytes')); pool_ids.add(b.get('pool_id'))
        if not b.get('slot_formats'): add('MEMORY_ACCOUNTING',f'missing_slot_formats:{item["tag"]}')
        for fmt in b.get('slot_formats',[]):
            if 'ERROR' in str(fmt.get('format_string')) or 'ERROR' in str(fmt.get('format_code')): add('MEMORY_ACCOUNTING',f'format_error:{item["tag"]}:{fmt.get("name")}')
            if fmt.get('aliases_slot_storage') is not True: add('MEMORY_ACCOUNTING',f'slot_view_not_alias:{item["tag"]}:{fmt.get("name")}')
    if layers!=EXPECTED_INSTALL_LAYERS: add('MEMORY_ACCOUNTING',f'install_layer_sequence:{layers}')
    if not gens or any(int(g)<=0 for g in gens) or any(int(b)<=int(a) for a,b in zip(gens,gens[1:])): add('MEMORY_ACCOUNTING',f'install_generations:{gens}')
    if len(ptrs)!=1 or None in ptrs: add('MEMORY_ACCOUNTING',f'slot_storage_identity:{sorted(map(str,ptrs))}')
    if len(bytes_seen)!=1 or next(iter(bytes_seen),0)<=0: add('MEMORY_ACCOUNTING',f'slot_bytes:{bytes_seen}')
    if len(pool_ids)!=1 or None in pool_ids: add('MEMORY_ACCOUNTING',f'owner_pool_identity:{pool_ids}')

    # Functional cases and input/task evidence.
    expected_names=[x[0] for x in EXPECTED_CASES]
    for r in (0,1):
        got=[x.get('case_id') for x in ds.get('functional',{}).get(r,{}).get('matrix',[])]
        if got!=expected_names: add('INPUT_FIDELITY',f'case_order_or_missing_rank{r}:{got}')
        if len(got)!=len(set(got)): add('INPUT_FIDELITY',f'duplicate_cases_rank{r}')
    for name,kind,layer,n0,n1 in EXPECTED_CASES:
        rows=[]
        for r in (0,1):
            cell=ds.get('cases',{}).get((name,r));
            if not cell: add('INPUT_FIDELITY',f'missing_case:{name}:rank{r}'); continue
            rec=cell['record']; mans=cell['manifests']; rows.append(rec)
            if rec.get('campaign_id')!=cid or rec.get('code_hash')!=code or rec.get('rank')!=r: add('CAMPAIGN',f'case_identity:{name}:rank{r}')
            if int(mans[0].get('tokens',-1))!=n0 or int(mans[1].get('tokens',-1))!=n1: add('INPUT_FIDELITY',f'case_token_manifest:{name}')
            hs=ds.get('hidden_sha',{}).get((name,r))
            if hs is None: add('INPUT_FIDELITY',f'missing_hidden_file:{name}:rank{r}')
            elif rec.get('input_sha256_before')!=hs or rec.get('input_sha256_after')!=hs: add('INPUT_FIDELITY',f'hidden_file_changed:{name}:rank{r}')
            if rec.get('input_unchanged') is not True: add('INPUT_FIDELITY',f'caller_input_changed:{name}:rank{r}')
            if kind=='REJECT':
                if rec.get('status')!='REJECTED' or rec.get('dispatch_delta')!=0: add('OWNER_STATE',f'negative_not_rejected_without_dispatch:{name}:rank{r}')
            elif kind=='EMPTY':
                if rec.get('status')!='PASS_EMPTY' or rec.get('dispatch_delta')!=0: add('INPUT_FIDELITY',f'empty_not_early_exit:{name}:rank{r}')
            else:
                if rec.get('actual_expert_inputs_equal') is not True or rec.get('actual_input_mismatches'): add('INPUT_FIDELITY',f'actual_expert_input_mismatch:{name}:rank{r}')
                if rec.get('status')!='PASS': add('INPUT_FIDELITY',f'pass_status:{name}:rank{r}')
                err=rec.get('reference_max_abs'); rel=rec.get('reference_relative_error')
                if err is None or rel is None or not (err<=ATOL and rel<=ATOL): add('INPUT_FIDELITY',f'error_over_tolerance:{name}:rank{r}:{err}/{rel}')
                if rec.get('reference_finite') is not True or rec.get('candidate_finite') is not True: add('INPUT_FIDELITY',f'nonfinite:{name}:rank{r}')
                exp=expected_tasks(mans,r); act=actual_tasks(rec)
                if exp!=act or rec.get('task_multiset_equal') is not True: add('INPUT_FIDELITY',f'task_identity_mismatch:{name}:rank{r}:{len(exp)}!={len(act)}')
                expected_rows=plan_task_count(mans,r)
                audit=rec.get('actual_input_rows',[])
                if len(audit)!=expected_rows or not all(x.get('equal') is True for x in audit): add('INPUT_FIDELITY',f'per_row_input_audit:{name}:rank{r}')
                nonpad=sorted((x['sender'],x['token'],x['slot'],None,x['comm']) for x in audit if not x.get('padding'))
                if [x[:3]+x[4:] for x in act] != [x[:3]+x[4:] for x in nonpad]: add('INPUT_FIDELITY',f'input_row_task_order:{name}:rank{r}')
        if len(rows)!=2:
            continue
        if rows[0].get('status')!=rows[1].get('status'): add('OWNER_STATE',f'rank_status_disagreement:{name}')
        if name=='large_259_partial':
            total_act=sum(len(actual_tasks(x)) for x in rows); total_exp=sum(len(expected_tasks(ds['cases'][(name,r)]['manifests'],r)) for r in (0,1))
            if total_exp!=4144 or total_act!=4144: add('INPUT_FIDELITY',f'large_259_task_total:{total_exp}/{total_act}')
            if rows[0].get('global_expected_task_count')!=4144: add('INPUT_FIDELITY','large_259_global_count')
        if name=='no_input_receives_tasks':
            if len(actual_tasks(rows[1]))==0: add('INPUT_FIDELITY','no_input_rank_did_not_receive_tasks')
        if name=='input_but_zero_expert_tasks':
            if len(actual_tasks(rows[1]))!=0: add('INPUT_FIDELITY','zero_expert_task_rank_has_tasks')
            if rows[1].get('local_expected_task_count')!=0: add('INPUT_FIDELITY','zero_expert_task_claim')
    # Real owner records for required negatives and the fixed-lease overwrite guard.
    def owner(name):
        cell=ds.get('cases',{}).get((name,1))
        return cell['record'].get('owner_slot_snapshot') if cell else None
    snap=owner('reserved_but_claims_ready')
    if not snap or snap.get('actual_reserved',0)<1 or snap.get('actual_ready_count')!=0 or not any(x.get('state')=='reserved' for x in snap.get('records',[])): add('OWNER_STATE','reserved_negative_not_real_reserved')
    if not snap or not all(x.get('ready') is True for x in snap.get('requests',[])): add('OWNER_STATE','reserved_negative_request_not_claiming_ready')
    snap=owner('one_rank_bad_slot')
    if not snap or len({x.get('generation') for x in snap.get('requests',[])})<2: add('OWNER_STATE','one_rank_bad_not_one_sided')
    snap=owner('replay_old_49_after_install50')
    if not snap or snap.get('lease_layer')!=50 or not any(x.get('layer')==49 for x in snap.get('requests',[])): add('OWNER_STATE','cross_layer_replay_not_real_state')
    snap=owner('replay_old_generation_49')
    if not snap or not any(x.get('generation')!=snap.get('lease_generation') for x in snap.get('requests',[])): add('OWNER_STATE','stale_generation_replay_not_real_state')
    guard=ds.get('cases',{}).get(('fixed_lease_overwrite_blocked',1),{}).get('record',{}).get('lease_overwrite_guard',{})
    during=guard.get('actual_records_during_use',[])
    if not guard.get('called') or guard.get('errors') or not any(x.get('state')=='in_use' and x.get('users',0)>=1 for x in during): add('OWNER_STATE','fixed_lease_guard_not_proven')
    # Reuse records.
    reuse0=ds.get('functional',{}).get(0,{}).get('reuse',[]); reuse1=ds.get('functional',{}).get(1,{}).get('reuse',[])
    if len(reuse0)!=6 or len(reuse1)!=6: add('MEMORY_ACCOUNTING','reuse_count')
    else:
        if [x.get('layer') for x in reuse0]!=[50,49,50,49,50,49]: add('MEMORY_ACCOUNTING','reuse_layer_sequence')
        for a,b in zip(reuse0,reuse1):
            for rec in (a['result'],b['result']):
                if rec.get('status')!='PASS' or rec.get('within_tolerance') is not True or rec.get('task_multiset_equal') is not True or rec.get('input_unchanged') is not True or rec.get('actual_expert_inputs_equal') is not True: add('MEMORY_ACCOUNTING',f'reuse_failure:{rec.get("case_id")}')
                hs=ds.get('hidden_sha',{}).get((rec.get('case_id'),rec.get('rank')))
                if hs is None or rec.get('input_sha256_before')!=hs: add('INPUT_FIDELITY',f'reuse_input_hash:{rec.get("case_id")}')
    final=ds.get('functional',{}).get(1,{}).get('final_slot',{})
    if final.get('pool',{}).get('active_bytes')!=0 or not all(x.get('state')=='free' for x in final.get('records',[])): add('OWNER_STATE','final_slot_not_released')
    # Independent memory processes.
    mem=ds.get('memory',{})
    for scen in ('baseline','candidate'):
        for r in (0,1):
            rec=mem.get(scen,{}).get(r,{})
            if rec.get('campaign_id')!=cid or rec.get('subrun_id')!=subruns.get(f'memory-{scen}') or rec.get('code_hash')!=code or rec.get('rank')!=r: add('CAMPAIGN',f'memory_identity:{scen}:rank{r}')
            names=[x.get('stage') for x in rec.get('stage_records',[])]
            required=['resources_created','common_init'] + (['install','q0','split','release'] if scen=='candidate' else ['q0','release'])
            if names!=required: add('MEMORY_ACCOUNTING',f'memory_stage_order:{scen}:rank{r}:{names}')
            for st in rec.get('stage_records',[]):
                if not isinstance(st.get('peak_allocated'),int) or not isinstance(st.get('peak_reserved'),int): add('MEMORY_ACCOUNTING',f'missing_stage_peak:{scen}:rank{r}:{st.get("stage")}')
            q0=rec.get('q0',{})
            if q0.get('status')!='PASS' or q0.get('task_multiset_equal') is not True or q0.get('input_unchanged') is not True or q0.get('actual_expert_inputs_equal') is not True: add('MEMORY_ACCOUNTING',f'memory_q0_failure:{scen}:rank{r}')
            if scen=='candidate':
                sp=rec.get('split',{})
                if sp.get('status')!='PASS' or sp.get('task_multiset_equal') is not True or sp.get('input_unchanged') is not True or sp.get('actual_expert_inputs_equal') is not True: add('MEMORY_ACCOUNTING',f'memory_split_failure:rank{r}')
    b0=mem.get('baseline',{}).get(0,{}); b1=mem.get('baseline',{}).get(1,{})
    c0=mem.get('candidate',{}).get(0,{}); c1=mem.get('candidate',{}).get(1,{})
    if b0.get('flat_buffer') is not None or b0.get('resource_objects',{}).get('rank0_flat') is not None: add('MEMORY_ACCOUNTING','baseline_rank0_has_flat')
    if b1.get('resource_objects',{}).get('rank1_pool') is not None or b1.get('slot_before_release') is not None: add('MEMORY_ACCOUNTING','baseline_rank1_has_pool')
    sb=int(c0.get('state_bytes',0));
    if sb<=0 or c0.get('flat_buffer',{}).get('bytes')!=sb: add('MEMORY_ACCOUNTING',f'candidate_source_flat_bytes:{c0.get("flat_buffer")}/{sb}')
    if c1.get('resource_objects',{}).get('rank1_pool',{}).get('committed_bytes')!=sb or c1.get('resource_objects',{}).get('rank1_pool',{}).get('K')!=1: add('MEMORY_ACCOUNTING','candidate_pool_identity_or_bytes')
    if (c1.get('slot_before_release') or {}).get('ready') is not True: add('MEMORY_ACCOUNTING','candidate_slot_not_ready_before_release')
    if not all(x.get('state')=='free' for x in c1.get('final_slot',{}).get('records',[])): add('MEMORY_ACCOUNTING','memory_candidate_slot_not_released')
    for r in (0,1):
        if mem['baseline'][r].get('main_storage_bytes')!=mem['candidate'][r].get('main_storage_bytes'): add('MEMORY_ACCOUNTING',f'main_storage_mismatch_rank{r}')
        peaks=[x.get('peak_allocated',0) for x in mem['candidate'][r].get('stage_records',[])]
        if peaks and mem['candidate'][r].get('whole_process_peak',{}).get('peak_allocated',0)<max(peaks): add('MEMORY_ACCOUNTING',f'whole_peak_too_small_rank{r}')
    # Candidate must not expose a second executable persistent replica.
    candidate_slot_ptrs={c1.get('install',{}).get('slot_storage_ptr')} if c1.get('install') else set()
    if len(candidate_slot_ptrs)!=1 or c1.get('install',{}).get('slot_storage_ptr') not in ptrs: add('MEMORY_ACCOUNTING','candidate_slot_storage_not_unique')
    if any('ERROR' in str(x.get('format_string')) for f in ds['functional'].values() for x in f.get('bank_formats',[])): add('MEMORY_ACCOUNTING','bank_format_error')
    return e


def flat(e): return [f'{k}:{x}' for k,v in e.items() for x in v]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('campaign_dir'); args=ap.parse_args()
    campaign=Path(args.campaign_dir).resolve(); ds=collect(campaign)
    areas=validate(ds); errors=flat(areas)
    # Deterministic mutation fixtures on copies of the collected evidence.
    def mutation(name,fn):
        d=copy.deepcopy(ds); fn(d); errs=flat(validate(d)); return name,errs
    def stale_id(d): d['functional'][0]['campaign_id']='stale'
    def changed_input(d): d['hidden_sha'][('partial_slot',0)]='0'*64
    def bad_slot(d): d['cases'][('reserved_but_claims_ready',1)]['record']['owner_slot_snapshot']['actual_reserved']=0
    def excess_error(d): d['cases'][('all_slot',0)]['record']['reference_max_abs']=99.0
    def format_error(d): d['installs'][2]['rank1']['slot_formats'][0]['format_string']='ERROR:test'
    def task_mismatch(d): d['cases'][('partial_slot',0)]['record']['actual_tasks'].pop()
    def baseline_flat(d): d['memory']['baseline'][0]['flat_buffer']={'bytes':1,'storage_ptr':1}
    def missing_rank(d): del d['functional'][1]
    def missing_install(d): d['installs'].pop()
    def bad_generation(d): d['installs'][4]['rank1']['generation']=99
    def unready_final(d): d['functional'][1]['final_slot']['pool']['active_bytes']=1
    regressions=[]
    for name,fn in [('stale_campaign_or_run_id',stale_id),('changed_input',changed_input),('wrong_owner_slot_state',bad_slot),('excess_error',excess_error),('format_ERROR',format_error),('task_not_conserved',task_mismatch),('baseline_contains_flat',baseline_flat),('missing_rank',missing_rank),('missing_install',missing_install),('bad_generation',bad_generation),('unready_final',unready_final)]:
        n,errs=mutation(name,fn); regressions.append({'mutation':n,'rejected':bool(errs),'errors':errs})
    regression_ok=all(x['rejected'] for x in regressions)
    stages={k:('PASS' if not v else 'FAIL') for k,v in areas.items() if k!='CAMPAIGN'}
    if areas['CAMPAIGN']: stages={k:'FAIL' for k in stages}
    if not regression_ok: stages['VALIDATOR']='FAIL'
    else: stages['VALIDATOR']='PASS'
    # CAMPAIGN is not one of the four user-facing stages; identity failures fail all.
    verdict='GO_REPAIRED_HARNESS_LIMITED' if all(stages.get(x)=='PASS' for x in ('INPUT_FIDELITY','OWNER_STATE','MEMORY_ACCOUNTING','VALIDATOR')) else ('INCONCLUSIVE' if not errors else 'FAIL')
    f=ds['functional']; installs=ds['installs']; mem=ds['memory']
    input_report={'status':stages['INPUT_FIDELITY'],'large_case':next(x for x in f[0]['matrix'] if x['case_id']=='large_259_partial'),'cases':{x['case_id']:{str(r):{k:f[r]['matrix'][i].get(k) for k in ('status','input_unchanged','input_sha256_before','input_sha256_after','actual_expert_inputs_equal','actual_input_mismatches','expected_task_count','actual_task_count','task_multiset_equal','reference_max_abs','reference_relative_error','local_expected_task_count','global_expected_task_count')} for r in (0,1)} for i,x in enumerate(f[0]['matrix'])}}
    owner_report={'status':stages['OWNER_STATE'],'installs':[{'tag':x['tag'],'layer':x['rank1'].get('layer'),'generation':x['rank1'].get('generation'),'ready':x['rank1'].get('ready'),'pool_id':x['rank1'].get('pool_id'),'source_hash':x['rank0'].get('source_hash'),'target_hash':x['rank1'].get('target_hash'),'hash_equal':x['rank0'].get('hash_equal') and x['rank1'].get('hash_equal'),'slot_storage_ptr':x['rank1'].get('slot_storage_ptr'),'slot_storage_bytes':x['rank1'].get('slot_storage_bytes'),'slot_formats':x['rank1'].get('slot_formats')} for x in installs],'negatives':{n:{str(r):{k:ds['cases'][(n,r)]['record'].get(k) for k in ('status','dispatch_delta','reasons','owner_slot_snapshot','lease_overwrite_guard')} for r in (0,1)} for n in sorted(NEGATIVES)},'guard':ds['cases'][('fixed_lease_overwrite_blocked',1)]['record'].get('lease_overwrite_guard'),'final_slot':f[1].get('final_slot')}
    stage_summary={scen:{str(r):mem[scen][r].get('stage_records') for r in (0,1)} for scen in mem}
    storage={'status':stages['MEMORY_ACCOUNTING'],'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','state_bytes':mem['candidate'][0].get('state_bytes'),'main_storage_by_rank':{str(r):mem['candidate'][r].get('main_storage_bytes') for r in (0,1)},'slot':{'count':1,'owner_rank':1,'storage_ptr':installs[-1]['rank1'].get('slot_storage_ptr'),'bytes':installs[-1]['rank1'].get('slot_storage_bytes'),'formats':installs[-1]['rank1'].get('slot_formats')},'source_flat_rank0':mem['candidate'][0].get('flat_buffer'),'baseline_resources':{str(r):mem['baseline'][r].get('resource_objects') for r in (0,1)},'candidate_resources':{str(r):mem['candidate'][r].get('resource_objects') for r in (0,1)},'stage_records':stage_summary,'whole_process_peak':{str(r):mem['candidate'][r].get('whole_process_peak') for r in (0,1)},'baseline_current_final':{str(r):mem['baseline'][r].get('memory_final') for r in (0,1)},'candidate_current_final':{str(r):mem['candidate'][r].get('memory_final') for r in (0,1)}}
    reuse={'status':stages['MEMORY_ACCOUNTING'],'installs':owner_report['installs'],'rank0_runs':f[0].get('reuse'),'rank1_runs':f[1].get('reuse'),'final_slot':f[1].get('final_slot')}
    out={'verdict':verdict,'stages':stages,'validation_errors':errors,'area_errors':areas,'regressions':regressions,'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','performance_status':'NOT_RUN','scope':'synthetic hidden/top8, finite main bank, layer49/50, expert43, two Ascend 910C, native DeepEP alltoall, real W8A8'}
    root=ROOT
    for name,obj in [('INPUT_FIDELITY.json',input_report),('OWNER_STATE_CASES.json',owner_report),('STORAGE_AUDIT.json',storage),('LAYER_REUSE.json',reuse),('VALIDATOR_REGRESSIONS.json',{'status':stages['VALIDATOR'],'regressions':regressions}),('VERDICT.json',out),('MANIFEST.json',{'campaign_id':ds['manifest'].get('campaign_id'),'campaign_dir':str(campaign),'code_hash':ds['manifest'].get('code_hash'),'subruns':ds['manifest'].get('subruns'),'image':ds['manifest'].get('image'),'devices':ds['manifest'].get('devices'),'started_at':ds['manifest'].get('started_at'),'finished_at':__import__('datetime').datetime.now().astimezone().isoformat(),'verdict':verdict,'production_format_config':'UNKNOWN_NO_ACTIVE_SERVICE','performance_status':'NOT_RUN'})]:
        text=json.dumps(obj,indent=2,default=str)+'\n'; (root/name).write_text(text); (campaign/name).write_text(text)
    print(json.dumps(out,indent=2))
    if verdict!='GO_REPAIRED_HARNESS_LIMITED': raise SystemExit(1)

if __name__=='__main__': main()
