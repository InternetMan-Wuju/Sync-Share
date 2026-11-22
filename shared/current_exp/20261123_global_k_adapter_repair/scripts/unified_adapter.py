#!/usr/bin/env python3
"""Unified 258-slot adapter with side metadata, owner-state validation and real W8A8."""
from __future__ import annotations
import copy, hashlib, json, time
from pathlib import Path
import torch, torch_npu, torch.distributed as dist
from global_slot_pool import GlobalSlotPool
from real_expert_chain_rework import LAYER_ATTRS, PARTS, ExpertLayer, RuntimeChain, alloc, external_state_from_bank, load_raw_state, memory_snapshot, slot_state_views, state_layout

EMPTY_COMM=128
REPLICA_COMM=257
BASE_COMM={0:0,1:129}
PADDING_TOKEN=-1
TOP_K=8


def logical_to_comm(e): return e if e<128 else e+1

def comm_to_logical(c):
    if 0<=c<128: return c
    if 129<=c<=256: return c-1
    if c==REPLICA_COMM: return 43
    raise ValueError(f'invalid comm id {c}')

def sha(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def plan_tasks(manifests):
    """Build task order from sender plans; padding is generated here, not by callers."""
    total=sum(int(m['tokens']) for m in manifests)
    sender_tasks=[]
    for sender,m in enumerate(manifests):
        tasks=[]
        if int(m['tokens'])>0:
            for t,row in enumerate(m['comm_rows']):
                for s,c in enumerate(row):
                    tasks.append({'sender':sender,'token':t,'slot':s,'logical':int(m['logical_rows'][t][s]),'comm':int(c),'padding':False,'padding_id':None})
        elif total>0:
            for s in range(TOP_K):
                tasks.append({'sender':sender,'token':PADDING_TOKEN,'slot':s,'logical':None,'comm':EMPTY_COMM,'padding':True,'padding_id':f'pad-s{sender}-r{sender}-{s}'})
        # Native alltoall strategy sorts each sender's tasks by global expert id.
        tasks.sort(key=lambda x:x['comm'])
        sender_tasks.append(tasks)
    plans=[[],[]]
    for sender,tasks in enumerate(sender_tasks):
        for target in (0,1):
            lo,hi=(0,129) if target==0 else (129,258)
            plans[target].extend([x for x in tasks if lo<=x['comm']<hi])
    for target in (0,1):
        plans[target].sort(key=lambda x:x['comm']-BASE_COMM[target])
    return plans


class UnifiedK1Adapter:
    def __init__(self, *, rank, group, dispatcher, bank_chains, main_experts_by_rank, layers, pool, layout, state_bytes, flat, method, matmul, fixed_lease_hook=None, run_context=None, control_group=None):
        self.rank=rank; self.group=group; self.dispatcher=dispatcher; self.bank_chains=bank_chains; self.control_group=control_group
        self.main_experts_by_rank=main_experts_by_rank; self.layers=tuple(layers); self.pool=pool
        self.layout=layout; self.state_bytes=state_bytes; self.flat=flat; self.method=method; self.matmul=matmul
        self.lease=None; self.replica_chain=None; self.dispatch_calls=0; self.dispatch_attempts=0
        self.fixed_lease_hook=fixed_lease_hook
        self.run_context=dict(run_context or {})
        self.memory_events=[{'stage':'adapter_init','rank':rank,'memory':memory_snapshot()}]
        self.install_events=[]

    def _sync(self): torch.npu.synchronize()
    def _global_max(self, value):
        t=torch.tensor([int(value)],dtype=torch.int32,device='cpu'); dist.all_reduce(t,op=dist.ReduceOp.MAX,group=self.control_group); return int(t.item())

    def install(self, layer, out_dir:Path, tag:str):
        started=time.perf_counter_ns(); info={'rank':self.rank,'layer':layer,'tag':tag,**self.run_context}
        if self.rank==0:
            state=external_state_from_bank(self.bank_chains[(layer,43)].layer)
            for name,off,nbytes,_,_ in self.layout:
                self.flat[off:off+nbytes].copy_(state[name].view(torch.uint8).reshape(-1))
            self._sync(); dist.send(self.flat,dst=1,group=self.group); info['source_hash']=sha(self.flat)
        else:
            if self.lease is not None: self.pool.release(self.lease)
            self.lease=self.pool.acquire(request_id=tag,layer_id=layer,expert_id=43)
            view=self.pool.slot_buffer(self.lease,allow_reserved=True); dist.recv(view,src=0,group=self.group); self.pool.mark_ready(self.lease)
            with self.pool.use(self.lease) as active:
                views=slot_state_views(active,self.layout); fmt=[]
                for part in PARTS:
                    t=views[part]
                    try: fs=torch_npu.get_npu_format(t)
                    except Exception as ex: fs=f'ERROR:{type(ex).__name__}:{ex}'
                    try: fc=int(torch.ops.npu.get_npu_format(t))
                    except Exception as ex: fc=f'ERROR:{type(ex).__name__}:{ex}'
                    st=t.untyped_storage()
                    fmt.append({'name':part,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'format_string':fs,'format_code':fc,'storage_ptr':int(st.data_ptr()),'storage_bytes':int(st.nbytes()),'data_ptr':int(t.data_ptr()),'aliases_slot_storage':int(st.data_ptr())==int(active.untyped_storage().data_ptr())})
                ext=type('ReplicaLayer',(),{})()
                for part in PARTS: setattr(ext,LAYER_ATTRS[part],views[part])
                self.replica_chain=RuntimeChain(self.method,self.matmul,ext,layer)
                info.update({'pool_id':self.pool._pool_id,'generation':self.lease.generation,'ready':True,'owner':self.lease.owner_rank,'local_slot':self.lease.local_slot,'layer':self.lease.layer_id,'expert':self.lease.expert_id,'target_hash':sha(active),'slot_storage_ptr':int(active.untyped_storage().data_ptr()),'slot_storage_bytes':int(active.untyped_storage().nbytes()),'slot_formats':fmt})
        dist.barrier(group=self.control_group)
        (out_dir/f'{tag}_install_rank{self.rank}.json').write_text(json.dumps(info,indent=2))
        dist.barrier(group=self.control_group)
        peer_path=out_dir/f'{tag}_install_rank{1-self.rank}.json'
        deadline=time.monotonic()+20.0
        while not peer_path.exists():
            if time.monotonic()>=deadline: raise TimeoutError(f'peer install record not visible: {tag}')
            time.sleep(0.01)
        peer=json.loads(peer_path.read_text())
        info['hash_equal']=(info.get('source_hash')==peer.get('target_hash') if self.rank==0 else peer.get('source_hash')==info.get('target_hash'))
        info['install_ms']=(time.perf_counter_ns()-started)/1e6
        (out_dir/f'{tag}_install_rank{self.rank}.json').write_text(json.dumps(info,indent=2))
        self.install_events.append(info); self.memory_events.append({'stage':f'installed_l{layer}','rank':self.rank,'memory':memory_snapshot()})
        return info

    def slot_info(self):
        if self.rank!=1 or self.lease is None: return None
        records=self.pool.owner_records().get('1',[])
        record=next((x for x in records if x['slot_id']==self.lease.local_slot),None)
        return {'pool_id':self.pool._pool_id,'ready':bool(record and record['state'] in ('ready','in_use')),'actual_state':record['state'] if record else None,'layer':self.lease.layer_id,'expert':self.lease.expert_id,'generation':self.lease.generation,'owner':self.lease.owner_rank,'local_slot':self.lease.local_slot,'record':record}

    def owner_records(self):
        if self.rank!=1 or self.pool is None: return []
        return self.pool.owner_records().get('1',[])

    def _owner_slot_errors(self, layer, requests):
        errs=[]
        if self.rank!=1: return errs
        try: actual=self.pool.stats()['local'][1]
        except Exception as ex: return [f'owner_stats_error:{type(ex).__name__}:{ex}']
        records=self.owner_records()
        for i,r in enumerate(requests):
            if r is None: errs.append(f'rank{i}_missing_slot_request'); continue
            if r.get('ready') is not True: errs.append(f'rank{i}_request_not_ready')
            if r.get('pool_id')!=self.pool._pool_id: errs.append(f'rank{i}_request_pool_mismatch')
            if r.get('owner')!=1: errs.append(f'rank{i}_request_owner_mismatch:{r.get("owner")}')
            record=next((x for x in records if x['slot_id']==r.get('local_slot')),None)
            if record is None:
                errs.append(f'rank{i}_actual_slot_missing')
                continue
            if record['state'] not in ('ready','in_use'):
                errs.append(f'rank{i}_actual_slot_not_ready:{record["state"]}')
            checks=[('layer',r.get('layer'),record.get('layer')),('expert',r.get('expert'),record.get('expert')),('generation',r.get('generation'),record.get('generation')),('pool_id',r.get('pool_id'),record.get('global_pool_id')),('owner',r.get('owner'),record.get('owner_rank'))]
            for label,want,got in checks:
                if want!=got: errs.append(f'rank{i}_{label}_mismatch:{want}!={got}')
            if r.get('pool_id')!=self.pool._pool_id: errs.append(f'rank{i}_request_pool_mismatch')
            if r.get('layer')!=layer: errs.append(f'rank{i}_request_case_layer_mismatch:{r.get("layer")}!={layer}')
            if self.lease is None:
                errs.append('owner_has_no_live_lease')
            elif record['generation']!=self.lease.generation or record['layer']!=self.lease.layer_id or record['expert']!=self.lease.expert_id:
                errs.append(f'rank{i}_record_vs_lease_mismatch')
        return errs

    def execute(self, *, case_id, layer, hidden, manifests, hidden_by_rank, mode='none', slot_requests=None, inject_invalid_comm=False, fixed_lease_hook=None):
        before=memory_snapshot(); reasons=[]; hook_record={'called':False,'errors':[]}; local_tokens=int(manifests[self.rank]['tokens']); total=sum(int(m['tokens']) for m in manifests)
        if local_tokens!=int(hidden.shape[0]): reasons.append('hidden_token_count_mismatch')
        plan_manifests=copy.deepcopy(manifests)
        for m in plan_manifests:
            for row in m['logical_rows']:
                if len(row)!=8 or len(set(row))!=8 or any(not (0<=int(e)<256) for e in row): reasons.append(f'invalid_top8_rank{m["rank"]}'); break
        # Explicit caller plan after candidate mapping, with an intentional injection point.
        if inject_invalid_comm:
            if plan_manifests[self.rank]['comm_rows']:
                plan_manifests[self.rank]['comm_rows'][0][0]=EMPTY_COMM
            else: reasons.append('cannot_inject_empty_without_token')
        # Padding is generated only here; never infer it from hidden/weight/comm.
        plans=plan_tasks(plan_manifests); local_plan=plans[self.rank]
        for plan in plans:
            for task in plan:
                if not task['padding'] and task['comm']==EMPTY_COMM:
                    reasons.append('empty_comm_nonpadding_task')
                if not (0<=task['comm']<258):
                    reasons.append(f'comm_out_of_range:{task["comm"]}')
        replica_required=any(t['comm']==REPLICA_COMM for p in plans for t in p)
        replica_flag=torch.tensor([int(replica_required)],dtype=torch.int32,device='cpu'); dist.all_reduce(replica_flag,op=dist.ReduceOp.MAX,group=self.control_group); replica_required=int(replica_flag.item())==1
        requests=slot_requests or [None,None]
        owner_snapshot=None
        if replica_required and self.rank==1:
            st=self.pool.stats()['local'][1]
            owner_snapshot={'pool_id':self.pool._pool_id,'actual_ready':bool(st.ready==1 or st.in_use==1),'actual_reserved':st.reserved,'actual_ready_count':st.ready,'actual_in_use':st.in_use,'records':self.owner_records(),'lease_layer':self.lease.layer_id if self.lease else None,'lease_expert':self.lease.expert_id if self.lease else None,'lease_generation':self.lease.generation if self.lease else None,'lease_owner':self.lease.owner_rank if self.lease else None,'lease_local_slot':self.lease.local_slot if self.lease else None,'requests':requests}
        slot_errors=self._owner_slot_errors(layer,requests) if replica_required else []
        reasons.extend(slot_errors)
        bad=self._global_max(bool(reasons))
        if bad:
            return {'case_id':case_id,'rank':self.rank,'status':'REJECTED','reasons':sorted(set(reasons)),'dispatch_called':False,'dispatch_delta':0,'global_tokens':total,'replica_required':replica_required,'owner_slot_snapshot':owner_snapshot,'lease_overwrite_guard':hook_record}
        if total==0:
            return {'case_id':case_id,'rank':self.rank,'status':'PASS_EMPTY','reasons':[],'dispatch_called':False,'dispatch_delta':0,'global_tokens':0,'output_shape':[0,hidden.shape[1]],'padding_tokens_sent':0,'padding_tasks_sent':0,'padding_tasks_received':0,'replica_tasks':0,'empty_legal_tasks':0,'input_unchanged':True,'actual_expert_inputs_equal':True}
        local_valid=local_tokens
        if local_valid==0:
            input_hidden=torch.zeros((1,hidden.shape[1]),dtype=torch.bfloat16,device='npu')
        else:
            input_hidden=hidden
        weight_rows=[None]*len(local_plan); # weights are looked up from manifests below
        # Need sender-local weights for local input only; alltoall carries weights.
        local_weights=manifests[self.rank]['weights']
        comm_rows=manifests[self.rank]['comm_rows']
        weight_t=torch.tensor(local_weights if local_valid else [[0.0]*8],dtype=torch.float32,device='npu')
        comm_t=torch.tensor(comm_rows if local_valid else [[EMPTY_COMM]*8],dtype=torch.int32,device='npu')
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput
        owner_ctx=None
        if replica_required:
            if self.rank == 1 and (self.lease is None or self.replica_chain is None):
                reasons.append('replica_required_but_owner_not_executable')
            elif self.rank == 1:
                owner_ctx=self.pool.use(self.lease); owner_ctx.__enter__()
                hook=fixed_lease_hook or self.fixed_lease_hook
                hook_errors = hook() if hook else []
                hook_record={'called':bool(hook is not None),'errors':hook_errors,'actual_records_during_use':self.owner_records()}
                reasons.extend(hook_errors)
        # Hook failures are also a collective decision.  Every rank must enter
        # this control collective even when only the owner ran the hook.
        post_install_bad=self._global_max(bool(reasons)) if replica_required else False
        if post_install_bad or reasons:
            if owner_ctx is not None:
                owner_ctx.__exit__(None,None,None); owner_ctx=None
            return {'case_id':case_id,'rank':self.rank,'status':'REJECTED','reasons':sorted(set(reasons)),'dispatch_called':False,'dispatch_delta':0,'global_tokens':total,'replica_required':replica_required,'owner_slot_snapshot':owner_snapshot,'lease_overwrite_guard':hook_record}
        self.dispatch_attempts += 1
        calls_before=self.dispatch_calls
        try:
            dout=self.dispatcher.dispatch(input_hidden,StandardTopKOutput(weight_t,comm_t,None)); self.dispatch_calls += 1
            recv=dout.hidden_states[0] if isinstance(dout.hidden_states,tuple) else dout.hidden_states
            counts=[int(x) for x in dout.num_recv_tokens_per_expert]
            if sum(counts)!=len(local_plan): reasons.append(f'plan_count_mismatch:{sum(counts)}!={len(local_plan)}')
            # Verify actual rows are unchanged source token rows, row-by-row.
            input_mismatches=[]; input_rows=[]; padding_tasks_received=0; pos=0
            for task in local_plan:
                if task['padding']:
                    expected=torch.zeros_like(recv[0]); padding_tasks_received+=1
                else:
                    expected=hidden_by_rank[task['sender']][task['token']].to(recv.device)
                equal=bool(torch.equal(recv[pos],expected))
                input_rows.append({'row':pos,'sender':task['sender'],'token':task['token'],'slot':task['slot'],'comm':task['comm'],'padding':task['padding'],'equal':equal})
                if not equal:
                    input_mismatches.append({'row':pos,'sender':task['sender'],'token':task['token'],'padding':task['padding']})
                pos+=1
            if input_mismatches: reasons.append(f'actual_expert_input_mismatch:{len(input_mismatches)}')
            # Build expert output in native local-expert row order.
            expert_out=torch.zeros_like(recv); pos=0; replica_tasks=0; padding_skipped=0
            for li,count in enumerate(counts):
                if count<=0: continue
                rows=local_plan[pos:pos+count]; comm=BASE_COMM[self.rank]+li
                if any(t['comm']!=comm for t in rows): reasons.append(f'row_plan_comm_mismatch:{li}')
                if comm==EMPTY_COMM:
                    if not all(t['padding'] for t in rows): reasons.append('empty_comm_nonpadding_row')
                    padding_skipped+=count
                elif comm==REPLICA_COMM:
                    replica_tasks+=count
                    if self.rank!=1 or self.replica_chain is None: reasons.append('replica_execute_wrong_rank')
                    else: expert_out[pos:pos+count].copy_(self.replica_chain(recv[pos:pos+count]))
                else:
                    logical=comm_to_logical(comm); key=(layer,logical)
                    if key not in self.bank_chains: reasons.append(f'nonlocal_logical:{rank if False else self.rank}:{logical}')
                    else: expert_out[pos:pos+count].copy_(self.bank_chains[key](recv[pos:pos+count]))
                pos+=count
            bad_after=self._global_max(bool(reasons))
            if bad_after:
                return {'case_id':case_id,'rank':self.rank,'status':'FAIL_AFTER_DISPATCH','reasons':sorted(set(reasons)),'dispatch_called':True,'dispatch_delta':self.dispatch_calls-calls_before,'global_tokens':total,'replica_required':replica_required,'owner_slot_snapshot':owner_snapshot,'lease_overwrite_guard':hook_record}
            combined=self.dispatcher.combine(DeepEPNormalCombineInput(expert_out,None,None))
            out=combined[:local_valid]
            # The lease itself remains fixed after this call.  It is not returned
            # to the pool until install/release performs the pool's mandatory
            # device synchronization, so an in-flight combine cannot be overwritten.
            actual=[{'sender':t['sender'],'token':t['token'],'slot':t['slot'],'logical':t['logical'],'comm':t['comm'],'padding':t['padding'],'layer':layer,'task_id':f'{t["sender"]}:{t["token"]}:{t["slot"]}:{t["logical"]}:{t["comm"]}'} for t in local_plan if not t['padding']]
            return {'case_id':case_id,'rank':self.rank,'status':'PASS','reasons':[],'dispatch_called':True,'dispatch_delta':self.dispatch_calls-calls_before,'global_tokens':total,'replica_required':replica_required,'output_shape':list(out.shape),'padding_tokens_sent':1 if local_valid==0 else 0,'padding_tasks_sent':8 if local_valid==0 else 0,'padding_tasks_received':padding_tasks_received,'padding_expert_calls':0,'padding_rows_skipped':padding_skipped,'replica_tasks':replica_tasks,'empty_legal_tasks':0,'actual_tasks':actual,'input_unchanged':None,'actual_expert_inputs_equal':not input_mismatches,'actual_input_rows':input_rows,'actual_input_mismatches':input_mismatches,'owner_slot_snapshot':owner_snapshot,'owner_records_after':self.owner_records() if self.rank==1 else [],'lease_overwrite_guard':hook_record,'dispatch_layout_handle':{k:(v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v) for k,v in (getattr(self.dispatcher._normal_dispatcher,'handle',{}) or {}).items() if k in ('reversed_local_mapping','reversed_global_mapping','input_splits','output_splits')},'memory':memory_snapshot()},out
        finally:
            if owner_ctx: owner_ctx.__exit__(None,None,None)

    def release_slot(self):
        if self.rank==1 and self.lease is not None:
            self.pool.release(self.lease); self.lease=None; self.replica_chain=None

    def format_snapshot(self):
        rows=[]
        for (l,e),chain in self.bank_chains.items():
            for attr in LAYER_ATTRS.values():
                t=getattr(chain.layer,attr,None)
                if t is None: continue
                try: fs=torch_npu.get_npu_format(t)
                except Exception as ex: fs=f'ERROR:{type(ex).__name__}:{ex}'
                try: fc=int(torch.ops.npu.get_npu_format(t))
                except Exception as ex: fc=f'ERROR:{type(ex).__name__}:{ex}'
                st=t.untyped_storage(); rows.append({'layer':l,'expert':e,'attr':attr,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'format_string':fs,'format_code':fc,'data_ptr':int(t.data_ptr()),'storage_ptr':int(st.data_ptr()),'storage_bytes':int(st.nbytes()),'offset':int(t.storage_offset())})
        return rows
