import json,pathlib,hashlib,datetime,subprocess,gzip
P=pathlib.Path(__file__).resolve().parents[1]
tz=datetime.timezone(datetime.timedelta(hours=8))
started=datetime.datetime.fromisoformat('2026-11-09T06:13:33+08:00')
now=datetime.datetime.now(tz)
v=json.loads((P/'VALIDATION.json').read_text())
last=json.loads((P/'artifacts/validate_partial_4081.json').read_text())
base=subprocess.check_output(['git','rev-parse','d5a8396'],text=True,cwd=P).strip()
files=sorted([p for folder in ('plugin','scripts','tests') for p in (P/folder).glob('*.py')]+list((P/'scripts').glob('*.sh'))+[P/'layout.json',P/'artifacts/measured_experiment_v2.py'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rs=[json.loads(x) for x in gzip.decompress((P/'artifacts/rank_records.jsonl.gz').read_bytes()).decode().splitlines()]
blocks=[r for r in rs if r['kind']=='timing_block']
clients=[]
for p in (P/'artifacts').glob('*.json'):
    r=json.loads(p.read_text())
    if 'client_ttft_ms' in r:clients.append(r)
formal=[r for r in clients if r['request'].startswith('e2e_')]
assert len(blocks)==960 and len(formal)==60
assert all(r['meta']['prompt_tokens']==8192 and r['meta']['completion_tokens']==32 and r['meta']['cached_tokens']==0 for r in clients)
assert all(r['numeric']['max_abs']==0 for r in rs if r.get('request','').startswith('validate_cloned_') or r.get('request')=='validate_partial_4081')
assert all(r['implementation_revision']==2 for r in formal)
assert all(r['sha256']==sha(P/'artifacts/measured_experiment_v2.py') for r in rs if r['kind']=='implementation' and r['revision']==2)
assert all(r['sha256']==sha(P/'plugin/experiment.py') for r in rs if r['kind']=='implementation' and r['revision']==3)
evidence=['INPUTS.json','INPUTS_PLANS.json','TIMINGS.csv','SUMMARY.json','VALIDATION.json','ENVIRONMENT.json','RESULTS.md','artifacts/rank_records.jsonl.gz']
manifest={
 'started_at':started.isoformat(),
 'sampling_finished_at':datetime.datetime.fromtimestamp(last['meta']['request_finished_ts'],tz).isoformat(),
 'packaged_at':now.isoformat(),'elapsed_minutes_to_packaging':round((now-started).total_seconds()/60,2),
 'host':'npu1-23','devices':list(range(16)),'base_commit':base,
 'status':'COMPLETE_SINGLE_NODE_EXPERIMENT_WITH_LIMITATIONS',
 'scope':'TP16 EP16 DP1; full GLM model; target routed layer49;8192 input /32 output; no cross-node work',
 'input_identity':'AgentX length/hash-block driven deterministic synthetic token materialization, NOT original AgentX prompt',
 'source_files_modified':False,
 'modes':'off/shadow/test_only active_cold/active_warm; real native dispatch/runner/combine',
 'production_ready':False,'e2e_acceleration_demonstrated':False,
 'generation_reproducibility':'UNRESOLVED: off itself has5/5/4 different sequences across5 repetitions',
 'correctness':{'cpu_tests_passed':6,'whole_input_cases':3,'ranks_per_case':16,'max_abs':0.0,'max_nrmse':0.0,'partial_quota_npu_case':v['partial_quota_case'],'state_copy_bytes':37810176,'expected_global_rows':8192,'expected_global_valid_branches':65536},
 'timing':{'whole_requests_formal':60,'all_client_records':len(clients),'replay_concurrent_group_blocks':60,'rank_block_records':len(blocks),'warmup_calls_per_mode_per_input':20,'min_rank_block_ms':min(r['block_ms'] for r in blocks),'max_rank_block_ms':max(r['block_ms'] for r in blocks),'repetitions':sorted(set(r['calls'] for r in blocks)),'primary_scope':'steady repeated routed-expert subchain wall time, max concurrent rank block mean','profile_scope':'5 separate diagnostics per mode; event/rank maxima are not additive; not independently100ms component benchmarks','production_p95_measured':False},
 'memory':{'extra_slots':'1/rank at all75 MoE layers, not layer-local','estimated_extra_expert_bytes_per_rank':37810176*75,'measured_hbm_delta_vs_no_slots':None,'packed_state_buffer_bytes_per_rank':37810176,'kv_tokens':12288,'max_memory_allocated_bytes':max(r.get('hbm_peak_bytes',0) for r in rs)},
 'implementation_revisions':v['implementation_revisions'],
 'sampling_source_sha256':sha(P/'artifacts/measured_experiment_v2.py'),
 'final_code_sha256':{str(p.relative_to(P)):sha(p) for p in files},
 'evidence_sha256':{f:sha(P/f) for f in evidence},
 'cleanup':{'model_container_stopped':True,'all_npu_devices_released':True,'evidence':'artifacts/npu_released.txt'},
 'not_completed':['production cost-aware gate and graph validation','full-generation determinism/root cause','layer-local spare storage','representative original AgentX text workload','whole-service acceleration or SLO certification']
}
(P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(manifest['sampling_finished_at'],manifest['packaged_at'],manifest['elapsed_minutes_to_packaging'])
