"""Derive report tables from retained real measurements, never fill missing values."""
import collections,csv,json,statistics,gzip
from pathlib import Path
P=Path(__file__).resolve().parents[1]
clients=[]
for f in sorted((P/'artifacts').glob('*.json')):
    j=json.loads(f.read_text())
    if 'client_ttft_ms' in j:clients.append(j)
rows=[]
for f in sorted((P/'local').glob('rank*.jsonl')):
    rows.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
# Preserve raw small evidence, not hidden or weights.
(P/'artifacts/rank_records.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n').encode(),mtime=0))
plans={}
for r in rows:
    if r.get('rank')==0 and r.get('counts') is not None:
        plans[r['request']]={k:r[k] for k in ('counts','by_source','plan','q_by_source','old_slots','new_slots') if k in r}
entries=[]
for name,plan_record in plans.items():
    fields_text=[]
    for key,value in plan_record.items():
        if key in ('by_source','old_slots','new_slots'):
            body='[\n      '+',\n      '.join(json.dumps(row,separators=(',',':')) for row in value)+'\n    ]'
        else:body=json.dumps(value,separators=(',',':'))
        fields_text.append('    '+json.dumps(key)+': '+body)
    entries.append('  '+json.dumps(name)+': {\n'+',\n'.join(fields_text)+'\n  }')
(P/'INPUTS_PLANS.json').write_text('{\n'+',\n'.join(entries)+'\n}\n')
fields=['source','request','case_id','mode','rank','block','calls','block_ms','ms_per_call','metric','value_ms','scope']
with (P/'TIMINGS.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
    for c in clients:
        for key in ('client_ttft_ms','client_e2e_ms','client_decode_ms','client_tpot_ms'):
            w.writerow({'source':'whole_model_client','request':c['request'],'case_id':c['case_id'],'mode':c['mode'],'calls':1,'metric':key,'value_ms':c.get(key),'scope':'8192 input ->32 generated, TP16 EP16, eager; diagnostics excluded from E2E summary'})
    for r in rows:
        if r['kind'] in ('timing_block','calibration','short_block'):
            w.writerow({'source':r['kind'],'request':r['request'],'mode':r['mode'],'rank':r['rank'],'block':r.get('block'),'calls':r['calls'],'block_ms':r['block_ms'],'ms_per_call':r['block_ms']/r['calls'],'scope':'actual captured layer49 input; all16 ranks concurrent; complete native dispatch/runner/combine'})
        for k,v in r.get('timing',{}).items():
            if k.endswith('_ms'):
                w.writerow({'source':'diagnostic','request':r['request'],'mode':r['mode'],'rank':r['rank'],'calls':1,'metric':k,'value_ms':v,'scope':'diagnostic with events; verify=True includes expensive correctness checks'})
def stats(a):return {'n':len(a),'median':statistics.median(a),'min':min(a),'max':max(a)} if a else None
out={'e2e':{},'paired_ttft_saving':{},'layer_replay':{},'profile_rank_max':{},'correctness':[]}
for c in range(3):
    out['e2e'][c]={}
    for mode in ('off','shadow','active_cold','active_warm'):
        selected=[r for r in clients if r['case_id']==c and r['mode']==mode and r['request'].startswith('e2e_')]
        out['e2e'][c][mode]={metric:stats([r[metric] for r in selected]) for metric in ('client_ttft_ms','client_e2e_ms','client_tpot_ms')}
    out['paired_ttft_saving'][c]={}
    byname={r['request']:r for r in clients}
    for mode in ('shadow','active_cold','active_warm'):
        values=[]
        for block in range(5):
            a=byname.get(f'e2e_b{block}_c{c}_off');b=byname.get(f'e2e_b{block}_c{c}_{mode}')
            if a and b:values.append(a['client_ttft_ms']-b['client_ttft_ms'])
        out['paired_ttft_saving'][c][mode]={'saving_ms':stats(values),'paired_values_ms':values}
    out['layer_replay'][c]={}
    for mode in ('off','prepared','warm','cold'):
        values=[]
        for block in range(5):
            selected=[r for r in rows if r['kind']=='timing_block' and r['request']==f'replay_c{c}' and r['mode']==mode and r['block']==block]
            if selected:
                assert len(selected)==16 and len(set(r['rank'] for r in selected))==16
                assert all(r['block_ms']>=100 for r in selected)
                assert len(set(r['calls'] for r in selected))==1
                values.append(max(r['ms_per_call'] for r in selected))
        out['layer_replay'][c][mode]={'max_rank_block_mean_ms':stats(values),'values_ms':values}
for mode in ('off','shadow','active_cold','active_warm'):
    metrics=collections.defaultdict(list)
    for b in range(5):
        selected=[r for r in rows if r.get('request')==f'profile_b{b}_c0_{mode}' and r['kind']=='call']
        if selected:
            assert len(selected)==16
            for key in selected[0]['timing']:
                if key.endswith('_ms'):metrics[key].append(max(r['timing'].get(key,0) for r in selected))
            phase_keys=('source_prepare_ms','transfer_ms','destination_install_ms','ready_wait_ms')
            if all(k in selected[0]['timing'] for k in phase_keys):
                # Sum sequential phases within each rank, THEN max ranks.
                # Summing separate per-phase rank maxima would double count waits.
                metrics['state_prepare_through_ready_ms'].append(max(sum(r['timing'][k] for k in phase_keys) for r in selected))
    out['profile_rank_max'][mode]={k:stats(v) for k,v in metrics.items()}
out['correctness']=[{k:r[k] for k in ('request','rank','mode','global_tasks','numeric') if k in r} for r in rows if 'numeric' in r]
out['copy_role_times']={}
for role in ('src','dst'):
    selected=[r for r in rows if r.get('request','').startswith('profile_b') and r.get('mode')=='active_cold' and r['rank']==r['plan'][role]]
    out['copy_role_times'][role]={k:stats([r['timing'][k] for r in selected]) for k in ('source_prepare_ms','transfer_ms','destination_install_ms','ready_wait_ms')}
out['paired_replay_savings']={}
for c,modes in out['layer_replay'].items():
    out['paired_replay_savings'][c]={mode:stats([a-b for a,b in zip(modes['off']['values_ms'],modes[mode]['values_ms'])]) for mode in ('prepared','warm','cold')}
out['output_comparisons']={}
for c in range(3):
    selected=[r for r in clients if r['case_id']==c and r['request'].startswith('e2e_')]
    out['output_comparisons'][c]={'requests':len(selected),'all_ids_present':all(len(r['output_ids'])==32 for r in selected),'distinct_outputs':len({tuple(r['output_ids']) for r in selected}),'off_distinct_outputs':len({tuple(r['output_ids']) for r in selected if r['mode']=='off'})}
(P/'SUMMARY.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
