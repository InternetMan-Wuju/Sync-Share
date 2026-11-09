import json,csv,gzip,collections,statistics,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
records=[]
for p in sorted((ROOT/'local').glob('rank*.jsonl')):
    records.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
with gzip.open(ROOT/'artifacts/rank_records.jsonl.gz','wt') as out:
    for r in records:out.write(json.dumps(r)+'\n')
blocks=[r for r in records if r['kind']=='decision_block']
with (ROOT/'TIMINGS.csv').open('w') as out:
    cols=['request','split','case','scale','global_rows','scope','mode','block','rank','calls','block_ms','min_rank_block_ms','max_rank_block_ms']
    w=csv.DictWriter(out,cols,extrasaction='ignore');w.writeheader();w.writerows(blocks)
groups=collections.defaultdict(list)
for r in blocks:groups[(r['request'],r['scale'],r['mode'],r['block'])].append(r)
points=collections.defaultdict(dict)
for (req,s,m,b),rows in groups.items():
    assert len(rows)==16 and len({r['rank'] for r in rows})==16
    assert len({r['calls'] for r in rows})==1 and min(r['block_ms'] for r in rows)>=100
    points[(req,s,m)][b]=max(r['block_ms'] for r in rows)/rows[0]['calls']
summary={'measurements':[],'cases':[],'component_scope':'amortized repeated blocks; do not sum component maxima','timing':{'rank_blocks':len(blocks),'group_blocks':len(groups),'min_block_ms':min(r['block_ms'] for r in blocks),'max_block_ms':max(r['block_ms'] for r in blocks),'warmup_calls_per_variant':20,'formal_blocks':5}}
for (req,s,m),values in sorted(points.items()):
    assert len(values)==5
    v=list(values.values());row={'request':req,'scale':s,'mode':m,'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),'group_block_means_ms':values}
    summary['measurements'].append(row)
inputs=[r for r in records if r['kind']=='decision_input' and r['rank']==0]
(ROOT/'INPUTS_PLANS.json').write_text(json.dumps(inputs,indent=2))
for inp in inputs:
    req=inp['request'];s=inp['scale'];base=points[(req,s,'off')]
    entry={k:inp[k] for k in ('request','split','case','scale','global_rows','admitted','count_plan','policy_plan','policy_detail')}
    entry['off_ms']=statistics.median(base.values());entry['modes']={}
    for mode in ('count_cold','policy_cold','prepared_count'):
        if (req,s,mode) not in points:continue
        vals=points[(req,s,mode)];diff=[base[b]-vals[b] for b in base]
        entry['modes'][mode]={'median_ms':statistics.median(vals.values()),'paired_saving_ms':statistics.median(diff),'paired_saving_percent':statistics.median([(base[b]-vals[b])/base[b]*100 for b in base]),'min_paired_saving_ms':min(diff),'max_paired_saving_ms':max(diff),'positive_pairs':sum(x>0 for x in diff)}
    summary['cases'].append(entry)
validation=[r for r in records if r['kind']=='decision_validation']
v={'rank_checks':len(validation),'case_actions':len({(r['request'],r['scale'],r['label']) for r in validation}),
   'max_abs':max(r['numeric']['max_abs'] for r in validation),'max_nrmse':max(r['numeric']['nrmse'] for r in validation),
   'all_state_bytes_equal':all(r['copy_timing']['state_bytes_equal'] for r in validation),
   'scope':'independent-clone native same-hidden/topk reference, not independent FP-model certification',
   'cpu_test_log':'artifacts/CPU_TESTS.txt'}
assert v['all_state_bytes_equal'] and v['max_nrmse']<=.01
(ROOT/'VALIDATION.json').write_text(json.dumps(v,indent=2))
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(v))
for c in summary['cases']:
    print(c['request'],c['global_rows'],round(c['off_ms'],3),{m:(round(v['median_ms'],3),round(v['paired_saving_ms'],3)) for m,v in c['modes'].items()},'plan',c['policy_plan'])
