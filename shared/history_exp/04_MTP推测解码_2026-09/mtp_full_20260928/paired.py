#!/usr/bin/env python3
"""Compare shared request indices, separating complete pairs from window totals."""
import csv
import json
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1]);records=[];summaries={}
for scene in ('decode','mixed','prefix'):
    bymode={}
    for mode in ('off','on'):
        path=root/scene/mode/'completed_or_partial_results.json'
        bymode[mode]={r['case_index']:r for r in json.loads(path.read_text())} if path.exists() else {}
    shared=sorted(bymode['off'].keys()&bymode['on'].keys())
    complete=[];comparisons=[]
    for index in shared:
        a=bymode['off'][index];b=bymode['on'][index]
        ids=[]
        for mode in ('off','on'):
            p=root/scene/mode/f'request_{index:04d}/measurement/output_ids_final_or_partial.json'
            ids.append(json.loads(p.read_text()))
        n=min(map(len,ids));first_mismatch=next((i for i in range(n) if ids[0][i]!=ids[1][i]),None)
        finished=all(r['actual_output']==r['requested_output'] and r['done_events']==1 and r['error'] is None for r in (a,b))
        row={'scenario':scene,'index':index,'same_input_sha256':a['input_sha256']==b['input_sha256'],
            'both_complete':finished,'compared_output_prefix_tokens':n,'first_output_id_mismatch_index':first_mismatch,
            'off_output':a['actual_output'],'on_output':b['actual_output'],
            'off_ttft_s':a['ttft_s'],'on_ttft_s':b['ttft_s'],
            'off_tpot_ms':a['observed_tpot_ms'],'on_tpot_ms':b['observed_tpot_ms'],
            'off_e2e_s':a['e2e_s'],'on_e2e_s':b['e2e_s'],'on_accept_length':b['meta_info_last'].get('spec_accept_length'),
            'complete_pair_e2e_speedup':a['e2e_s']/b['e2e_s'] if finished and b['e2e_s'] else None}
        records.append(row);comparisons.append(row)
        if finished:complete.append(row)
    summaries[scene]={'shared_input_pairs':len(shared),'complete_pairs':len(complete),
        'same_input_all_shared':all(x['same_input_sha256'] for x in comparisons),
        'output_prefix_mismatch_pairs':sum(x['first_output_id_mismatch_index'] is not None for x in comparisons),
        'complete_pair_mean_e2e_speedup':statistics.mean(x['complete_pair_e2e_speedup'] for x in complete) if complete else None,
        'complete_pair_off_mean_ttft_s':statistics.mean(x['off_ttft_s'] for x in complete) if complete else None,
        'complete_pair_on_mean_ttft_s':statistics.mean(x['on_ttft_s'] for x in complete) if complete else None,
        'complete_pair_off_mean_tpot_ms':statistics.mean(x['off_tpot_ms'] for x in complete) if complete else None,
        'complete_pair_on_mean_tpot_ms':statistics.mean(x['on_tpot_ms'] for x in complete) if complete else None,
        'note':'Shared indices only. Timing means use complete pairs; raw output prefix equality is descriptive, not a proof of distributional equivalence.'}
out=root/'analysis';out.mkdir(exist_ok=True)
(out/'paired_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
if records:
    with (out/'paired_requests.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=records[0].keys());wr.writeheader();wr.writerows(records)
print(json.dumps(summaries,ensure_ascii=False))
