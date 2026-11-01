#!/usr/bin/env python3
"""Aggregate matched NEXTN preview results from ON/OFF directories."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run_dir', type=Path)
a = p.parse_args()
rows = []
for scenario in ('decode', 'mixed', 'long'):
    data = {}
    for mode in ('off', 'on'):
        f = a.run_dir/mode/(scenario+'_summary.json')
        if f.exists():
            x = json.loads(f.read_text())
            rate_denom = x.get('spec_num_proposed_drafts') or 0
            verify = x.get('spec_verify_ct') or 0
            x['weighted_draft_accept_rate'] = (x.get('spec_num_correct_drafts', 0)/rate_denom if rate_denom else None)
            x['weighted_accept_length'] = (sum(int(json.loads(line).get('output_tokens') or 0)
                          for line in (a.run_dir/mode/'requests.jsonl').read_text().splitlines()
                          if json.loads(line).get('scenario', '').startswith(scenario + '_') and json.loads(line).get('role') == 'measurement') / verify if verify else None)
            data[mode] = x
    off, on = data.get('off', {}), data.get('on', {})
    def mean(x, key):
        return (x.get(key) or {}).get('mean')
    def ratio(onval, offval):
        return (onval / offval) if offval and onval is not None else None
    rows.append({'scenario': scenario, 'input_tokens': on.get('input_tokens', off.get('input_tokens')),
       'output_tokens': on.get('output_tokens', off.get('output_tokens')),
       'off': off, 'on': on,
       'throughput_on_over_off': ratio(mean(on, 'output_tok_s'), mean(off, 'output_tok_s')),
       'ttft_on_over_off': ratio(mean(on, 'ttft_s'), mean(off, 'ttft_s')),
       'tpot_on_over_off': ratio(mean(on, 'tpot_ms'), mean(off, 'tpot_ms'))})
(a.run_dir/'comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
for row in rows:
    print(row['scenario'], 'speedup', row['throughput_on_over_off'],
          'TTFT_ratio', row['ttft_on_over_off'], 'TPOT_ratio', row['tpot_on_over_off'],
          'accept_length', row['on'].get('weighted_accept_length'),
          'draft_accept_rate', row['on'].get('weighted_draft_accept_rate'))
