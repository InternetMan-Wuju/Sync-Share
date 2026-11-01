#!/usr/bin/env python3
"""Recompute a summary from saved terminal records without sending requests."""
import argparse
import json
from pathlib import Path

from bench_official_mtp_full import aggregate, validate, write_json
from run_official_mtp_full import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenario', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    directory = out/args.scenario
    dataset = out/'datasets'/(args.scenario+'.json')
    ds = json.loads(dataset.read_text())
    if (directory/'completed_requests.json').is_file():
        saved = json.loads((directory/'completed_requests.json').read_text())
        records, wall = saved['requests'], saved['wall_s']
        timing_method = 'original measured scenario wall time'
    else:
        records = [json.loads((directory/f'request_{i:03d}.json').read_text()) for i in range(ds['n_requests'])]
        wall = max(r['ended_unix'] for r in records)-min(r['started_unix'] for r in records)
        timing_method = 'recovered request interval: latest ended_unix - earliest started_unix; excludes pool startup/final aggregation'
    checks = [validate(r,ds['input_tokens'],ds['target_output_tokens'],bool(ds.get('shared_prefix_tokens'))) for r in records]
    failures = [c for c in checks if c['failures']]
    result = {'dataset':str(dataset),'scenario':args.scenario,'concurrency':ds['concurrency'],
              'target_output_tokens':ds['target_output_tokens'],'requests_expected':ds['n_requests'],
              'aggregate':aggregate(records,wall),'requests':records,'validation':checks,'valid':not failures,
              'summary_recovery':{'reason':'server histogram omitted unobserved trailing zero bins; pad for arithmetic only, preserve raw meta',
                                  'timing_method':timing_method,'requests_resent':False}}
    write_json(directory/'result.json',result)
    state = json.loads((out/'experiment_status.json').read_text())
    report(out,state.get('scenario_status',{}))
    if (directory/'progress.json').is_file():
        progress = json.loads((directory/'progress.json').read_text())
        progress['state'] = 'completed' if not failures else 'validation_failed'
        write_json(directory/'progress.json',progress)
    print(json.dumps(result['aggregate'],indent=2))
    print('validation_failures',failures)
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
