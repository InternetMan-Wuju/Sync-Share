#!/usr/bin/env python3
"""Validate saved results and reconcile recovery with durable runner status."""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from bench_official_mtp_full import validate, write_json
from run_official_mtp_full import report


def inspect_results(out):
    manifest = json.loads((out/'manifest.json').read_text())
    states = {}
    for scenario in manifest['workloads']:
        name = scenario['name']
        file = out/name/'result.json'
        if not file.is_file():
            states[name] = {'valid':False,'state':'pending'}
            continue
        result = json.loads(file.read_text())
        rows = result['requests']
        checks = [validate(r,scenario['input'],scenario['output'],bool(scenario.get('shared_prefix_tokens'))) for r in rows]
        failures = [c for c in checks if c['failures']]
        valid = len(rows)==scenario['requests'] and not failures
        states[name] = {'state':'completed' if valid else 'validation_failed','valid':valid,
                        'requests':len(rows),'expected_requests':scenario['requests'],
                        'output_tokens':sum(r['completion_tokens'] for r in rows),'checks':checks}
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    out = Path(args.out)
    while True:
        states = inspect_results(out)
        write_json(out/'validated_status.json',{'updated_unix':time.time(),'scenarios':states,
                                              'completed_scenarios':sum(s['valid'] for s in states.values())})
        status_file = out/'experiment_status.json'
        state = json.loads(status_file.read_text())
        terminal = state['state'] in ('completed','completed_with_failures','failed')
        if terminal and state.get('container_running') is False:
            break
        time.sleep(15)
    write_json(out/'runner_terminal_status.json',state)
    errors = []
    for scenario, info in states.items():
        if not info['valid']:
            errors.append({'scenario':scenario,'error':'result missing or validation failed'})
            continue
        result = json.loads((out/scenario/'result.json').read_text())
        for record in result['requests']:
            recovered_ids = []
            with gzip.open(record['events_file'],'rt') as stream:
                for line in stream:
                    recovered_ids.extend(json.loads(line).get('delta_ids',[]))
            payload = json.loads(Path(record['payload_file']).read_text())
            checksum = hashlib.sha256(json.dumps(payload['input_ids'],separators=(',',':')).encode()).hexdigest()
            if recovered_ids != record['output_ids'] or checksum != record['input_sha256']:
                errors.append({'scenario':scenario,'index':record['index'],'error':'journal/output or payload/SHA mismatch'})
    verified = all(s['valid'] for s in states.values()) and not errors
    write_json(out/'final_validation.json',{'completed':verified,'scenarios':states,'failures':errors,
                                          'journals_reconstructed_and_payload_sha_checked':True})
    state.update(completed=verified,state='completed' if verified else 'completed_with_failures',
                 scenario_status={k:v['state'] for k,v in states.items()},
                 updated_unix=time.time(),failed_scenarios=[k for k,v in states.items() if not v['valid']],
                 reconciled_from_saved_records=True,
                 recovery_note='w1 collector aggregation error was recovered on CPU; all raw meta retained, no requests resent')
    write_json(out/'experiment_status.json',state)
    report(out,state['scenario_status'])
    with (out/'summary.md').open('a') as summary:
        summary.write('\n最终数据核验：'+('通过' if verified else '未全部通过')+'。逐请求事件可重建完整输出，输入payload与SHA一致。\n')
        summary.write('w1原采集器汇总时遇到省略零桶的直方图；已仅在CPU补零后恢复统计，无重发。w1吞吐墙钟由最早请求开始至最晚请求结束恢复，未包含线程池启动及汇总开销；详见result.json的summary_recovery。\n')
        summary.write('runner原始终态保存在runner_terminal_status.json，核验后的终态在experiment_status.json。\n')
    print('final validation',verified,errors,flush=True)
    if not verified:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
