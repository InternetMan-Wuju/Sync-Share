#!/usr/bin/env python3
"""Small read-only status report for the active autonomous experiment."""
import datetime
import json
import re
import time
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')


def main():
    print(datetime.datetime.now().isoformat())
    for name in ('overnight_status.json', 'chunk_repeat_status.json', 'final_mtp_isolation_status.json', 'finish_autonomous_status.json'):
        path = ROOT / name
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        print(name, {key: state.get(key) for key in ('phase', 'container', 'cell', 'error', 'child_pid', 'completed')})
        container = state.get('container')
        logfile = ROOT / 'logs' / str(container) / 'server.log'
        if state['phase'] in ('waiting_service', 'waiting_health') and logfile.exists():
            with logfile.open('rb') as stream:
                stream.seek(max(0, logfile.stat().st_size - 262144))
                tail = stream.read().decode(errors='replace')
            progress = re.findall(r'Multi-thread loading shards:[^\r\n]*', tail)
            print('loading', progress[-1] if progress else tail[-150:])
        out = None
        if name == 'overnight_status.json' and not state.get('completed'):
            command = state.get('command') or []
            if '--out' in command:
                out = Path(command[command.index('--out') + 1])
        elif name == 'chunk_repeat_status.json' and state['phase'] == 'running_cell' and state.get('cell') and not state.get('completed'):
            out = ROOT.parent / 'Qchunksize/results/autonomous_20261012' / state['cell']
        elif name == 'final_mtp_isolation_status.json' and state.get('result') and not state.get('completed'):
            out = Path(state['result']).parent if state['result'].endswith('status.json') else Path(state['result'])
        if out is not None:
            path = out / 'status.json'
            if path.exists():
                data = json.loads(path.read_text())
                print('client', {'output':out.name, **{key: data.get(key) for key in ('stage', 'current_group', 'done', 'completed', 'error')}})
                records = data.get('request_progress', data.get('requests_progress')) or {}
                print('progress', {key: {'state': row.get('state'), 'tokens': row.get('output_tokens'),
                                        'target': row.get('target_output_tokens'),
                                        'last_event_age_s': round(time.time()-row['last_event_unix'],1) if row.get('last_event_unix') else None}
                                   for key,row in records.items()})
            path = out / 'target_prefill_audit/summary.json'
            if path.exists():
                data = json.loads(path.read_text())
                print('audit', {'checks': len(data['prefill_checks']), 'rechecks': len(data.get('mismatch_prefill_rechecks',[])),
                                'windows': len(data['windows']), 'completed': data.get('completed')})
    files = list((ROOT.parent / 'Qchunksize/results/autonomous_20261012').glob('*/result.json'))
    print('chunk cells', len(files))
    for path in files[-3:]:
        data = json.loads(path.read_text())
        print(path.parent.name, {key: data['aggregate'].get(key) for key in ('complete', 'total_output_tok_s', 'cache_hit_ratio')})


if __name__ == '__main__':
    main()
