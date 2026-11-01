#!/usr/bin/env python3
"""Continue the authorized sequence after the running base ChunkSize grid.

No overlapping NPU jobs, no automatic retries with changed configurations.
All intermediate state and failure evidence remain in the workspace.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
STATE = {'pid': os.getpid(), 'started_unix': time.time(), 'completed': False}


def status(phase, **extra):
    STATE.update(phase=phase, updated_unix=time.time(), **extra)
    path = ROOT / 'finish_autonomous_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2))
    temp.replace(path)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), phase, extra, flush=True)


def main():
    deadline = time.time() + 6 * 3600
    while True:
        prior = json.loads((ROOT / 'overnight_status.json').read_text())
        if prior.get('completed') and prior['phase'] == 'planned_grid_completed':
            break
        if prior['phase'] == 'needs_conservative_review':
            raise RuntimeError('base grid needs review: ' + str(prior.get('error')))
        os.kill(prior['runner_pid'], 0)
        if time.time() > deadline:
            raise RuntimeError('base grid wait exceeded six hours; preserve its state')
        status('waiting_base_grid', base_phase=prior['phase'], container=prior.get('container'))
        time.sleep(30)
    for script in ('repeat_chunk_candidates.py', 'final_mtp_isolation.py',
                   'summarize_progress.py', 'analyze_mtp_completed.py',
                   'combine_mtp_history.py', 'analyze_dp_prefixes.py',
                   'analyze_chunk_completed.py', 'validate_completed.py', 'plot_completed.py',
                   'publish_final_report.py'):
        status('running', script=script)
        with (ROOT / 'logs' / ('finish_' + script.removesuffix('.py') + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / script)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    status('completed', completed=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('needs_conservative_review', error=repr(error))
        raise
