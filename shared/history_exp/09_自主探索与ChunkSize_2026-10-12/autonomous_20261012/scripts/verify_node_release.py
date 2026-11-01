#!/usr/bin/env python3
"""Read-only resource verification after the autonomous experiment finishes."""
import datetime
import json
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=45).stdout


def main():
    for name in ('overnight_status.json', 'chunk_repeat_status.json',
                 'final_mtp_isolation_status.json', 'finish_autonomous_status.json'):
        assert json.loads((ROOT / name).read_text()).get('completed'), name
    assert socket.gethostname() == 'npu1-23', 'verify the authorized node only'
    ids = command(['sudo', '-n', 'docker', 'ps', '-aq',
                   '--filter', 'label=owner=chengyetai',
                   '--filter', 'label=experiment=autonomous_mtp_chunk']).split()
    containers = []
    if ids:
        for item in json.loads(command(['sudo', '-n', 'docker', 'inspect', *ids])):
            labels = item['Config'].get('Labels') or {}
            assert labels.get('owner') == 'chengyetai'
            assert labels.get('experiment') == 'autonomous_mtp_chunk'
            assert item['Name'].startswith('/cyt-auto-')
            containers.append({'name': item['Name'].lstrip('/'), 'id': item['Id'],
                'running': item['State']['Running'], 'pid': item['State']['Pid'],
                'status': item['State']['Status'], 'finished_at': item['State']['FinishedAt'],
                'exit_code': item['State']['ExitCode'], 'labels': {
                    key: labels[key] for key in ('owner', 'experiment')}})
    npu = command(['npu-smi', 'info'])
    (ROOT / 'logs/node23_release_npu_smi.txt').write_text(npu)
    running = [row['name'] for row in containers if row['running']]
    report = {'checked_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'node': socket.gethostname(), 'containers': containers,
        'owned_running_containers': running, 'owned_containers_all_stopped': not running,
        'npu_reports_no_running_processes_on_all_8_devices': npu.count('No running processes found') == 8,
        'npu_observation': str(ROOT / 'logs/node23_release_npu_smi.txt'),
        'verification_scope': 'Read-only inspection. No container removed and no other user process stopped.'}
    (ROOT / '节点释放核对.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'containers'}, ensure_ascii=False))
    assert containers and not running, 'owned service remains running; inspect ownership before any stop'


if __name__ == '__main__':
    main()
