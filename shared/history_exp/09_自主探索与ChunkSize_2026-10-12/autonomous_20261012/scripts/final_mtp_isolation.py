#!/usr/bin/env python3
"""After ChunkSize, isolate NPU multi-stream execution on the existing DP2 probe."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
STATE = {'pid': os.getpid(), 'phase': 'initializing', 'completed': False, 'started_unix': time.time()}
NAME = 'cyt-auto-mtp-v19-dp2-no-multistream'
ACTIVE = False


def status(phase, **extra):
    STATE.update(phase=phase, updated_unix=time.time(), **extra)
    path = ROOT / 'final_mtp_isolation_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, indent=2))
    temp.replace(path)
    print(phase, extra, flush=True)


def stop():
    global ACTIVE
    if ACTIVE:
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/service.py'), 'stop', '--name', NAME],
                       timeout=60, check=True)
        ACTIVE = False


def main():
    global ACTIVE
    prior = json.loads((ROOT / 'overnight_status.json').read_text())
    repeat = json.loads((ROOT / 'chunk_repeat_status.json').read_text())
    assert prior.get('completed') and repeat.get('completed'), 'finish ChunkSize and repeats first'
    args = [sys.executable, '-B', str(ROOT / 'scripts/service.py'), 'start', '--name', NAME,
            '--repaired', '--slim-draft', '--dp', '2', '--dp-attention', '--context', '180000',
            '--mrr', '8', '--diagnostics', '--env', 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0',
            '--env', 'SGLANG_NPU_USE_MULTI_STREAM=0']
    status('starting', container=NAME, command=args)
    subprocess.run(args, check=True, timeout=120)
    ACTIVE = True
    deadline = time.time() + 3600
    while time.time() < deadline:
        inspect = subprocess.run(['sudo', '-n', 'docker', 'inspect', NAME, '--format', '{{json .State}}'],
                                 capture_output=True, text=True, timeout=20)
        if inspect.returncode or not json.loads(inspect.stdout)['Running']:
            raise RuntimeError('isolation service exited before health')
        path = ROOT / 'logs' / NAME / 'server.log'
        if path.exists():
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 262144))
                tail = f.read()
            if b'crashed with exit code' in tail or b'scheduler died during initialization' in tail:
                raise RuntimeError('isolation scheduler failed before health')
        try:
            with urllib.request.urlopen('http://127.0.0.1:30080/health', timeout=3):
                pass
            break
        except Exception:
            status('waiting_health', container=NAME)
            time.sleep(20)
    else:
        raise RuntimeError('isolation service health timeout')
    out = ROOT / 'results/mtp-v19-dp2-no-multistream'
    args = [sys.executable, '-B', str(ROOT / 'scripts/mtp_probe_suite.py'), '--container', NAME,
            '--out', str(out), '--framing-pair', '--only', 'sanity_chat,native_exact_1k256,native_exact_eos_1k256,native_longinput_1k1024']
    with (ROOT / 'logs/final_mtp_isolation_suite.log').open('w') as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while process.poll() is None:
            status('running_suite', container=NAME, child_pid=process.pid, result=str(out / 'status.json'))
            path = ROOT / 'logs' / NAME / 'server.log'
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 262144))
                tail = f.read()
            if b'crashed with exit code' in tail:
                stop()
                process.wait(timeout=45)
                raise RuntimeError('isolation scheduler crashed; failed stream preserved')
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            if time.time() - started > 3600:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError('isolation suite timed out')
        if process.returncode:
            raise RuntimeError('isolation suite failed; preserve logs')
    state = json.loads((out / 'status.json').read_text())
    if state.get('target_prefill_audit_rc') != 0:
        raise RuntimeError('isolation target scoring incomplete')
    status('fixed_baseline_prefix_replay', container=NAME)
    with (ROOT / 'logs/final_mtp_fixed_prefix_replay.log').open('x') as log:
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/replay_fixed_prefixes.py'),
                        '--out', str(out / 'fixed_baseline_prefix_replay')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    stop()
    status('completed', completed=True, result=str(out))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('needs_conservative_review', error=repr(error))
        stop()
        raise
