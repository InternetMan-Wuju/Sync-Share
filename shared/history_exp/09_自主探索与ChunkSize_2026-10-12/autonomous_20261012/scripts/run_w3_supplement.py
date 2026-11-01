#!/usr/bin/env python3
"""Run the missing full W3: MTP ON, then four fixed-config ChunkSize cells."""
from report_labels import humanize_report
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
SCRIPTS = ROOT / 'scripts'
CURRENT = ROOT.parent
URL = 'http://127.0.0.1:30080'
sys.path.insert(0, str(CURRENT / 'Q1MTPExp/scripts'))
from bench_official_mtp_full import aggregate, capture, validate, write_json


def http(path, payload=None, timeout=15):
    request = urllib.request.Request(
        URL + path, data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runner:
    def __init__(self, run_id, adopt):
        if not run_id.replace('_', '').isalnum():
            raise ValueError('invalid run id')
        self.run_id = run_id
        self.out = ROOT / run_id
        self.out.mkdir(exist_ok=False)
        self.active = adopt
        self.state = dict(runner_pid=os.getpid(), run_id=run_id,
                          started_unix=time.time(), completed=False,
                          scope='W3 only; MTP OFF comparison explicitly cancelled',
                          cells=[], active_container=adopt)
        self.dataset = self.out / 'datasets/w3_1k_32k.json'
        self.dataset.parent.mkdir()
        source = CURRENT / 'Qchunksize/datasets/native_20261012'
        for name in ('w3_1k_32k.json', 'sanity_chat.json', 'installed_random.py'):
            shutil.copyfile(source / name, self.dataset.parent / name)
        ds = json.loads(self.dataset.read_text())
        assert ds['input_tokens'] == 1024 and ds['target_output_tokens'] == 32768
        assert ds['concurrency'] == ds['n_requests'] == len(ds['cases']) == 1
        for case in ds['cases']:
            assert len(case['input_ids']) == 1024
            assert hashlib.sha256(json.dumps(case['input_ids'], separators=(',', ':')).encode()).hexdigest() == case['input_sha256']
        self.state['dataset_sha256'] = digest(self.dataset)
        self.state['input_sha256'] = ds['cases'][0]['input_sha256']
        self.state['requested_matrix'] = [
            dict(task='MTP ON', chunk=16384, mtp=True, concurrency=1, requests=1),
            *[dict(task='ChunkSize', chunk=c, mtp=False, concurrency=1, requests=1)
              for c in (16384, 1024, 4096, 8192)]]
        write_json(self.out / 'plan.json', self.state)
        self.publish('initialized')
        self.report()

    def publish(self, phase, **extra):
        self.state.update(phase=phase, updated_unix=time.time(),
                          active_container=self.active, **extra)
        write_json(self.out / 'status.json', self.state)
        print(datetime.datetime.now().isoformat(timespec='seconds'), phase,
              json.dumps(extra, ensure_ascii=False), flush=True)

    def command(self, args, logfile, timeout=28800):
        self.publish('command_running', command=args, command_log=str(logfile))
        start = time.monotonic()
        with Path(logfile).open('a') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            self.state['child_pid'] = child.pid
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.publish('command_running', child_pid=child.pid,
                                 command_elapsed_s=round(time.monotonic() - start, 1))
                if time.monotonic() - start > timeout and child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
                    raise TimeoutError('command exceeded explicit deadline')
            self.state.pop('child_pid', None)
            if child.returncode:
                raise RuntimeError(f'command rc={child.returncode}; see {logfile}')

    def owned_state(self):
        result = subprocess.run(['sudo', '-n', 'docker', 'inspect', self.active],
                                capture_output=True, text=True, timeout=20, check=True)
        container = json.loads(result.stdout)[0]
        if container['Config'].get('Labels', {}).get('owner') != 'chengyetai':
            raise RuntimeError('container ownership differs')
        return container['State']

    def ready(self, cell):
        started = time.monotonic()
        while time.monotonic() - started < 3600:
            if not self.owned_state()['Running']:
                raise RuntimeError('owned service exited before health')
            log = ROOT / 'logs' / self.active / 'server.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    tail = stream.read().decode(errors='replace')
                if ('Subprocess scheduler_' in tail and 'crashed with exit code' in tail) or 'scheduler died during initialization' in tail:
                    raise RuntimeError('scheduler crashed during initialization')
            try:
                http('/health', timeout=3)
                info = json.loads(http('/get_server_info', timeout=30))
            except Exception:
                self.publish('waiting_health', cell=cell['id'],
                             wait_s=round(time.monotonic() - started, 1))
                time.sleep(10)
                continue
            expected = dict(tp_size=16, ep_size=16, dp_size=1,
                            enable_dp_attention=False, enable_dp_lm_head=False,
                            chunked_prefill_size=cell['chunk'], context_length=131072,
                            deepep_mode='normal' if cell['mtp'] else 'auto',
                            disable_cuda_graph=cell['mtp'],
                            max_running_requests=8 if cell['mtp'] else 16)
            mismatches = {k: (value, info.get(k)) for k, value in expected.items()
                          if info.get(k) != value}
            if mismatches:
                raise RuntimeError(f'CLI/effective config mismatch: {mismatches}')
            if bool(info.get('speculative_algorithm')) != cell['mtp']:
                raise RuntimeError('effective MTP mode differs')
            if info['max_total_num_tokens'] < 1024 + 32768 + 512:
                raise RuntimeError('KV capacity cannot hold the full W3')
            if cell['mtp']:
                assert info['speculative_num_steps'] == 3
                assert info['speculative_num_draft_tokens'] == 4
            cell['effective_config'] = {k: info.get(k) for k in (*expected, 'version', 'max_total_num_tokens', 'speculative_algorithm', 'cuda_graph_config')}
            write_json(cell['out'] / 'preflight_server_info.json', info)
            if cell['mtp']:
                write_json(cell['out'].parent / 'server_info.json', info)
            self.publish('service_healthy', cell=cell['id'], kv_tokens=info['max_total_num_tokens'])
            return
        raise TimeoutError('health deadline 3600s exceeded')

    def sanity(self, cell):
        self.publish('sanity', cell=cell['id'])
        ds = json.loads((self.dataset.parent / 'sanity_chat.json').read_text())
        checks = []
        for case in ds['cases']:
            payload = dict(input_ids=case['input_ids'], stream=True,
                           sampling_params=dict(max_new_tokens=128, temperature=0))
            record = capture(URL, payload, cell['out'] / f'sanity_{case["index"]}.json', case, timeout=900)
            visible = record['text'].strip()
            readable = '你好' in visible if case['index'] == 0 else visible == '2'
            checks.append(dict(index=case['index'], text=visible, valid=(readable and not record['error'] and record['stream_done'])))
        write_json(cell['out'] / 'sanity_checks.json', checks)
        if not all(row['valid'] for row in checks):
            raise RuntimeError(f'sanity failed: {checks}')

    def stop(self):
        if self.active:
            name = self.active
            self.publish('stopping_owned_container', container=name)
            subprocess.run([sys.executable, '-B', str(SCRIPTS / 'service.py'),
                            'stop', '--name', name], check=True, timeout=90)
            self.active = None
        for _ in range(12):
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                                    timeout=20, check=True)
            if result.stdout.count('No running processes found') == 8:
                write_json(self.out / 'release.json', dict(checked_unix=time.time(), free=True))
                (self.out / 'npu_after.txt').write_text(result.stdout)
                return
            time.sleep(5)
        raise RuntimeError('NPU not empty after stopping our service; preserve other users')

    def run_cell(self, *, mtp, chunk):
        cell_id = 'mtp_on_w3' if mtp else f'chunk{chunk}_w3_c1'
        target = (ROOT / 'results' / self.run_id / 'w3_1k_32k' if mtp else
                  CURRENT / 'Qchunksize/results' / self.run_id / cell_id)
        target.mkdir(parents=True, exist_ok=False)
        cell = dict(id=cell_id, mtp=mtp, chunk=chunk, out=target, valid=False)
        self.state['cells'].append(cell)
        # Paths are serialized explicitly in publish/report, not passed to json as Path.
        cell['out'] = str(target)
        self.publish('cell_start', cell=cell_id)
        cell['out'] = target
        try:
            if mtp:
                if not self.active:
                    raise RuntimeError('initial MTP ON container must already exist')
            else:
                self.active = f'cyt-auto-w3-chunk-{chunk}-20261013'
                flags = ['--mtp', 'off', '--dp', '1', '--ep', '16', '--deepep', 'auto',
                         '--graph', '--mrr', '16', '--graph-bs', '1 2 4 8 16',
                         '--graph-max-bs', '16', '--dispatch-tokens', '64',
                         '--env', 'HCCL_BUFFSIZE=650', '--env', 'HCCL_OP_EXPANSION_MODE=AIV',
                         '--context', '131072', '--mem', '0.90', '--maxprefill', '32768',
                         '--chunk', str(chunk)]
                self.command([sys.executable, '-B', str(SCRIPTS / 'service.py'), 'start',
                              '--name', self.active, '--repaired', '--slim-draft', *flags],
                             self.out / f'{cell_id}_launch.log', timeout=120)
            cell['container'] = self.active
            self.ready(cell)
            self.sanity(cell)
            args = [sys.executable, '-B', str(SCRIPTS / 'chunk_collect.py'),
                    '--dataset', str(self.dataset), '--out', str(target),
                    '--concurrency', '1', '--chunk', str(chunk)]
            if mtp:
                args.append('--mtp-on')
            self.command(args, self.out / f'{cell_id}_collect.log')
            result = json.loads((target / 'result.json').read_text())
            records = result['requests']
            assert len(records) == 1 and result['concurrency'] == 1
            assert digest(self.dataset) == self.state['dataset_sha256']
            checks = []
            for record in records:
                assert record['input_sha256'] == self.state['input_sha256']
                assert record['input_tokens_sent'] == record['meta']['prompt_tokens'] == 1024
                assert record['completion_tokens'] == len(record['output_ids']) == record['meta']['completion_tokens'] == 32768
                assert record['sampling_params']['max_new_tokens'] == 32768
                if mtp:
                    checks.append(validate(record, 1024, 32768))
            assert all(not item['failures'] for item in checks), checks
            if mtp:
                result['aggregate'].update(aggregate(records, result['aggregate']['wall_s']))
            result.update(container=self.active, target_output_tokens=32768, ignore_eos=True,
                          validation=checks, valid=True,
                          configuration_scope='W3 supplemental pass; runnable DP1 config; not the default DP2 leaderboard configuration',
                          mtp_off_comparison=False)
            write_json(target / 'result.json', result)
            cell.update(valid=True, aggregate=result['aggregate'], result=str(target / 'result.json'),
                        finished_unix=time.time())
            self.publish('cell_completed', cell=cell_id, aggregate=result['aggregate'])
        except Exception as error:
            cell['error'] = f'{type(error).__name__}: {error}'
            self.publish('cell_failed', cell=cell_id, error=cell['error'])
        finally:
            self.stop()
            self.report()

    def report(self):
        clean = json.loads(json.dumps(self.state, default=str))
        write_json(self.out / 'status.json', clean)
        rows = [
            '# W3 完整长输出补跑', '',
            '范围：补跑缺失的 W3（输入 1024 / 输出 32768）。MTP 只跑 ON；用户已取消 MTP OFF 对照。ChunkSize 固定关闭 MTP，复用原自主实验配置。', '',
            'W3 文档未指定并发；本轮明确采用并发 1、请求 1。所有五项使用同一份 SGLang 原生 random-ids 输入，固定 seed，不缩短输出。每项先做问候/算术检查、16-token 形状预热，再清缓存实测。', '',
            '配置差异：MTP 使用此前成功的 v0.5.19、TP16/EP16/DP1、DP attention 关闭、DeepEP normal、graph 关闭、MRR8；ChunkSize 为同版本同并行、DeepEP auto、decode graph 开启、MRR16。两者均 mem0.90/context131072/最大prefill32768、修复权重只读外挂。它们不是严格开关配对，不计算 MTP 加速比。', '',
            'W3 只有 1024 个输入 token，四个 ChunkSize 均可一次容纳完整输入；因此本场景通常不能体现长输入多 chunk 的主要差异。仍保留四项独立实测。单请求结果仅描述该输入，不代表总体。', '',
            '| 项目 | ChunkSize | 状态 | 实际输出 | TTFT s | TPOT ms | E2E s | 总输出 tok/s | AL | P1 / P2 / P3 |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---|',
        ]
        def fmt(value):
            return '—' if value is None else f'{value:.4f}'
        for planned in self.state['requested_matrix']:
            cell = next((c for c in self.state['cells'] if c['mtp'] == planned['mtp'] and c['chunk'] == planned['chunk']), None)
            a = cell.get('aggregate', {}) if cell else {}
            state = ('完成并验收' if cell.get('valid') else '失败：' + cell['error'] if cell.get('error') else '进行中') if cell else '待运行'
            positions = a.get('pos_rates_weighted')
            ptext = ' / '.join(fmt(p) for p in positions) if positions else '—'
            rows.append(f"| {planned['task']} | {planned['chunk']} | {state} | {a.get('completion_tokens', a.get('output_tokens', '—'))} | {fmt(a.get('ttft_s_mean'))} | {fmt(a.get('tpot_ms_mean'))} | {fmt(a.get('e2e_s_mean'))} | {fmt(a.get('total_output_tok_s'))} | {fmt(a.get('AL_weighted'))} | {ptext} |")
        rows += ['', '输入与生成器来源见 [datasets](datasets/)。输入文件 SHA256：`' + self.state['dataset_sha256'] + '`。', '',
                 '每项保存 payload、完整输出 IDs/文本、压缩 SSE、客户端时间戳、服务器元信息、指标前后及采样、实际服务配置、预热和 sanity。只有固定长度、流结束、输入 SHA、缓存条件和 MTP 直方图全部验收后才标记完成。', '']
        for cell in self.state['cells']:
            if cell.get('result'):
                rows.append(f"- [{cell['id']} 原始结果]({cell['result']})")
        (self.out / 'W3补跑结果.md').write_text(humanize_report('\n'.join(rows)))

    def run(self):
        try:
            self.run_cell(mtp=True, chunk=16384)
            for chunk in (16384, 1024, 4096, 8192):
                self.run_cell(mtp=False, chunk=chunk)
            complete = len(self.state['cells']) == 5 and all(c['valid'] for c in self.state['cells'])
            self.publish('completed' if complete else 'incomplete', completed=complete,
                         finished_unix=time.time())
        except BaseException as error:
            self.publish('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if self.active:
                self.stop()
            self.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='w3_supplement_20261013')
    parser.add_argument('--adopt', default='cyt-auto-w3-mtp-on-20261013')
    args = parser.parse_args()
    # Serialize Path objects only in this orchestration status; experiment JSON
    # retains the existing collectors' validated native types.
    global write_json
    original = write_json
    def serializable_write(path, obj):
        original(path, json.loads(json.dumps(obj, default=str)))
    write_json = serializable_write
    Runner(args.run_id, args.adopt).run()


if __name__ == '__main__':
    main()
