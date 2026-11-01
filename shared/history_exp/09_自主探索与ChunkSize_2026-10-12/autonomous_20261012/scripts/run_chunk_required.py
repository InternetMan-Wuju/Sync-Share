#!/usr/bin/env python3
"""Only the latest required Chunk workload: W1 C32/N64."""
import hashlib
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_w3_supplement import Runner, ROOT, SCRIPTS, CURRENT, http, digest
from bench_official_mtp_full import write_json

ADOPT = 'cyt-auto-chunk-w1-noep-8192-20261013'


class ChunkRunner(Runner):
    def __init__(self, mem=0.90, attempt='', adopt=ADOPT):
        if attempt and not attempt.isalnum():
            raise ValueError('invalid attempt name')
        self.mem = mem
        self.attempt = attempt
        self.run_id = 'chunk_required_20261013' + ('_'+attempt if attempt else '')
        self.out = CURRENT / ('Qchunksize/retest_20261013' + ('_'+attempt if attempt else ''))
        self.out.mkdir(exist_ok=True)
        self.active = adopt
        self.dataset = self.out / 'datasets/w1_8k_1k.json'
        self.dataset.parent.mkdir(exist_ok=False)
        source = ROOT / 'logs' / ADOPT / 'native_required_datasets'
        for path in source.iterdir():
            if path.is_file():
                shutil.copyfile(path, self.dataset.parent / path.name)
        ds1 = json.loads(self.dataset.read_text())
        assert ds1['concurrency'] == 32 and ds1['n_requests'] == len(ds1['cases']) == 64
        inputs = {}
        for ds in (ds1,):
            assert all(len(c['input_ids']) == ds['input_tokens'] for c in ds['cases'])
            for case in ds['cases']:
                assert hashlib.sha256(json.dumps(case['input_ids'], separators=(',', ':')).encode()).hexdigest() == case['input_sha256']
            inputs[ds['name']] = dict(sha256=digest(self.dataset.parent / (ds['name'] + '.json')),
                                     concurrency=ds['concurrency'], requests=ds['n_requests'],
                                     input_tokens=ds['input_tokens'], output_tokens=ds['target_output_tokens'])
        self.state = dict(runner_pid=os.getpid(), run_id=self.run_id,
                          started_unix=time.time(), completed=False, cells=[], inputs=inputs,
                          scope='Only W1 8192/1024 C32/N64; W4 already tested and explicitly excluded by user',
                          active_container=adopt, configurations={}, mem_fraction_static=mem,
                          requested_matrix=[dict(scene=scene, chunk=chunk, **inputs[scene])
                                            for scene in inputs for chunk in (8192, 1024, 4096, 16384)])
        write_json(self.out / 'plan.json', self.state)
        self.publish('initialized')
        self.report()

    def ready(self, cell):
        start = time.monotonic()
        while time.monotonic() - start < 3600:
            if not self.owned_state()['Running']:
                raise RuntimeError('service exited before health')
            log = ROOT / 'logs' / self.active / 'server.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    tail = stream.read().decode(errors='replace')
                if ('Subprocess scheduler_' in tail and 'crashed with exit code' in tail) or 'scheduler died during initialization' in tail:
                    raise RuntimeError('scheduler initialization crash; see preserved server.log')
            try:
                http('/health', timeout=3)
                info = json.loads(http('/get_server_info', timeout=30))
            except Exception:
                self.publish('waiting_health', cell=cell['id'], wait_s=round(time.monotonic()-start, 1))
                time.sleep(10)
                continue
            w1 = cell['scene'] == 'w1_8k_1k'
            expected = dict(tp_size=16, ep_size=1 if w1 else 16, dp_size=2 if w1 else 1,
                            enable_dp_attention=w1, enable_dp_lm_head=w1,
                            moe_a2a_backend='none' if w1 else 'deepep',
                            context_length=180000 if w1 else 131072,
                            chunked_prefill_size=cell['chunk'], max_prefill_tokens=32768)
            if info.get('speculative_algorithm'):
                raise RuntimeError('Chunk requires fixed MTP OFF')
            mismatch = {k: [v, info.get(k)] for k, v in expected.items() if info.get(k) != v}
            if mismatch:
                raise RuntimeError(f'actual service config differs: {mismatch}')
            if info.get('disable_cuda_graph') or info['cuda_graph_config']['decode']['backend'] != 'full':
                raise RuntimeError('decode graph did not remain enabled')
            if info['max_total_num_tokens'] < (8192+1024+512 if w1 else 32768+300+512):
                raise RuntimeError('insufficient KV for one full request; concurrency will not be reduced')
            fixed = {k: info.get(k) for k in (*expected, 'version', 'mem_fraction_static',
                                             'max_running_requests', 'disable_shared_experts_fusion',
                                             'deepep_mode', 'disable_cuda_graph', 'cuda_graph_config',
                                             'max_total_num_tokens') if k != 'chunked_prefill_size'}
            previous = self.state['configurations'].get(cell['scene'])
            if previous and previous != fixed:
                raise RuntimeError(f'non-Chunk config drift: previous={previous}; current={fixed}')
            self.state['configurations'][cell['scene']] = fixed
            cell['effective_config'] = {**fixed, 'chunked_prefill_size': cell['chunk']}
            write_json(Path(cell['out']) / 'preflight_server_info.json', info)
            self.publish('service_healthy', cell=cell['id'], kv_tokens=info['max_total_num_tokens'])
            return
        raise TimeoutError('health deadline 3600s exceeded')

    def run_cell(self, scene, chunk, adopt=False):
        assert scene == 'w1_8k_1k', 'latest user scope permits W1 Chunk only'
        inputs = self.state['inputs'][scene]
        self.dataset = self.out / 'datasets' / (scene + '.json')
        cell_id = f'{scene}_chunk{chunk}_c{inputs["concurrency"]}'
        target = CURRENT / 'Qchunksize/results' / self.run_id / cell_id
        target.mkdir(parents=True, exist_ok=False)
        cell = dict(id=cell_id, scene=scene, chunk=chunk, out=str(target), valid=False)
        self.state['cells'].append(cell)
        self.publish('cell_start', cell=cell_id)
        try:
            w1 = scene == 'w1_8k_1k'
            if not adopt:
                short = 'w1-noep' if w1 else 'w4-ep'
                self.active = f'cyt-auto-chunk-{short}-{chunk}' + ('-'+self.attempt if self.attempt else '') + '-20261013'
                common = ['--mtp', 'off', '--graph', '--mem', str(self.mem), '--maxprefill', '32768']
                if w1:
                    # This SGLang divides the CLI Chunk by DP2; compare effective per-DP Chunk.
                    flags = common + ['--dp', '2', '--dp-attention', '--ep', '1', '--backend', 'none',
                                      '--fusion', '--graph-bs', '8 16 24 32', '--graph-max-bs', '32',
                                      '--context', '180000', '--mrr', '64', '--chunk', str(chunk*2)]
                else:
                    flags = common + ['--dp', '1', '--ep', '16', '--deepep', 'auto', '--mrr', '16',
                                      '--graph-bs', '1 2 4 8 16', '--graph-max-bs', '16', '--dispatch-tokens', '64',
                                      '--env', 'HCCL_BUFFSIZE=650', '--env', 'HCCL_OP_EXPANSION_MODE=AIV',
                                      '--context', '131072', '--chunk', str(chunk)]
                self.command([sys.executable, '-B', str(SCRIPTS/'service.py'), 'start',
                              '--name', self.active, '--repaired', *flags], self.out/(cell_id+'_launch.log'), timeout=120)
            cell['container'] = self.active
            self.ready(cell)
            # Inherited sanity reads the new dataset directory's native chat templates.
            sanity_cell = {**cell, 'out': target}
            self.sanity(sanity_cell)
            self.command([sys.executable, '-B', str(SCRIPTS/'chunk_collect.py'),
                          '--dataset', str(self.dataset), '--out', str(target),
                          '--concurrency', str(inputs['concurrency']), '--chunk', str(chunk)],
                         self.out/(cell_id+'_collect.log'), timeout=21600)
            result = json.loads((target/'result.json').read_text())
            assert result['dataset_sha256'] == inputs['sha256'] == digest(self.dataset)
            assert result['concurrency'] == inputs['concurrency']
            assert len(result['requests']) == inputs['requests']
            assert result['aggregate']['complete'] and result['aggregate']['cache_condition_valid']
            for request, case in zip(result['requests'], json.loads(self.dataset.read_text())['cases']):
                assert request['input_sha256'] == case['input_sha256']
                assert request['input_tokens_sent'] == request['meta']['prompt_tokens'] == inputs['input_tokens']
                assert request['completion_tokens'] == len(request['output_ids']) == inputs['output_tokens']
                if not w1:
                    assert .585 <= request['meta'].get('cached_tokens', 0)/inputs['input_tokens'] <= .615
            result.update(container=self.active, valid=True,
                          effective_config=cell['effective_config'],
                          scope='W1 C32/N64 only; only Chunk changes',
                          chunk_semantics='effective per-DP ChunkSize; W1 CLI value is twice effective under DP2')
            write_json(target/'result.json', result)
            cell.update(valid=True, aggregate=result['aggregate'], result=str(target/'result.json'), finished_unix=time.time())
            self.publish('cell_completed', cell=cell_id, aggregate=result['aggregate'])
        except Exception as error:
            cell['error'] = f'{type(error).__name__}: {error}'
            self.publish('cell_failed', cell=cell_id, error=cell['error'])
            raise
        finally:
            self.stop()
            self.report()

    def report(self):
        lines = ['# ChunkSize W1 并发32补测', '',
                 '最新范围：只测8K/1K。用户已明确前缀32K/300/60%组测过，不再重复；W2/W3 Chunk也不跑。W3 MTP ON已完整补齐。', '',
                 '客户端并发32、正式64请求，TTFT包含排队。固定输入SHA、温度0/ignore_eos、先形状预热再清缓存。四个Chunk档位保持同一输入和其余配置。', '',
                 f'W1采用榜单NoEP布局：TP16/DP2/EP1、DP attention/LM head开、DeepEP关、graph开、mem{self.mem:.2f}/context180000/MRR64。v0.5.19和修复权重是可运行基础，与文档默认镜像标签不同。首次mem0.90在并发32的形状预热发生prefill OOM，没有正式结果；重试只修改mem比例，不降低客户端并发。DP2会把CLI Chunk除以2，表里核对并使用实际每DP Chunk；四档对应CLI2048/8192/16384/32768。', '',
                 '| 场景 | 实际Chunk | 状态 | 请求通过/要求 | 总输出 tok/s | TTFT均值 s | TPOT均值 ms | E2E均值 s | 实测缓存 |',
                 '|---|---:|---|---|---:|---:|---:|---:|---:|']
        fmt = lambda x: '—' if x is None else f'{x:.4f}'
        for plan in self.state['requested_matrix']:
            cell = next((c for c in self.state['cells'] if c['scene'] == plan['scene'] and c['chunk'] == plan['chunk']), None)
            a = cell.get('aggregate', {}) if cell else {}
            status = ('完成验收' if cell['valid'] else '失败：'+cell['error'] if cell.get('error') else '进行中') if cell else '待运行'
            lines.append(f"| {plan['scene']} | {plan['chunk']} | {status} | {a.get('requests_ok', '—')}/{plan['requests']} | {fmt(a.get('total_output_tok_s'))} | {fmt(a.get('ttft_s_mean'))} | {fmt(a.get('tpot_ms_mean'))} | {fmt(a.get('e2e_s_mean'))} | {fmt(a.get('cache_hit_ratio'))} |")
        lines += ['', '逐请求、p50/p95、E2E阶段分解、metrics和原始压缩SSE保存于各结果目录；sanity与预热分开保存且不计正式时间。排队和阶段跨度来自服务器时间戳，不将它们解释成纯算子时间。', '']
        for cell in self.state['cells']:
            if cell.get('result'):
                lines.append(f"- [{cell['id']} 原始结果]({cell['result']})")
        (self.out/'ChunkSize_W1并发32补测.md').write_text('\n'.join(lines))

    def run(self):
        try:
            for scene in ('w1_8k_1k',):
                for chunk in (8192, 1024, 4096, 16384):
                    self.run_cell(scene, chunk, adopt=(scene=='w1_8k_1k' and chunk==8192))
            complete = len(self.state['cells']) == 4 and all(c['valid'] for c in self.state['cells'])
            self.publish('completed' if complete else 'incomplete', completed=complete, finished_unix=time.time())
        except BaseException as error:
            self.publish('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if self.active:
                self.stop()
            self.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mem', type=float, default=0.90)
    parser.add_argument('--attempt', default='')
    parser.add_argument('--adopt', default=ADOPT)
    args = parser.parse_args()
    ChunkRunner(args.mem, args.attempt, args.adopt).run()
