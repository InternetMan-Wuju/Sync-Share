#!/usr/bin/env python3
"""Durable orchestration for the four complete native random-ids workloads."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from bench_official_mtp_full import capture, write_json

ROOT = Path('/home/chengyetai/workspace/current_exp/Q1MTPExp')
URL = 'http://127.0.0.1:30080'


def http(path, timeout=15):
    with urllib.request.urlopen(URL+path, timeout=timeout) as response:
        return response.read()


def docker_state(container):
    result = subprocess.run(['sudo','-n','docker','inspect',container,'--format','{{json .State}}'],
                            capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def report(out, statuses):
    lines = ['# 原生 random-ids 四组 MTP ON 重跑', '',
             '23单机，原生生成函数直接发送ID；第四组将原生ID前缀/后缀拼接以实现共享缓存。', '',
             '| 场景 | 输入/输出 | 并发/请求 | 校验 | AL | 草稿接受率 | P1/P2/P3 | 输出tok/s | TTFT均s | TPOT均ms | E2E均s | 缓存命中 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    manifest = json.loads((out/'manifest.json').read_text())
    for scenario in manifest['workloads']:
        name = scenario['name']
        result_file = out/name/'result.json'
        if not result_file.is_file():
            lines.append(f"| {name} | {scenario['input']}/{scenario['output']} | {scenario['concurrency']}/{scenario['requests']} | {statuses.get(name,'待运行')} | | | | | | | | |")
            continue
        data = json.loads(result_file.read_text())
        agg = data['aggregate']
        def value(key, digits=3):
            item = agg.get(key)
            return f'{item:.{digits}f}' if item is not None else '缺失'
        positions = '/'.join(f'{v:.4f}' for v in agg.get('pos_rates_weighted',[]))
        lines.append(f"| {name} | {scenario['input']}/{scenario['output']} | {scenario['concurrency']}/{scenario['requests']} | {'通过' if data['valid'] else '未通过'} | {value('AL_weighted')} | {value('accept_rate_weighted',4)} | {positions} | {value('output_tokens_per_s')} | {value('ttft_s_mean')} | {value('tpot_ms_mean')} | {value('e2e_s_mean')} | {value('cache_hit_ratio',4)} |")
    lines += ['', '只跑MTP ON；没有同配置OFF，本轮不能报告MTP加速收益。P1/P2/P3采用无条件位置概率，合计计数后求比值。',
              'TTFT从客户端发送到首次收到新增output_ids；TPOT=(最后新增输出时刻−首次新增输出时刻)/(输出数−1)。',
              'E2E为客户端请求至流结束；保存finalization时间、服务器queue/prefill时间戳和逐事件记录以便CPU分析。',
              '预热、清缓存、语义检查不计入正式workload的计数和墙钟。每个场景独立清缓存；缓存实测单列。',
              '样本数量较少，按AboutModels倍率执行；不以接受长度是否接近2判断通过。',
              '', '相对AboutModels的服务差异：v0.5.19镜像/实际模型路径，DP1与DP attention关闭，DeepEP normal，Graph被镜像禁用，',
              'context131072（文档180000）、MRR8（文档64）、shared-expert fusion关闭。MEM0.90/chunk16384/maxprefill32768/tokenizer-workers4保留。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    manifest = json.loads((out/'manifest.json').read_text())
    name = manifest['container']
    stop_monitor = threading.Event()
    monitor_thread = None
    statuses = {}
    state = {'run_id': manifest['run_id'], 'runner_pid': os.getpid(), 'container': name,
             'completed': False, 'state': 'waiting_for_health', 'scenario_status': statuses,
             'started_unix': time.time()}

    def status(stage, **extra):
        state.update(state=stage, updated_unix=time.time(), **extra)
        write_json(out/'experiment_status.json', state)
        print(stage, extra, flush=True)

    def monitor():
        directory = out/'monitor'
        directory.mkdir(exist_ok=True)
        while not stop_monitor.is_set():
            stamp = time.strftime('%Y%m%d_%H%M%S')
            for endpoint, suffix in (('/metrics','prom'),('/get_server_info','json')):
                try:
                    (directory/(stamp+'_'+suffix)).write_bytes(http(endpoint, timeout=10))
                except Exception as exc:
                    (directory/(stamp+'_'+suffix+'.error')).write_text(str(exc))
            result = subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20)
            (directory/(stamp+'_npu.txt')).write_text(result.stdout+result.stderr)
            write_json(directory/(stamp+'_container.json'),docker_state(name))
            stop_monitor.wait(60)

    failed = []
    try:
        status('waiting_for_health')
        for _ in range(360):
            if not docker_state(name)['Running']:
                raise RuntimeError('service container exited before healthy')
            try:
                http('/health', timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError('service health timeout')
        status('exporting_official_datasets')
        container_dir = '/exp-logs/native_full_datasets'
        exporter = ROOT/'scripts/export_official_randomids_full.py'
        with exporter.open('rb') as stdin:
            subprocess.run(['sudo','-n','docker','exec','-i','-e','PYTHONDONTWRITEBYTECODE=1',
                            '-e','HF_HOME=/exp-logs/hf-cache',name,'python3','-',container_dir],
                           stdin=stdin,check=True,timeout=300)
        import shutil
        shutil.copytree(out/'logs'/manifest['run_id']/'native_full_datasets',out/'datasets', dirs_exist_ok=True)
        (out/'server_info.json').write_bytes(http('/get_server_info'))
        inspect = subprocess.run(['sudo','-n','docker','inspect',name,
                                  '--format','{"image_id":{{json .Image}},"mounts":{{json .Mounts}},"state":{{json .State}}}'],
                                 capture_output=True,text=True,check=True)
        write_json(out/'container_provenance.json',json.loads(inspect.stdout))
        files = [ROOT/'scripts'/f for f in ('export_official_randomids_full.py','bench_official_mtp_full.py',
                 'run_official_mtp_full.py','serve_mtp_q1.sh','mtp_output_audit.py')]
        files += [ROOT.parent/'docs'/'AboutModels.md',ROOT.parent/'docs'/'MTP.md',
                  Path('/data_lib/data/models/glm-q/config.json'),Path('/data_lib/data/models/glm-q/tokenizer.json')]
        write_json(out/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        monitor_thread = threading.Thread(target=monitor,daemon=True)
        monitor_thread.start()
        status('sanity_chat')
        sanity = json.loads((out/'datasets/sanity_chat.json').read_text())
        sanity_results = []
        (out/'sanity').mkdir(exist_ok=True)
        for case in sanity['cases']:
            payload = {'input_ids':case['input_ids'],'stream':True,
                       'sampling_params':{'max_new_tokens':128,'temperature':0,'ignore_eos':False}}
            record = capture(URL,payload,out/'sanity'/f"request_{case['index']}.json",case,timeout=300)
            record['question'] = case['question']
            sanity_results.append(record)
            if record['error'] or not record['output_ids']:
                raise RuntimeError('sanity chat returned no output')
            print('sanity reply:',record['text'],flush=True)
        write_json(out/'sanity_chat.json',sanity_results)
        # Finish the short workloads first; the full 32K output runs last.
        for scenario in ('w1_8k_1k','w2_32k_1k','w4_prefix32k_300','w3_1k_32k'):
            statuses[scenario] = 'running'
            status('running',current_scenario=scenario)
            report(out,statuses)
            destination = out/scenario
            destination.mkdir(exist_ok=True)
            with (destination/'run.log').open('w') as log:
                result = subprocess.run([sys.executable,'-u',str(ROOT/'scripts/bench_official_mtp_full.py'),
                                         '--url',URL,'--dataset',str(out/'datasets'/(scenario+'.json')),
                                         '--out',str(destination)],stdout=log,stderr=subprocess.STDOUT)
            statuses[scenario] = 'completed' if result.returncode == 0 else 'failed'
            if result.returncode:
                failed.append(scenario)
            report(out,statuses)
            status('scenario_finished',current_scenario=scenario,exit_code=result.returncode)
            if not docker_state(name)['Running']:
                raise RuntimeError('service exited during workload')
        status('completed' if not failed else 'completed_with_failures',completed=not failed,failed_scenarios=failed)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop_monitor.set()
        if monitor_thread:
            monitor_thread.join(timeout=35)
        result = subprocess.run(['sudo','-n','docker','stop',name],capture_output=True,text=True,timeout=120)
        final = docker_state(name)
        state.update(container_preserved=True,container_running=final['Running'],
                     container_oom_killed=final['OOMKilled'],container_status=final['Status'],
                     stop_exit_code=result.returncode,updated_unix=time.time())
        write_json(out/'experiment_status.json',state)
        print('final container state',final,flush=True)


if __name__ == '__main__':
    main()
