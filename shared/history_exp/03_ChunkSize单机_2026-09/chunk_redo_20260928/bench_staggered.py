#!/usr/bin/env python3
"""Four fixed-32K/300 requests, staggered to expose Prefill/Decode interference.

All raw SSE frames, input IDs, output IDs, server meta, client event timestamps,
service metrics, and NPU samples are stored beneath --out-dir. Never abort an
in-flight request when the 30-minute dispatch deadline expires.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import random
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PREFIX_TOKENS = 19661
TOTAL_TOKENS = 32768
VOCAB_SIZE = 154880
NAMES = ('A', 'B', 'C', 'D')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def make_dataset(path, seed):
    if path.exists():
        data = json.loads(path.read_text())
        if data['seed'] != seed or data['prompt_tokens'] != TOTAL_TOKENS:
            raise ValueError('dataset parameters disagree with existing file')
        return data
    rng = random.Random(seed)
    prefix = [rng.randrange(1, VOCAB_SIZE) for _ in range(PREFIX_TOKENS)]
    cases = {}
    for name in NAMES:
        ids = prefix + [rng.randrange(1, VOCAB_SIZE) for _ in range(TOTAL_TOKENS - PREFIX_TOKENS)]
        cases[name] = {'input_ids': ids, 'sha256': digest(ids)}
    data = {'seed': seed, 'vocab_id_range': [1, VOCAB_SIZE - 1],
            'prefix_tokens': PREFIX_TOKENS, 'prompt_tokens': TOTAL_TOKENS,
            'requested_output_tokens': 300, 'prefix_ids': prefix,
            'prefix_sha256': digest(prefix), 'cases': cases}
    save(path, data)
    return data


class StreamRequest:
    def __init__(self, name, ids, out_dir, url, max_new_tokens, input_digest):
        self.name = name
        self.ids = ids
        self.out_dir = out_dir
        self.url = url.rstrip('/')
        self.max_new_tokens = max_new_tokens
        self.input_digest = input_digest
        self.count = 0
        self.first_event = threading.Event()
        self.twenty_event = threading.Event()
        self.result = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        payload = {'input_ids': self.ids, 'sampling_params':
                   {'max_new_tokens': self.max_new_tokens, 'temperature': 0, 'ignore_eos': True},
                   'stream': True}
        save(self.out_dir / 'request.json', {'endpoint': self.url + '/generate',
            'payload': payload, 'input_sha256': self.input_digest})
        req = urllib.request.Request(self.url + '/generate', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'}, method='POST')
        began_ns = time.perf_counter_ns()
        began_wall_ns = time.time_ns()
        first_ns = None
        completed = 0
        output_ids = []
        meta = {}
        headers = {}
        status = None
        finish_reason = None
        last_frame_ns = None
        event_count = 0
        error = None
        try:
            with urllib.request.urlopen(req, timeout=1500) as response, gzip.open(
                    self.out_dir / 'sse_raw.jsonl.gz', 'wt', encoding='utf-8') as frames:
                status = response.status
                headers = dict(response.headers)
                for line in response:
                    if not line.startswith(b'data:'):
                        frames.write(json.dumps({'recv_perf_ns': time.perf_counter_ns(),
                            'recv_wall_ns': time.time_ns(), 'raw_line': line.decode('utf-8','replace')}) + '\n')
                        continue
                    now = time.perf_counter_ns()
                    wall = time.time_ns()
                    text = line.decode('utf-8','replace')
                    frames.write(json.dumps({'recv_perf_ns': now, 'recv_wall_ns': wall,
                        'raw_line': text}, ensure_ascii=False) + '\n')
                    if line[5:].strip() == b'[DONE]':
                        break
                    obj = json.loads(line[5:])
                    event_count += 1
                    if obj.get('error'):
                        raise RuntimeError(f"server error: {obj['error']}")
                    server = obj.get('meta_info') or {}
                    meta.update(server)
                    ids = obj.get('output_ids') or []
                    output_ids.extend(ids)
                    value = server.get('completion_tokens')
                    if isinstance(value, int):
                        completed = max(completed, value)
                    else:
                        completed += len(ids)
                    self.count = completed
                    if first_ns is None and completed > 0:
                        first_ns = now
                        self.first_event.set()
                    if completed >= 20:
                        self.twenty_event.set()
                    if isinstance(server.get('finish_reason'), dict):
                        finish_reason = server['finish_reason']
                    last_frame_ns = now
        except urllib.error.HTTPError as exc:
            status = exc.code
            error = f'HTTPError({exc.code}): {exc.read(4096).decode("utf-8", "replace")}'
        except Exception:
            error = traceback.format_exc()
        ended_ns = time.perf_counter_ns()
        ended_wall_ns = time.time_ns()
        output = meta.get('completion_tokens', completed)
        if not isinstance(output, int):
            output = completed
        r = {'name': self.name, 'http_status': status, 'http_headers': headers,
             'input_sha256': self.input_digest, 'prompt_tokens': meta.get('prompt_tokens'),
             'cached_tokens': meta.get('cached_tokens'), 'requested_output_tokens': self.max_new_tokens,
             'output_tokens': output, 'finish_reason': finish_reason, 'error': error,
             'started_perf_ns': began_ns, 'started_wall_ns': began_wall_ns,
             'first_token_perf_ns': first_ns, 'last_token_perf_ns': last_frame_ns,
             'ended_perf_ns': ended_ns, 'ended_wall_ns': ended_wall_ns,
             'ttft_s': (first_ns-began_ns)/1e9 if first_ns else None,
             'e2e_s': (ended_ns-began_ns)/1e9,
             'tpot_ms': (ended_ns-first_ns)/1e6/(output-1) if first_ns and output>1 else None,
             'event_count': event_count, 'output_ids_observed': len(output_ids),
             'output_ids_sha256': digest(output_ids), 'meta_info': meta}
        save(self.out_dir / 'result.json', r)
        save(self.out_dir / 'output_ids.json', output_ids)
        self.result = r
        self.first_event.set()
        self.twenty_event.set()
        return r


def flush_cache(url, path):
    req = urllib.request.Request(url.rstrip('/') + '/flush_cache', b'', method='POST')
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            result = {'status': res.status, 'body': res.read().decode('utf-8', 'replace')}
    except Exception:
        result = {'error': traceback.format_exc()}
    save(path, result)
    if result.get('status') != 200:
        raise RuntimeError(f'cache flush failed: {result}')


def sampler(url, out_dir, stop):
    with gzip.open(out_dir / 'metrics_raw.jsonl.gz', 'wt', encoding='utf-8') as f:
        iteration = 0
        while not stop.is_set():
            sample = {'t_perf_ns': time.perf_counter_ns(), 't_wall_ns': time.time_ns()}
            try:
                with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=8) as res:
                    sample['metrics_http'] = res.status
                    sample['metrics_raw'] = res.read().decode('utf-8', 'replace')
            except Exception as exc:
                sample['metrics_error'] = repr(exc)
            if iteration % 3 == 0:
                try:
                    proc = subprocess.run(['npu-smi','info'], capture_output=True, text=True, timeout=12)
                    sample['npu_smi'] = {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
                except Exception as exc:
                    sample['npu_smi_error'] = repr(exc)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            f.flush()
            iteration += 1
            stop.wait(5)


def wait_for_milestone(req, future, deadline_ns):
    while True:
        remaining = (deadline_ns-time.perf_counter_ns())/1e9
        if remaining <= 0:
            return False, '30-minute dispatch deadline'
        if req.twenty_event.wait(min(remaining, 1)):
            if req.count >= 20 and not future.done():
                return True, '20 tokens emitted'
            return False, f'{req.name} ended before reaching 20 output tokens'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://10.120.72.23:30080')
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--dataset', required=True, type=Path)
    ap.add_argument('--chunk', required=True, type=int, choices=(8192,1024))
    ap.add_argument('--seed', type=int, default=2026092801)
    ap.add_argument('--time-limit', type=int, default=1800)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    data = make_dataset(a.dataset, a.seed)
    save(a.out_dir / 'plan.json', {'url':a.url, 'chunk':a.chunk,
         'time_limit_s': a.time_limit, 'dataset':str(a.dataset),
         'input_sha256':{k:v['sha256'] for k,v in data['cases'].items()},
         'rule':'A20->B; B20->C; after A completes and C20->D; max three active; four total'})
    prefix = data['prefix_ids']
    flush_cache(a.url, a.out_dir/'flush_cache.json')
    warmup = StreamRequest('warmup', prefix, a.out_dir/'warmup', a.url, 1, digest(prefix)).run()
    if warmup['error'] or warmup['output_tokens'] != 1:
        raise RuntimeError(f'prefix warmup failed: {warmup}')
    requests = {name: StreamRequest(name, data['cases'][name]['input_ids'],
        a.out_dir/'requests'/name, a.url, 300, data['cases'][name]['sha256']) for name in NAMES}
    dispatches = []
    stop_sampler = threading.Event()
    thread = threading.Thread(target=sampler, args=(a.url,a.out_dir,stop_sampler),daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            start = time.perf_counter_ns()
            deadline = start + int(a.time_limit*1e9)
            futures = {}
            def send(name, trigger):
                if time.perf_counter_ns() >= deadline:
                    dispatches.append({'name': name, 'trigger':trigger, 'submitted':False,
                        'reason':'30-minute dispatch deadline'})
                    return None
                entry={'name':name,'trigger':trigger,'submitted':True,
                       'dispatch_perf_ns':time.perf_counter_ns(), 'dispatch_wall_ns':time.time_ns()}
                dispatches.append(entry)
                print(f"[{time.strftime('%H:%M:%S')}] submit {name}: {trigger}", flush=True)
                futures[name]=pool.submit(requests[name].run)
                save(a.out_dir/'dispatches.json', dispatches)
                return futures[name]
            if send('A','start') is not None:
                ok,reason=wait_for_milestone(requests['A'], futures['A'],deadline)
                if ok and send('B','A emitted 20 tokens') is not None:
                    ok,reason=wait_for_milestone(requests['B'],futures['B'],deadline)
                    if ok and send('C','B emitted 20 tokens') is not None:
                        ok,reason=wait_for_milestone(requests['C'],futures['C'],deadline)
                        if ok:
                            futures['A'].result()  # frees KV; C must still be outputting to collide with D
                            still_outputting = not futures['C'].done()
                            send('D',f'A finished, C emitted >=20, C_live={still_outputting}')
                        else:
                            dispatches.append({'name':'D','submitted':False,'reason':reason})
                    else:
                        dispatches.extend({'name':n,'submitted':False,'reason':reason} for n in ('C','D'))
                else:
                    dispatches.extend({'name':n,'submitted':False,'reason':reason} for n in ('B','C','D'))
            save(a.out_dir/'dispatches.json', dispatches)
            results = {name: future.result() for name,future in futures.items()}
            save(a.out_dir/'all_results.json', results)
    finally:
        stop_sampler.set()
        thread.join(timeout=20)
    verdict = {name:{'status':r['http_status'],'output':r['output_tokens'],
        'cached':r['cached_tokens'],'prompt':r['prompt_tokens'],'error':r['error'],
        'valid_length':r['output_tokens']==300 and r['prompt_tokens']==TOTAL_TOKENS,
        'valid_cache':isinstance(r['cached_tokens'],int) and
             0.585 <= r['cached_tokens']/TOTAL_TOKENS <= 0.615}
        for name,r in results.items()}
    save(a.out_dir/'verdict.json', verdict)
    print(json.dumps(verdict,ensure_ascii=False,indent=2),flush=True)
    if len(results)!=4 or any(not v['valid_length'] or not v['valid_cache'] or v['error'] for v in verdict.values()):
        raise SystemExit('experiment incomplete: see verdict.json; never call it a completed four-request comparison')

if __name__ == '__main__':
    main()
