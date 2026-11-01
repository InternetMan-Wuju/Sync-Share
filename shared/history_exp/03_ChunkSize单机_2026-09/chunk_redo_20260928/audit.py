#!/usr/bin/env python3
"""Audit immutable Chunk experiment records and write an SHA-256 inventory.

The currently running 1K server.log is append-only/live; its fixed snapshot is
inventoried instead. The inventory never includes itself.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def main(root):
    data = json.loads((root / 'dataset.json').read_text())
    report = {'dataset_seed': data['seed'], 'groups': {}, 'ok': True}
    for size in (8192, 1024):
        folder = root / f'node_23/chunk_{size}/bench'
        plan = json.loads((folder / 'plan.json').read_text())
        dispatches = json.loads((folder / 'dispatches.json').read_text())
        warmup = json.loads((folder / 'warmup/result.json').read_text())
        flush = json.loads((folder / 'flush_cache.json').read_text())
        assert flush['status'] == 200 and warmup['output_tokens'] == 1
        assert warmup['prompt_tokens'] == data['prefix_tokens']
        assert len(dispatches) == 4 and [e['name'] for e in dispatches] == list('ABCD')
        assert all(e['submitted'] for e in dispatches)
        group = {'cache_flush_http_status': flush['status'], 'warmup':
            {k: warmup[k] for k in ('http_status', 'prompt_tokens', 'cached_tokens', 'output_tokens')},
            'dispatch_span_seconds': (dispatches[-1]['dispatch_perf_ns']-
                                     dispatches[0]['dispatch_perf_ns'])/1e9,
            'requests': {}}
        previous = None
        for name in 'ABCD':
            path = folder / 'requests' / name
            request = json.loads((path / 'request.json').read_text())
            r = json.loads((path / 'result.json').read_text())
            integrity = json.loads((path / 'output_integrity.json').read_text())
            final_ids = json.loads((path / 'output_ids_final.json').read_text())
            input_ids = request['payload']['input_ids']
            counts = []
            last_ids = None
            done = 0
            with gzip.open(path / 'sse_raw.jsonl.gz', 'rt', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    raw = entry['raw_line']
                    if not raw.startswith('data:'):
                        continue
                    if raw[5:].strip() == '[DONE]':
                        done += 1
                        continue
                    event = json.loads(raw[5:])
                    counts.append(event['meta_info']['completion_tokens'])
                    last_ids = event['output_ids']
                    assert isinstance(entry['recv_perf_ns'], int)
                    assert isinstance(entry['recv_wall_ns'], int)
            assert input_ids == data['cases'][name]['input_ids']
            assert input_ids[:data['prefix_tokens']] == data['prefix_ids']
            assert digest(input_ids) == plan['input_sha256'][name] == r['input_sha256']
            assert len(input_ids) == r['prompt_tokens'] == 32768
            assert r['cached_tokens'] == 19584
            assert (r['http_status'],r['output_tokens'],r['event_count'],r['error']) == (200,300,300,None)
            assert r['meta_info']['num_retractions'] == 0
            assert counts == list(range(1,301)) and done == 1
            assert last_ids == final_ids and len(final_ids) == 300
            assert digest(final_ids) == integrity['actual_output_sha256']
            assert r['first_token_perf_ns'] < r['ended_perf_ns']
            assert previous is None or dispatches['ABCD'.index(name)]['dispatch_perf_ns'] > previous
            previous = dispatches['ABCD'.index(name)]['dispatch_perf_ns']
            group['requests'][name] = {'http_status':r['http_status'],'event_count':len(counts),
                'done_count':done,'prefix_same':True,'input_digest':r['input_sha256'],
                'final_output_digest':digest(final_ids),'prompt_tokens':len(input_ids),
                'cached_tokens':r['cached_tokens'],'final_output_tokens':len(final_ids),
                'num_retractions':r['meta_info']['num_retractions']}
        last = max(json.loads((folder / 'requests' / n / 'result.json').read_text())['ended_perf_ns'] for n in 'ABCD')
        group['a_dispatch_to_group_finish_seconds']=(last-dispatches[0]['dispatch_perf_ns'])/1e9
        assert group['dispatch_span_seconds'] < plan['time_limit_s']
        group['metrics_samples'] = sum(1 for _ in gzip.open(folder/'metrics_raw.jsonl.gz','rt'))
        group['semantic_checks'] = json.loads((root / f'node_23/chunk_{size}/semantic_checks/review.json').read_text())
        assert len(group['semantic_checks'])==2 and all(x['review']=='pass' and x['http_status']==200 for x in group['semantic_checks'])
        report['groups'][str(size)] = group
    (root/'analysis/integrity_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    sources = root.parent.parent
    manifest['postprocessing_code_sha256']['audit.py'] = hashlib.sha256((sources/'audit.py').read_bytes()).hexdigest()
    manifest['manifest_updated_machine_local'] = __import__('datetime').datetime.now().astimezone().isoformat()
    (root/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with (root/'file_inventory.sha256.tsv').open('w') as file:
        file.write('sha256\tbytes\trelative_path\n')
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in ('file_inventory.sha256.tsv', 'node_23/chunk_1024/server.log'):
                continue
            h = hashlib.sha256()
            with path.open('rb') as item:
                for block in iter(lambda:item.read(1<<20),b''):
                    h.update(block)
            file.write(f'{h.hexdigest()}\t{path.stat().st_size}\t{rel}\n')
    print(f'audited {sum(len(g["requests"]) for g in report["groups"].values())} requests, saved inventory')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
