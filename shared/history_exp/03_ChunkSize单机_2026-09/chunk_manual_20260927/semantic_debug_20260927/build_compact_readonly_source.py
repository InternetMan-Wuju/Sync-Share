#!/usr/bin/env python3
"""Save literal reference bytes for differing pages; never write either model."""
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
parser.add_argument('--reference', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--model')
parser.add_argument('--reference-stream', action='store_true')
parser.add_argument('--accept-current-base', action='store_true')
parser.add_argument('--reference-host', default='10.120.72.31')
args = parser.parse_args()
base, reference, out = map(Path, (args.base, args.reference, args.out))
out.mkdir(parents=True, exist_ok=True)
base_summary = json.loads((base / 'summary.json').read_text())
ref_summary = json.loads((reference / 'summary.json').read_text())
if base_summary['file_count'] != 182 or ref_summary['file_count'] != 182:
    raise RuntimeError('incomplete snapshot')
if ref_summary['streamed_nonfinite'] or ref_summary['streamed_abs_gt_1e10']:
    raise RuntimeError('reference snapshot contains abnormal floating values')
model = Path(args.model or base_summary['model'])
if not model.is_dir():
    raise RuntimeError('run on the node owning the base snapshot: ' + str(model))
files, fetch = {}, []
for path in sorted(base.glob('*.safetensors.json')):
    original = json.loads(path.read_text())
    ref = json.loads((reference / path.name).read_text())
    for key in ('bytes', 'header_bytes', 'header_sha256'):
        if original[key] != ref[key]:
            raise RuntimeError('different layouts: ' + original['file'])
    if len(original['blocks']) != len(ref['blocks']):
        raise RuntimeError('different block count')
    entry = {k: original[k] for k in ('file', 'bytes', 'mtime_ns', 'inode', 'header_bytes', 'header_sha256', 'tensors')}
    entry.update(original_sha256=original['sha256'], reference_sha256=ref['sha256'], blocks=[])
    for local_block, ref_block in zip(original['blocks'], ref['blocks']):
        if (local_block['offset'], local_block['length']) != (ref_block['offset'], ref_block['length']):
            raise RuntimeError('different block layout')
        block = dict(ref_block, original_sha256=local_block['sha256'], spans=[])
        entry['blocks'].append(block)
        if local_block['sha256'] != ref_block['sha256']:
            fetch.append(dict(ref_block, file=original['file'], reference_mtime_ns=ref['mtime_ns'], reference_inode=ref['inode']))
    files[original['file']] = entry

program = '''import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin)
root=Path('/mnt/raid/user/data/models/glm-qq')
for block in request:
    path=root/block['file']; stat=path.stat()
    if stat.st_mtime_ns!=block['reference_mtime_ns'] or stat.st_ino!=block['reference_inode']:
        raise RuntimeError('reference identity changed: '+str(path))
    with path.open('rb') as f:
        f.seek(block['offset']); data=f.read(block['length'])
    if len(data)!=block['length'] or hashlib.sha256(data).hexdigest()!=block['sha256']:
        raise RuntimeError('reference SHA changed: '+str(path)+' '+str(block['offset']))
    sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
'''
request = out / 'fetch_request.json'
request.write_text(json.dumps(fetch))
print(json.dumps({'fetch_blocks': len(fetch), 'transfer_gib': sum(b['length'] for b in fetch) / 2**30}), flush=True)
start = time.monotonic()
external_bytes = different_bytes = spans_total = base_digest_drift = 0
with request.open('rb') as stdin, (out / 'fetch_stderr.log').open('w') as stderr:
    proc = None if args.reference_stream else subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'LogLevel=ERROR', args.reference_host, shlex.join(['python3', '-c', program])], stdin=stdin, stdout=subprocess.PIPE, stderr=stderr)
    stream = sys.stdin.buffer if args.reference_stream else proc.stdout
    try:
        for index, item in enumerate(fetch, 1):
            parts, remaining = [], item['length']
            while remaining:
                data = stream.read(min(remaining, 4*1024*1024))
                if not data:
                    raise RuntimeError('incomplete reference transfer; see fetch_stderr.log')
                parts.append(data)
                remaining -= len(data)
            reference_bytes = b''.join(parts)
            if hashlib.sha256(reference_bytes).hexdigest() != item['sha256']:
                raise RuntimeError('reference transfer SHA mismatch')
            entry = files[item['file']]
            path = model / item['file']
            stat = path.stat()
            if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
                raise RuntimeError('base file identity changed')
            with path.open('rb') as local_reader:
                local_reader.seek(item['offset'])
                original_bytes = local_reader.read(item['length'])
            block = next(b for b in entry['blocks'] if b['offset'] == item['offset'])
            current_sha = hashlib.sha256(original_bytes).hexdigest()
            if current_sha != block['original_sha256']:
                if not args.accept_current_base:
                    raise RuntimeError('base block SHA changed: ' + str(path) + ' ' + str(item['offset']))
                block['snapshot_original_sha256'] = block['original_sha256']
                block['original_sha256'] = current_sha
                base_digest_drift += 1
            mask = np.frombuffer(original_bytes, dtype=np.uint8) != np.frombuffer(reference_bytes, dtype=np.uint8)
            different_bytes += int(np.count_nonzero(mask))
            full_pages = len(mask) // 4096
            page_mask = mask[:full_pages*4096].reshape(full_pages, 4096).any(axis=1)
            if len(mask) % 4096:
                page_mask = np.append(page_mask, bool(mask[full_pages*4096:].any()))
            pages = np.flatnonzero(page_mask)
            breaks = np.flatnonzero(np.diff(pages) != 1) + 1
            groups = np.split(pages, breaks)
            payload = bytearray()
            reconstructed = bytearray(original_bytes)
            for group in groups:
                if not len(group):
                    continue
                left, right = int(group[0])*4096, min((int(group[-1])+1)*4096, len(mask))
                piece = reference_bytes[left:right]
                block['spans'].append({'offset': left, 'length': len(piece), 'payload_offset': len(payload)})
                payload.extend(piece)
                reconstructed[left:right] = piece
            if hashlib.sha256(reconstructed).hexdigest() != block['sha256']:
                raise RuntimeError('reconstructed block does not match reference')
            relative = f"payloads/{item['file']}/{item['offset']:012d}.bin"
            block.update(payload=relative, payload_sha256=hashlib.sha256(payload).hexdigest(), payload_bytes=len(payload))
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.part')
            temp.write_bytes(payload)
            temp.replace(target)
            external_bytes += len(payload)
            spans_total += len(block['spans'])
            if index % 20 == 0 or index == len(fetch):
                print(json.dumps({'received_blocks': index, 'total': len(fetch), 'stored_mib': round(external_bytes / 2**20, 2), 'elapsed_s': round(time.monotonic()-start, 1)}), flush=True)
        if stream.read(1):
            raise RuntimeError('unexpected trailing data')
        if proc is not None and proc.wait() != 0:
            raise RuntimeError('reference process failed')
    except BaseException:
        if proc is not None:
            proc.kill()
            proc.wait()
        raise
manifest = {'created_at': datetime.datetime.now().astimezone().isoformat(), 'format': 'verified_compact_pages_v1', 'verification': 'each_reconstructed_block_matches_reference_sha256', 'reference_id': 'node31_glm-qq_' + reference.name, 'reference_snapshot_summary': ref_summary, 'original_snapshot_summary': base_summary, 'total_block_count': sum(len(f['blocks']) for f in files.values()), 'external_block_count': len(fetch), 'external_span_count': spans_total, 'external_bytes': external_bytes, 'different_bytes': different_bytes, 'base_digest_drift_during_build': base_digest_drift, 'files': files}
(out / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':'))+'\n')
summary = {k: v for k, v in manifest.items() if k != 'files'}
summary.update(manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(), elapsed_s=time.monotonic()-start)
(out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
