#!/usr/bin/env python3
"""Derive actual generated token sequence from preserved cumulative SSE output_ids."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('bench',type=Path)
a=ap.parse_args()
for name in 'ABCD':
    path=a.bench/'requests'/name
    last=None
    frames=0
    with gzip.open(path/'sse_raw.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            raw=json.loads(line).get('raw_line','')
            if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':
                continue
            obj=json.loads(raw[5:])
            if isinstance(obj.get('output_ids'),list):
                last=obj['output_ids']
                frames+=1
    if last is None: raise ValueError(f'{name}: no output_ids in saved SSE')
    r=json.loads((path/'result.json').read_text())
    if len(last)!=r['output_tokens']:
        raise ValueError(f'{name}: last cumulative output_ids len={len(last)} vs reported={r["output_tokens"]}')
    (path/'output_ids_final.json').write_text(json.dumps(last,ensure_ascii=False)+'\n')
    digest=hashlib.sha256(json.dumps(last,separators=(',',':')).encode()).hexdigest()
    (path/'output_integrity.json').write_text(json.dumps({'derived_from':'last SSE output_ids',
        'sse_frames_with_output_ids':frames,'output_tokens':len(last),'actual_output_sha256':digest,
        'capture_output_ids_observed_appended_all_frames':r['output_ids_observed'],
        'warning':'output_ids.json appends cumulative output_ids for every SSE frame; use output_ids_final.json'},indent=2)+'\n')
    print(name,len(last),digest)
