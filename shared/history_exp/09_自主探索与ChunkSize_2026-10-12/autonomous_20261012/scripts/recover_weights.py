#!/usr/bin/env python3
"""Recover audited reference pages, without writing any checkpoint or remote file.

Only blocks matching the September reference SHA256 may enter the overlay.
The incomplete manifest is diagnostic only; normal loading requires completeness.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

WORK = Path('/home/chengyetai/workspace')
ROOT = WORK/'current_exp/autonomous_20261012'
AUDIT = WORK/'history_exp/06_四节点服务与权重核查_2026-10-11/weight_audit'
MODEL = Path('/data_lib/data/models/glm-q')
OLD = WORK/'history_exp/03_ChunkSize单机_2026-09/chunk_manual_20260927/patches'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def block_read(path, block):
    with path.open('rb') as f:
        f.seek(block['offset'])
        return f.read(block['length'])


def remote_read(node, folder, block):
    # The remote interpreter opens only rb and writes matching bytes to stdout.
    path = str(Path(folder)/block['file'])
    script = ('import hashlib,sys\n'
              f'p={path!r}\n'
              'try:\n'
              ' with open(p,"rb") as f:\n'
              f'  f.seek({block["offset"]}); b=f.read({block["length"]})\n'
              f' if hashlib.sha256(b).hexdigest()=={block["sha256"]!r}: sys.stdout.buffer.write(b)\n'
              'except OSError: pass\n')
    r = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=6',
                        '-o','LogLevel=ERROR',f'10.120.72.{node}','python3 -'],
                       input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
    return r.stdout


def compact(base, target):
    page=4096
    pad=(-len(base))%page
    a=np.frombuffer(base+b'\0'*pad,dtype=np.uint64).reshape(-1,page//8)
    b=np.frombuffer(target+b'\0'*pad,dtype=np.uint64).reshape(-1,page//8)
    changed=np.flatnonzero(np.any(a!=b,axis=1))
    groups=[]
    for i in changed:
        left=int(i)*page;right=min(left+page,len(base))
        if groups and groups[-1][1]==left:groups[-1][1]=right
        else:groups.append([left,right])
    spans=[];payload=bytearray()
    for left,right in groups:
        spans.append({'offset':left,'length':right-left,'payload_offset':len(payload)})
        payload.extend(target[left:right])
    return bytes(payload),spans


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--missing-first',action='store_true')
    a=ap.parse_args()
    out=ROOT/'weights/recovered';out.mkdir(parents=True,exist_ok=True)
    differences=json.loads((ROOT/'weights/differing_blocks.json').read_text())
    missing={(17,536870912),(17,603979776),(17,3724541952),(119,3959422976)}
    if a.missing_first:
        differences=[b for b in differences if (int(b['file'].split('-')[1]),b['offset']) in missing]
    old_sources=[]
    for name in ('readonly_source23','readonly_source25'):
        p=OLD/name/'manifest.json'
        if p.is_file():old_sources.append((p.parent,json.loads(p.read_text())['files']))
    records=[];started=time.monotonic()
    for i,block in enumerate(differences):
        base=block_read(MODEL/block['file'],block)
        if len(base)!=block['length']:raise RuntimeError('short base read')
        data=None;source=None;attempts=[]
        # Resume previously verified compact pages, always verifying reconstruction.
        record_path=out/(block['file']+f'.{block["offset"]}.json')
        if record_path.exists():
            previous=json.loads(record_path.read_text());candidate=bytearray(base)
            p=(out/previous['payload']).read_bytes()
            if digest(p)==previous['payload_sha256']:
                for s in previous['spans']:
                    candidate[s['offset']:s['offset']+s['length']]=p[s['payload_offset']:s['payload_offset']+s['length']]
                if digest(candidate)==block['sha256']:data=bytes(candidate);source=previous['source']
        saved=AUDIT/'readonly_source_data/payloads'/block['file']/f'{block["offset"]:012d}.bin'
        if data is None and saved.is_file():
            candidate=saved.read_bytes()
            if digest(candidate)==block['sha256']:data=candidate;source=str(saved)
        for folder,files in old_sources:
            if data is not None:break
            old=next((b for b in files[block['file']]['blocks'] if b['offset']==block['offset']),None)
            if old and old.get('spans'):
                p=(folder/old['payload']).read_bytes()
                if digest(p)!=old['payload_sha256']:raise RuntimeError('old payload corrupted')
                candidate=bytearray(base)
                for s in old['spans']:
                    candidate[s['offset']:s['offset']+s['length']]=p[s['payload_offset']:s['payload_offset']+s['length']]
                if digest(candidate)==block['sha256']:data=bytes(candidate);source=str(folder)
        for folder in ('/mnt/raid/user/data/models/glm-qq','/mnt/raid/user/data/models/glm-q'):
            if data is not None:break
            try:candidate=block_read(Path(folder)/block['file'],block)
            except OSError:continue
            attempts.append({'source':'23:'+folder,'sha256':digest(candidate)})
            if digest(candidate)==block['sha256']:data=candidate;source='23:'+folder
        remote=[(32,'/mnt/raid/user/data/models/glm-qq'),(31,'/mnt/raid/user/data/models/glm-qq'),
                (32,'/mnt/raid/user/data/models/glm-q'),(31,'/mnt/raid/user/data/models/glm-q'),
                (20,'/data_lib/data/models/glm-q')]
        for node,folder in remote:
            if data is not None:break
            try:candidate=remote_read(node,folder,block)
            except (OSError,subprocess.TimeoutExpired) as e:
                attempts.append({'source':str(node)+':'+folder,'error':str(e)});continue
            attempts.append({'source':str(node)+':'+folder,'matched':bool(candidate)})
            if len(candidate)==block['length'] and digest(candidate)==block['sha256']:
                data=candidate;source=str(node)+':'+folder
        record={**block,'current_read_sha256':digest(base),'source':source,'attempts':attempts,'recovered':data is not None}
        if data is not None:
            assert digest(data)==block['sha256']
            payload,spans=compact(base,data)
            rel=Path('payloads')/block['file']/f'{block["offset"]:012d}.bin'
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            record.update(payload=str(rel),payload_bytes=len(payload),payload_sha256=digest(payload),spans=spans)
            record_path.write_text(json.dumps(record,indent=2))
        records.append(record)
        (out/('missing_first_status.json' if a.missing_first else 'recovery_status.json')).write_text(json.dumps({
            'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'done':len(records),
            'total':len(differences),'recovered':sum(r['recovered'] for r in records),
            'elapsed_s':time.monotonic()-started,'rows':records},indent=2))
        print(f'{i+1}/{len(differences)} recovered={data is not None} {block["file"]} @{block["offset"]} source={source}',flush=True)


if __name__=='__main__':main()
