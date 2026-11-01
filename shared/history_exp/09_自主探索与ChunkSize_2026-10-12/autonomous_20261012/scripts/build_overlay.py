#!/usr/bin/env python3
"""Assemble a complete, fail-closed read-only reference overlay manifest."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

WORK=Path('/home/chengyetai/workspace')
ROOT=WORK/'current_exp/autonomous_20261012'
AUDIT=WORK/'history_exp/06_四节点服务与权重核查_2026-10-11/weight_audit'
MODEL=Path('/data_lib/data/models/glm-q')
OUT=ROOT/'weights/recovered'


def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    status=read(OUT/'recovery_status.json')
    if status['done']!=333 or status['recovered']!=333:raise RuntimeError('incomplete recovery')
    recovered={(r['file'],r['offset']):r for r in status['rows']}
    files={};count=size=0;attention=[]
    for path in sorted((AUDIT/'snapshot_datalib').glob('*.safetensors.json')):
        original=read(path);reference=read(AUDIT/'audit_seed'/path.name)
        name=original['file'];stat=(MODEL/name).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(original['bytes'],original['mtime_ns']):
            raise RuntimeError('base identity changed: '+name)
        if original['tensors']!=reference['tensors'] or original['header_sha256']!=reference['header_sha256']:
            raise RuntimeError('incompatible reference layout')
        refs={b['offset']:b for b in reference['blocks']};blocks=[]
        for b in original['blocks']:
            expected=refs[b['offset']]
            block={'offset':b['offset'],'length':b['length'],'original_sha256':b['sha256'],
                   'sha256':expected['sha256'],'spans':[]}
            if b['sha256']!=expected['sha256']:
                r=recovered[(name,b['offset'])]
                payload=(OUT/r['payload']).read_bytes()
                assert len(payload)==r['payload_bytes'] and sha(payload)==r['payload_sha256']
                with (MODEL/name).open('rb') as f:f.seek(b['offset']);base=bytearray(f.read(b['length']))
                assert sha(base)==b['sha256']
                for s in r['spans']:
                    base[s['offset']:s['offset']+s['length']]=payload[s['payload_offset']:s['payload_offset']+s['length']]
                assert sha(base)==expected['sha256']
                block.update({k:r[k] for k in ('payload','payload_bytes','payload_sha256','spans')})
                count+=1;size+=len(payload)
                if (name,b['offset']) in [(f'quant_model_weights-{n:05d}-of-00181.safetensors',o) for n,o in [(1,67108864),(47,0),(89,0)]]:
                    for tensor,meta in original['tensors'].items():
                        if meta.get('dtype')!='BF16' or 'self_attn' not in tensor:continue
                        begin,end=[v+original['header_bytes'] for v in meta['data_offsets']]
                        left=max(begin,b['offset']);right=min(end,b['offset']+b['length'])
                        if left>=right:continue
                        u=np.frombuffer(base[left-b['offset']:right-b['offset']],dtype=np.uint16)
                        f32=(u.astype(np.uint32)<<16).view(np.float32)
                        attention.append({'tensor':tensor,'file':name,'offset':b['offset'],
                                          'nonfinite':int(np.sum(~np.isfinite(f32))),
                                          'abs_gt_1e10':int(np.sum(np.isfinite(f32)&(np.abs(f32)>1e10)))})
            blocks.append(block)
        files[name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'header_bytes':original['header_bytes'],
                     'header_sha256':original['header_sha256'],'tensors':original['tensors'],'blocks':blocks}
    assert len(files)==182 and count==333
    assert all(x['nonfinite']==0 and x['abs_gt_1e10']==0 for x in attention)
    manifest={'format':'verified_compact_pages_v1','verification':'each_reconstructed_block_matches_reference_sha256',
              'reference_id':'node31_glm-qq_snapshot_node31_20260927_recovered_20261012',
              'reference_snapshot_summary':read(AUDIT/'audit_seed/summary.json'),
              'original_snapshot_summary':read(AUDIT/'snapshot_datalib/summary.json'),
              'total_block_count':sum(len(f['blocks']) for f in files.values()),
              'external_block_count':count,'external_bytes':size,'files':files}
    data=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode()
    (OUT/'manifest.json').write_bytes(data)
    summary={k:v for k,v in manifest.items() if k!='files'}
    summary.update(manifest_sha256=sha(data),attention_repair_checks=attention,
                   complete=True,claim='Exact historical reference bytes; not independently certified against publisher checkpoint.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
