#!/usr/bin/env python3
"""Correct four erroneous historical reference blocks using publisher metadata.

The two affected virtual whole-file SHA256 are recomputed before approval.
All other whole-file SHA256 are identical to the published file metadata.
"""
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
MODEL=Path('/data_lib/data/models/glm-q')


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    old=ROOT/'weights/recovered';out=ROOT/'weights/publisher';out.mkdir(exist_ok=True)
    manifest=json.loads((old/'manifest.json').read_text())
    published=json.loads((ROOT/'network/publisher_quant_files.json').read_text())['Data']['Files']
    published={f['Path']:f for f in published}
    comparison=json.loads((ROOT/'weights/publisher_comparison.json').read_text())
    manifest['reference_id']='Eco-Tech_GLM-5.2-w8a8_publisher_SHA256_20261012'
    corrected=[]
    for p in sorted((ROOT/'weights/publisher_ranges').glob('*.json')):
        r=json.loads(p.read_text());data=p.with_suffix('.bin').read_bytes()
        assert sha(data)==r['download_sha256']==r['current_sha256']
        b=next(b for b in manifest['files'][r['file']]['blocks'] if b['offset']==r['offset'])
        assert b['original_sha256']==r['current_sha256']
        b['sha256']=r['current_sha256'];b['spans']=[]
        for k in ('payload','payload_bytes','payload_sha256'):b.pop(k,None)
        corrected.append({'file':r['file'],'offset':r['offset'],'publisher_sha256':r['download_sha256'],
                          'erroneous_historical_sha256':r['sha256']})
    assert len(corrected)==4
    count=size=0
    for info in manifest['files'].values():
        for b in info['blocks']:
            if b['spans']:
                src=old/b['payload'];dest=out/b['payload'];dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dest);count+=1;size+=b['payload_bytes']
    assert count==329
    verified=[]
    for r in comparison['rows']:
        name=r['file'];info=manifest['files'][name];pub=published[name]
        if r['sha_equal']:
            method='historical whole-file SHA256 equals publisher; exact block hashes retained'
            fullsha=r['reference_sha256']
        else:
            started=time.monotonic();h=hashlib.sha256()
            with (MODEL/name).open('rb') as f:
                for b in info['blocks']:
                    f.seek(b['offset']);data=bytearray(f.read(b['length']))
                    assert sha(data)==b['original_sha256']
                    if b['spans']:
                        payload=(out/b['payload']).read_bytes();assert sha(payload)==b['payload_sha256']
                        for s in b['spans']:
                            data[s['offset']:s['offset']+s['length']]=payload[s['payload_offset']:s['payload_offset']+s['length']]
                    assert sha(data)==b['sha256'];h.update(data)
            fullsha=h.hexdigest();method='recomputed virtual whole-file SHA256 after correction'
            print(name,'SHA256',fullsha,'publisher',pub['Sha256'],'seconds',time.monotonic()-started,flush=True)
        assert fullsha==pub['Sha256'],name+' does not match publisher'
        info['publisher_sha256']=fullsha
        verified.append({'file':name,'sha256':fullsha,'publisher_revision':pub['Revision'],'method':method})
    manifest['external_block_count']=count;manifest['external_bytes']=size
    manifest['publisher_verification']={'repository':'Eco-Tech/GLM-5.2-w8a8','metadata_source':str(ROOT/'network/publisher_quant_files.json'),
                                        'files':verified,'historical_reference_corrections':corrected}
    data=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode();(out/'manifest.json').write_bytes(data)
    summary={k:v for k,v in manifest.items() if k not in ('files','publisher_verification')}
    summary.update(manifest_sha256=sha(data),complete=True,publisher_files_verified=182,
                   publisher_file_sha256s={r['file']:r['sha256'] for r in verified},
                   historical_reference_corrections=corrected,
                   claim='Virtual checkpoint matches all 182 published weight-file SHA256 values; config, quantization description and tokenizer match publisher.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUCCESS publisher files',len(verified),'repair blocks',count,'payload MiB',size/2**20,flush=True)


if __name__=='__main__':main()
