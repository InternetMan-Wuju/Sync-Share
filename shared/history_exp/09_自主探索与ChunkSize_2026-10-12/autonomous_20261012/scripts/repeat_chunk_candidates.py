#!/usr/bin/env python3
"""After a complete base grid, repeat baseline and throughput candidates.

Run only after the main supervisor has completed and released NPUs. All new
services copy the original flags, changing only chunk and container name.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK=ROOT.parent/'Qchunksize'
STATE={'pid':os.getpid(),'phase':'initializing','completed':False,'started_unix':time.time(),'runs':[]}
ACTIVE=None


def status(phase,**extra):
    STATE.update(phase=phase,updated_unix=time.time(),**extra)
    p=ROOT/'chunk_repeat_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(STATE,indent=2));temp.replace(p)
    print(phase,extra,flush=True)


def stop():
    global ACTIVE
    if ACTIVE:
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/service.py'),'stop','--name',ACTIVE],check=True,timeout=60)
        ACTIVE=None
    for _ in range(12):
        d=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20)
        if d.returncode==0 and d.stdout.count('No running processes found')==8:break
        time.sleep(5)
    else:raise RuntimeError('NPUs not free; do not disturb existing users')
    time.sleep(15)


def main():
    global ACTIVE
    prior=json.loads((ROOT/'overnight_status.json').read_text())
    assert prior.get('completed') and prior['phase']=='planned_grid_completed',prior['phase']
    base=[]
    for p in (CHUNK/'results/autonomous_20261012').glob('*/result.json'):
        d=json.loads(p.read_text())
        if d.get('repeat',1)==1 and d['aggregate']['complete'] and d['aggregate']['cache_condition_valid']:
            base.append(d)
    assert len(base)==20,len(base)
    planned={}
    for scene,conc in sorted({(d['name'],d['concurrency']) for d in base}):
        candidates=sorted([d for d in base if (d['name'],d['concurrency'])==(scene,conc)],
                          key=lambda d:d['aggregate']['total_output_tok_s'],reverse=True)
        winner=candidates[0]
        chunks={16384,winner['chunk']}
        # A close second needs a repeat to avoid selecting noise as a winner.
        if candidates[1]['aggregate']['total_output_tok_s']>=.95*winner['aggregate']['total_output_tok_s']:
            chunks.add(candidates[1]['chunk'])
        for chunk in chunks:planned.setdefault(chunk,[]).append((scene,conc,winner['chunk']))
    status('planned',plan=planned)
    for chunk,cells in sorted(planned.items()):
        original=next(x for x in prior['runs'] if any(c['name'].endswith('_chunk'+str(chunk)+'_c'+str(cells[0][1]))
                                                     for c in x.get('cells',[])))
        cfg=json.loads((ROOT/'logs'/original['container']/'config.json').read_text())
        args=['--image',cfg['image'],'--mtp','off','--deepep',cfg['deepep'],'--mem',str(cfg['mem']),
              '--context',str(cfg['context']),'--mrr',str(cfg['mrr']),'--chunk',str(chunk),
              '--maxprefill',str(cfg['maxprefill']),'--dp',str(cfg['dp']),'--ep',str(cfg['ep'])]
        for switch,key in [('--fusion','fusion'),('--dp-attention','dp_attention'),('--repaired','repaired'),
                           ('--loader-v0516','loader_v0516'),('--deepep-fix','deepep_fix')]:
            if cfg.get(key):args.append(switch)
        if cfg['graph']:args+=['--graph','--graph-bs',cfg['graph_bs'],'--graph-max-bs',str(cfg['graph_max_bs'])]
        if cfg.get('dispatch_tokens'):args+=['--dispatch-tokens',str(cfg['dispatch_tokens'])]
        # config.env contains the final environment, not only explicit overrides.
        for key,value in cfg['env'].items():
            if key in ('HCCL_BUFFSIZE','HCCL_OP_EXPANSION_MODE','DEEP_NORMAL_MODE_USE_INT8_QUANT'):
                args+=['--env',key+'='+value]
        name=f'cyt-auto-chunk-repeat-{chunk}';ACTIVE=name
        status('starting',container=name)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/service.py'),'start','--name',name,*args],check=True,timeout=120)
        deadline=time.time()+3600
        while time.time()<deadline:
            p=subprocess.run(['sudo','-n','docker','inspect',name,'--format','{{json .State}}'],capture_output=True,text=True,timeout=20)
            if p.returncode or not json.loads(p.stdout)['Running']:raise RuntimeError('repeat service exited')
            logfile=ROOT/'logs'/name/'server.log'
            if logfile.exists():
                with logfile.open('rb') as f:f.seek(max(0,logfile.stat().st_size-262144));tail=f.read()
                if b'crashed with exit code' in tail or b'scheduler died during initialization' in tail:raise RuntimeError('repeat scheduler failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:30080/health',timeout=3):pass
                break
            except Exception:status('waiting_health',container=name);time.sleep(20)
        else:raise RuntimeError('repeat health timeout')
        for scene,conc,winner in cells:
            out=CHUNK/'results/autonomous_20261012'/f'{scene}_chunk{chunk}_c{conc}_repeat2'
            dataset=CHUNK/'datasets/native_20261012'/f'{scene}.json'
            command=[sys.executable,'-B',str(ROOT/'scripts/chunk_collect.py'),'--dataset',str(dataset),
                     '--out',str(out),'--concurrency',str(conc),'--chunk',str(chunk),'--repeat','2']
            status('running_cell',cell=out.name)
            with (ROOT/'logs'/(out.name+'.log')).open('w') as f:
                p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                started=time.time()
                while p.poll() is None:
                    status('running_cell',cell=out.name,child_pid=p.pid)
                    logfile=ROOT/'logs'/name/'server.log'
                    with logfile.open('rb') as stream:
                        stream.seek(max(0,logfile.stat().st_size-262144));tail=stream.read()
                    if b'crashed with exit code' in tail or b'scheduler died during initialization' in tail:
                        stop();p.wait(timeout=45)
                        raise RuntimeError('repeat scheduler crashed; failed client stream preserved')
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    if time.time()-started>7200:p.terminate();p.wait(timeout=30);raise RuntimeError('repeat cell timeout')
                if p.returncode:raise RuntimeError('repeat cell invalid; preserve data')
            STATE['runs'].append({'cell':out.name,'container':name,'result':str(out/'result.json')})
        stop()
    status('completed',completed=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('needs_conservative_review',error=repr(e))
        try:stop()
        except Exception as e:status('needs_conservative_review',stop_error=repr(e))
        raise
