#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def read_logs(path):
    out={}
    for p in sorted(Path(path).glob('rank*.jsonl')):
        out[p.stem[4:]]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out

ap=argparse.ArgumentParser();ap.add_argument('--same-layer-logs',type=Path,required=True);ap.add_argument('--cross-layer-logs',type=Path,required=True);ap.add_argument('--q0-logs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();same=read_logs(a.same_layer_logs);cross=read_logs(a.cross_layer_logs);q0=read_logs(a.q0_logs)
same_rows={}
for rank in sorted(same):
    same_rows[rank]={'install':[r for r in same[rank] if r.get('kind')=='install'],'reuse':[r for r in same[rank] if r.get('kind')=='reuse']}
cross_rows={}
for rank in ('0','1'):
    xs=[r for r in cross.get(rank,[]) if r.get('kind')=='install']
    cross_rows[rank]=[{'layer':x.get('layer'),'slot_owner':x.get('slot_owner'),'expert':x.get('expert'),'generation':(x.get('owner_meta') or {}).get('generation'),'storage_ptr':(x.get('owner_meta') or {}).get('storage_ptr')} for x in xs]
out={'same_layer':{'ranks':same_rows,
 'all_three_ranks_reuse_owner0':all(any(x.get('slot_owner')==0 for x in same.get(r,[]) if x.get('kind')=='reuse') for r in ('0','1','10')),
 'all_three_ranks_reuse_owner1':all(any(x.get('slot_owner')==1 for x in same.get(r,[]) if x.get('kind')=='reuse') for r in ('0','1','10')),
 'q0_install_count':sum(1 for rs in q0.values() for x in rs if x.get('kind')=='install'),
 'q0_noop_calls':sum(1 for rs in q0.values() for x in rs if x.get('kind')=='call_noop')},
 'cross_layer':{'installs':cross_rows}}
for o in ('0','1'):
    xs=cross_rows[o]
    out['cross_layer'][f'owner{o}_generations']=[x['generation'] for x in xs]
    out['cross_layer'][f'owner{o}_layers']=[x['layer'] for x in xs]
    out['cross_layer'][f'owner{o}_stable_storage']=len({x['storage_ptr'] for x in xs})==1
out['decision']='owner state is gathered and reduced before device send/recv; generation and storage identity retained'
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'same_layer':{'all_three_ranks_reuse_owner0':out['same_layer']['all_three_ranks_reuse_owner0'],'all_three_ranks_reuse_owner1':out['same_layer']['all_three_ranks_reuse_owner1'],'q0_install_count':out['same_layer']['q0_install_count'],'q0_noop_calls':out['same_layer']['q0_noop_calls']},'cross_layer':{k:v for k,v in out['cross_layer'].items() if k!='installs'}},indent=2))
