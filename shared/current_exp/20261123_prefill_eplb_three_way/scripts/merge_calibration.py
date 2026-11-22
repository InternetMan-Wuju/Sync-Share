#!/usr/bin/env python3
import glob,json,os
from pathlib import Path
root=Path(os.environ.get('OBS',Path(__file__).resolve().parents[1]/'logs/off_off_observe_all_v4/observer'))
out=Path(__file__).resolve().parents[1]/'artifacts/calibration/layer49_8k_all_ranks.jsonl'
rows=[]
for p in sorted(glob.glob(str(root/'routes_rank*.jsonl'))):
 sel=[json.loads(x) for x in Path(p).read_text().splitlines() if json.loads(x).get('seq')==0]
 assert len(sel)==1,(p,len(sel));rows.append(sel[0])
assert len(rows)==16,len(rows)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
print(json.dumps({'ranks':len(rows),'rows':sum(r['local_rows'] for r in rows),'tasks':sum(r['physical_task_count'] for r in rows),'out':str(out)},indent=2))
