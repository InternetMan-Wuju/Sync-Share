#!/usr/bin/env python3
import argparse,hashlib,json,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
d=json.loads(a.source.read_text());rows=d['requests'];a.outdir.mkdir(parents=True,exist_ok=True);cache={}
def block_ids(trace,blocks):
 out=[]
 for b in blocks:
  key=(trace,str(b))
  if key not in cache:
   dig=hashlib.sha256(('agentx-20260928-block64-v1\0'+str(trace)+'\0'+str(b)).encode()).digest();rng=random.Random(int.from_bytes(dig,'big'));cache[key]=[rng.randrange(1,154880) for _ in range(64)]
  out.extend(cache[key])
 return out
# first row, first 128 / all 256 blocks
r8=rows[0];r16=rows[1]
# find adjacent same-trace pair for 32K
pair=None
for i in range(len(rows)-1):
 if rows[i]['trace_id']==rows[i+1]['trace_id'] and len(rows[i]['block_ids'])>=256 and len(rows[i+1]['block_ids'])>=256:
  pair=(rows[i],rows[i+1]);break
assert pair
specs={'8192':block_ids(r8['trace_id'],r8['block_ids'][:128]),'16384':block_ids(r16['trace_id'],r16['block_ids'][:256]),'32768':block_ids(pair[0]['trace_id'],pair[0]['block_ids'][:256]+pair[1]['block_ids'][:256])}
for size,ids in specs.items():
 assert len(ids)==int(size)
 meta={'input_tokens':len(ids),'input_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'source':str(a.source),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'materialization':'AgentX-derived deterministic token IDs from 64-token block hashes; not original text','target_size':int(size),'rows_used':1 if size!='32768' else 2}
 (a.outdir/f'agentx_{size}.json').write_text(json.dumps({**meta,'input_ids':ids})+'\n')
 (a.outdir/f'agentx_{size}.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(size,meta['input_sha256'])
