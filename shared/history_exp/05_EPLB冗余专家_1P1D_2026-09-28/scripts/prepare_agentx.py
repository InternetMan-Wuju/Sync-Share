#!/usr/bin/env python3
"""Select original AgentX trace requests and crop each to 16384 input / 512 output.

The source contains counts and 64-token content-hash IDs, NOT original text.
Token block materialization is deterministic and scoped by (trace id, block id).
"""
import collections
import datetime
import hashlib
import json
import statistics
from pathlib import Path
import argparse

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=4096);args=p.parse_args()
assert args.limit>=128 and args.limit%32==0
source_hash=hashlib.sha256();by_session={};seen=collections.Counter()
with args.source.open('rb') as f:
    for line in f:
        source_hash.update(line)
        s=json.loads(line);tid=s['id'];last_t=-1
        entries=[]
        for ix,r in enumerate(s.get('requests',[])):
            seen['total']+=1
            if r.get('type') not in ('s','n') or r.get('in',0)<16384 or r.get('out',0)<512:continue
            hashes=r.get('hash_ids') or []
            if len(hashes)<256:continue
            seen['eligible']+=1
            # Keep original relative timestamp and source counters for audit.
            entries.append({'trace_id':tid,'trace_request_index':ix,'t':r.get('t'),
                'source_input_tokens':r['in'],'source_output_tokens':r['out'],
                'block_ids':hashes[:256],'input_tokens':16384,'output_tokens':512,
                'block_size':64,'type':r['type'],'model':r.get('model')})
        if entries:by_session[tid]=entries
# Preserve local chronology and contiguous content within each session. Rotate
# sessions in eight-request segments so a single session cannot dominate all
# 30-minute runs, but adjacent requests still share real trace block hashes.
ids=sorted(by_session,key=lambda k:hashlib.sha256(('agentx-20260928:'+k).encode()).digest())
selected=[];segment=8;round_no=0
while len(selected)<args.limit:
    progress=False
    for tid in ids:
        start=round_no*segment
        if start>=len(by_session[tid]):continue
        progress=True
        selected.extend(by_session[tid][start:start+min(segment,args.limit-len(selected))])
        if len(selected)>=args.limit:break
    if not progress:break
    round_no+=1
if len(selected)<args.limit:raise RuntimeError('insufficient qualifying original trace rows')
# Sequence is identical for A/B/C; field t is preserved as trace metadata.
for i,row in enumerate(selected):row['sequence_index']=i
prefix=[]
for i in range(1,len(selected)):
    a,b=selected[i-1],selected[i]
    if a['trace_id']!=b['trace_id']:continue
    prefix.append(next((j for j,(x,y) in enumerate(zip(a['block_ids'],b['block_ids'])) if x!=y),256)*64/16384)
payload={'created_at_local':datetime.datetime.now().astimezone().isoformat(),
 'source_path':str(args.source),'source_sha256':source_hash.hexdigest(),'source_bytes':args.source.stat().st_size,
 'sampling':'from top-level type s/n only, original in>=16384 and out>=512, first 256 original 64-token content-hash blocks, eight contiguous requests per trace in deterministic session order; 4096 requests; no original text',
 'token_materialization':'deterministic token IDs seeded from SHA256(trace_id,block_id), each 64-token block generated once per request; distinct trace scopes; random IDs not original content',
 'seed_text':'agentx-20260928-block64-v1','fixed_input_tokens':16384,'fixed_output_tokens':512,
 'trace_request_count':len(selected),'eligible_source_requests':seen['eligible'],
 'distinct_trace_ids':len(set(r['trace_id'] for r in selected)),
 'adjacent_within_trace_shared_prefix_fraction_mean':statistics.mean(prefix) if prefix else None,
 'adjacent_within_trace_prefix_pair_count':len(prefix),'requests':selected}
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in payload.items() if k!='requests'},ensure_ascii=False,indent=2))
