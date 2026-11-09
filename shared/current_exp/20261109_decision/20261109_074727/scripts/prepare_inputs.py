"""Crop AgentX hash blocks to 8K, materialize deterministic synthetic IDs."""
import hashlib,json,random
from pathlib import Path
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'local').mkdir(parents=True,exist_ok=True)
SRC=Path('/data_lib/ustc/dqt/inferencex-container/cache/huggingface/datasets/semianalysisai___cc-traces-weka-062126-256k/default/0.0.0/8fecd2fc56694469f758f0afbbb6335ad3043740/cc-traces-weka-062126-256k-train-00000-of-00002.arrow')
cases=[]
with pa.memory_map(str(SRC),'r') as f:
    for batch in pa.ipc.open_stream(f):
        for trace in batch.to_pylist():
            for ix,entry in enumerate(trace['requests']):
                r=json.loads(entry) if isinstance(entry,str) else entry
                if r.get('in',0)<8192 or r.get('out',0)<32 or len(r.get('hash_ids',[]))<128:continue
                ids=[]
                for block in r['hash_ids'][:128]:
                    seed=hashlib.sha256(f'cyt-ep16-20261109:{trace["id"]}:{block}'.encode()).digest()
                    rng=random.Random(int.from_bytes(seed,'big'));ids.extend(rng.randrange(1000,150000) for _ in range(64))
                cases.append({'case_id':len(cases),'trace_id':trace['id'],'request_index':ix,'original_input_tokens':r['in'],'original_output_tokens':r['out'],'block_ids':r['hash_ids'][:128],'input_tokens':8192,'output_tokens':32,'input_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'input_ids':ids})
                break
            if len(cases)==3:break
        if len(cases)==3:break
assert len(cases)==3
meta={'source':str(SRC),'dataset':'semianalysisai/cc-traces-weka-062126-256k','revision':'8fecd2fc56694469f758f0afbbb6335ad3043740','identity':'AgentX length/hash-block driven synthetic token materialization; NOT original AgentX text','seed_rule':'SHA256(cyt-ep16-20261109:trace_id:block_id), Python Random randrange(1000,150000), 64 IDs/block','prefix_cache':False,'cases':cases}
(ROOT/'local/inputs.json').write_text(json.dumps(meta))
meta['cases']=[{k:v for k,v in c.items() if k not in ('input_ids','block_ids')} for c in cases]
(ROOT/'INPUTS.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
