#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path('/home/chengyetai/workspace/Share/shared/current_exp/20261116_compare_eplb')
SRC=Path('/data_lib/wzc/data/sharegpt.json')
MODEL=Path('/data_lib/data/models/glm-q')
TARGETS=(8192,16384,32768)
INSTRUCTION='\n\n请基于以上上下文，用中文简要说明这段内容主要在讨论什么，并给出一个结论。'
tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
data=json.loads(SRC.read_text())
# Use distinct human turns in order across the three prompt tiers.
items=[]
for conv in data:
    for msg in conv.get('conversations',[]):
        if msg.get('from')=='human':
            text=msg.get('value','').strip()
            if len(text)>=80:
                items.append((conv.get('id'),text))
print('human items',len(items),flush=True)
pos=0
for target in TARGETS:
    ins_ids=tok.encode(INSTRUCTION).ids
    limit=target-len(ins_ids)-8
    ids=[]; refs=[]
    while len(ids)<limit and pos<len(items):
        cid,text=items[pos]; pos+=1
        x=tok.encode(text).ids
        if not x: continue
        take=x[:max(0,limit-len(ids))]
        ids.extend(take); refs.append({'conversation_id':cid,'chars':len(text),'tokens_used':len(take)})
        if len(take)<len(x): break
    ids=ids[:limit]
    context=tok.decode(ids)
    text=context+INSTRUCTION
    final_ids=tok.encode(text).ids
    out=ROOT/'prompts'/f'prompt_{target//1024}k.txt'
    out.write_text(text)
    meta={'target_tokens':target,'actual_tokens':len(final_ids),'file':str(out),'sha256':hashlib.sha256(text.encode()).hexdigest(),'source':'/data_lib/wzc/data/sharegpt.json','human_only':True,'references':refs,'instruction':INSTRUCTION}
    (ROOT/'prompts'/f'prompt_{target//1024}k.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(target,'actual',len(final_ids),'chars',len(text),'refs',len(refs),flush=True)
