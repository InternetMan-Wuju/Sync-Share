#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path('/home/chengyetai/workspace/Share/shared/current_exp/20261116_compare_eplb')
SRC=Path('/data_lib/wzc/data/sharegpt.json')
tok=Tokenizer.from_file('/data_lib/data/models/glm-q/tokenizer.json')
TARGET=3072
INSTRUCTION='\n\n请基于以上上下文，用中文简要说明这段内容主要在讨论什么，并给出一个结论。'
data=json.loads(SRC.read_text())
items=[]
for conv in data:
    for msg in conv.get('conversations',[]):
        if msg.get('from')=='human' and len(msg.get('value',''))>=80:
            items.append((conv.get('id'),msg['value'].strip()))
ins_ids=tok.encode(INSTRUCTION).ids
limit=TARGET-len(ins_ids)-8
ids=[]; refs=[]
for cid,text in items:
    x=tok.encode(text).ids
    if not x: continue
    take=x[:max(0,limit-len(ids))]
    ids.extend(take); refs.append({'conversation_id':cid,'chars':len(text),'tokens_used':len(take)})
    if len(ids)>=limit: break
ids=ids[:limit]
text=tok.decode(ids)+INSTRUCTION
final=tok.encode(text).ids
out=ROOT/'prompts/prompt_3k.txt'; out.write_text(text)
meta={'target_tokens':TARGET,'actual_tokens':len(final),'file':str(out),'sha256':hashlib.sha256(text.encode()).hexdigest(),'source':str(SRC),'human_only':True,'references':refs,'instruction':INSTRUCTION}
(ROOT/'prompts/prompt_3k.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(meta,ensure_ascii=False,indent=2))
