#!/usr/bin/env python3
"""Apply the actual GLM chat template to saved, untruncated ShareGPT users."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from sglang.benchmark.utils import get_tokenizer

data=json.loads(Path('/study/real_chat_source.json').read_text())
tokenizer=get_tokenizer('/models/glm-q')
for c in data['cases']:
    ids=tokenizer.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if isinstance(ids,Mapping):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    c['input_ids']=list(ids)
    c['input_sha256']=hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
    c['input_tokens']=len(ids)
data.update(template='actual model apply_chat_template, enable_thinking=False; no truncation or role concatenation',
            tokenizer_sha256=hashlib.sha256(Path('/models/glm-q/tokenizer.json').read_bytes()).hexdigest())
Path('/exp-logs/real_chat_dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('real chat input lengths',[c['input_tokens'] for c in data['cases']],flush=True)
