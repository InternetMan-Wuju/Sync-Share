#!/usr/bin/env python3
"""Summarize the C32 Chunk retest progress without touching the service."""
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parent
state = json.loads((root/'status.json').read_text())
rows = []
for cell in state['cells']:
    row = {k: cell.get(k) for k in ('id', 'valid', 'error')}
    path = Path(cell['out'])/'status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    progress = list(data.get('requests_progress', {}).values())
    row.update(stage=data.get('stage'), finished=sum(p['state']=='finished' for p in progress),
               running=sum(p['state']=='running' for p in progress),
               output_tokens=sum(p.get('output_tokens',0) for p in progress))
    rows.append(row)
print(json.dumps(dict(time=time.strftime('%H:%M:%S'), phase=state['phase'],
                      active=state['active_container'], completed=state['completed'],
                      heartbeat_age_s=round(time.time()-state['updated_unix'],1), cells=rows),ensure_ascii=False))
