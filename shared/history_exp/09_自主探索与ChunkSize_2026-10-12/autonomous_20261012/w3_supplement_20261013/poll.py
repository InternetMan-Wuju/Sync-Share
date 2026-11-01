#!/usr/bin/env python3
"""Read-only status of the supplemental W3 matrix."""
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parent
state = json.loads((root / 'status.json').read_text())
cells = []
for cell in state['cells']:
    path = Path(cell['out']) / 'status.json'
    detail = json.loads(path.read_text()) if path.exists() else {}
    row = {k: cell.get(k) for k in ('id', 'valid', 'error')}
    row.update(stage=detail.get('stage'), progress=None if cell['valid'] else detail.get('requests_progress'))
    cells.append(row)
print(json.dumps(dict(time=time.strftime('%H:%M:%S'), phase=state['phase'],
                      completed=state['completed'], heartbeat_age_s=round(time.time()-state['updated_unix'], 1),
                      active=state['active_container'], cells=cells), ensure_ascii=False))
