#!/usr/bin/env python3
"""Read persisted progress without sending additional inference requests."""
import datetime
import json
import time
import zlib
from pathlib import Path
import sys
root=Path(sys.argv[1]);report={'captured_at_local':datetime.datetime.now().astimezone().isoformat(),'stages':{}}
for scene in ('decode','mixed','prefix'):
    schedule=root/scene/'schedule.json'
    if not schedule.exists():continue
    cfg=json.loads(schedule.read_text())
    stage={'elapsed_s':round(time.monotonic()-cfg['start_perf_s'],1),'remaining_s':round(max(0,cfg['stop_perf_s']-time.monotonic()),1),'modes':{}}
    for mode in ('off','on'):
        folder=root/scene/mode
        completed=folder/'completed_or_partial_results.json'
        result_rows=json.loads(completed.read_text()) if completed.exists() else []
        active=[p for p in sorted(folder.glob('request_*/measurement')) if not (p/'result.json').exists()]
        entry={'saved_requests':len(result_rows),'saved_output_tokens':sum(r['actual_output'] for r in result_rows)}
        if active:
            p=active[-1]/'sse_raw.jsonl.gz';entry['active_folder']=active[-1].parent.name
            entry['compressed_raw_bytes']=p.stat().st_size if p.exists() else None
            if p.exists() and p.stat().st_size:
                try:
                    raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(p.read_bytes()).decode('utf-8','replace')
                    for line in reversed(raw.splitlines()):
                        try:
                            item=json.loads(line);text=item.get('raw_line','')
                            if text.startswith('data:') and text[5:].strip()!='[DONE]':
                                m=json.loads(text[5:]).get('meta_info',{})
                                entry['last_persisted_output']=m.get('completion_tokens')
                                entry['last_persisted_wall_ns']=item['recv_wall_ns']
                                break
                        except Exception:continue
                except Exception as exc:entry['progress_error']=repr(exc)
        stage['modes'][mode]=entry
    report['stages'][scene]=stage
(root/'live_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (root/'progress_monitor.jsonl').open('a') as history:history.write(json.dumps(report,ensure_ascii=False)+'\n')
print(json.dumps(report,ensure_ascii=False))
