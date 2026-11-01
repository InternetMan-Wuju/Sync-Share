#!/usr/bin/env python3
"""Verify original checkpoint identity before a read-only reference overlay loads."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('model',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
manifest=json.loads((args.source/'manifest.json').read_text())
mismatches=[];total=0
for filename,info in manifest['files'].items():
    try:
        s=(args.model/filename).stat()
        if s.st_size!=info['bytes'] or s.st_mtime_ns!=info['mtime_ns']:
            mismatches.append({'file':filename,'expected':[info['bytes'],info['mtime_ns']],'actual':[s.st_size,s.st_mtime_ns]})
        total+=s.st_size
    except OSError as exc:mismatches.append({'file':filename,'error':str(exc)})
output={'checkpoint_files':len(manifest['files']),'bytes':total,'size_mtime_mismatches':mismatches,
    'reference_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest()}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v if k!='size_mtime_mismatches' else len(v) for k,v in output.items()}))
if mismatches:raise SystemExit(1)
