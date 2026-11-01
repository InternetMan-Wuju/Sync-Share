#!/usr/bin/env python3
import datetime,json,subprocess,time,urllib.request
from pathlib import Path
base=Path(__file__).resolve().parent
start=time.monotonic()
for attempt in range(120):
    status={"at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"elapsed_s":round(time.monotonic()-start,1)}
    p=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=8","-o","LogLevel=ERROR","10.120.72.25","sudo -n docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' chunk-manual25-8192"],capture_output=True,text=True,timeout=20)
    status["container"]=p.stdout.strip()
    if p.returncode or not p.stdout.startswith("running "):
        status["failure"]=True
        (base/"readiness.json").write_text(json.dumps(status,indent=2)+"\n")
        print(json.dumps(status),flush=True)
        raise SystemExit(1)
    try:
        with urllib.request.urlopen("http://10.120.72.25:30080/health",timeout=5) as r:
            status["health_http"]=r.status
    except Exception as e:
        status["health_error"]=str(e)
    print(json.dumps(status),flush=True)
    if status.get("health_http")==200:
        (base/"readiness.json").write_text(json.dumps(status,indent=2)+"\n")
        break
    time.sleep(10)
else:
    raise SystemExit("Readiness timeout")
