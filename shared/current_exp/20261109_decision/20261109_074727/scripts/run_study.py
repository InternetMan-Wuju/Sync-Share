"""Host orchestrator. No sampling after the recorded run deadline."""
from pathlib import Path
import subprocess,os,time,datetime,json,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ['DOCKER_HOST']='unix:///tmp/chengyetai-rootless-docker/docker.sock'
deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'START.json').read_text())['stop_by']).timestamp()
def run(cmd,name,timeout=900):
    remaining=deadline-time.time()-600 # reserve ten minutes for packaging
    assert remaining>0,'sampling deadline reached'
    with (ROOT/'local'/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=min(timeout,remaining),cwd=ROOT)
    assert result.returncode==0,(name,result.returncode)
    print('DONE',name,datetime.datetime.now().isoformat(),flush=True)
def request(case,split,name,scales='1,2,4,8'):
    run(['docker','exec','cyt-decision-23','python3','/runexp/scripts/request_decision.py','--case',str(case),'--split',split,'--name',name,'--scales',scales],name)
while time.time()<deadline-1200:
    log=(ROOT/'local/server.log').read_text(errors='replace')
    if 'The server is fired up and ready to roll!' in log:break
    if 'Scheduler hit an exception' in log:raise RuntimeError('server failed; inspect local/server.log')
    time.sleep(10)
else:raise TimeoutError('model not ready within sampling budget')
print('SERVER_READY',flush=True)
request(0,'train','dev_remaining','4,8')
run([sys.executable,'scripts/fit_model.py'],'fit_model',120)
request(1,'test','heldout_case1')
request(2,'test','heldout_case2')
(ROOT/'artifacts/SAMPLING_FINISHED.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('STUDY_DONE',flush=True)
