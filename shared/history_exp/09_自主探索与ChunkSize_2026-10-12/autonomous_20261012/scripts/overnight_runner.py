#!/usr/bin/env python3
"""Sequential owned-node experiments; durable status and conservative fallback."""
import datetime
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
SCRIPTS=ROOT/'scripts'
URL='http://127.0.0.1:30080'
STATE={'runner_pid':os.getpid(),'started_unix':time.time(),'completed':False,'phase':'initializing','runs':[]}
ACTIVE=None


def status(phase,**extra):
    STATE.update(phase=phase,updated_unix=time.time(),**extra)
    tmp=ROOT/'overnight_status.tmp';tmp.write_text(json.dumps(STATE,ensure_ascii=False,indent=2));tmp.replace(ROOT/'overnight_status.json')
    print(datetime.datetime.now().isoformat(),phase,extra,flush=True)


def command(args,log,timeout=7200):
    STATE['command_started']=time.time()
    status('command',command=args,command_log=str(log))
    with log.open('a') as f:
        p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            STATE['child_pid']=p.pid;status('command_running',command=args,command_log=str(log))
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            if time.time()-STATE['command_started']>timeout:
                p.terminate();p.wait(timeout=30);return 124
        STATE.pop('child_pid',None)
        return p.returncode


def http(path,timeout=8):
    with urllib.request.urlopen(URL+path,timeout=timeout) as r:return r.read()


def state_of(name):
    p=subprocess.run(['sudo','-n','docker','inspect',name,'--format','{{json .State}}'],capture_output=True,text=True,timeout=20)
    return json.loads(p.stdout) if p.returncode==0 else None


def stop():
    global ACTIVE
    if ACTIVE:
        name=ACTIVE;status('stopping_owned_container',container=name)
        subprocess.run([sys.executable,'-B',str(SCRIPTS/'service.py'),'stop','--name',name],timeout=60,check=False)
        ACTIVE=None
    for _ in range(12):
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20).stdout
        if info.count('No running processes found')==8:break
        time.sleep(5)
    else:raise RuntimeError('NPU remains occupied; preserve other users')
    time.sleep(15)


def launch(tag,flags):
    global ACTIVE
    name='cyt-auto-'+tag
    log=ROOT/'logs'/f'{tag}_launch_driver.log'
    args=[sys.executable,'-B',str(SCRIPTS/'service.py'),'start','--name',name,'--repaired','--slim-draft',*flags]
    rc=command(args,log,timeout=120)
    record={'tag':tag,'container':name,'flags':flags,'started_unix':time.time(),'launch_rc':rc}
    STATE['runs'].append(record)
    if rc:
        record['healthy']=False;record['failure']='launch command failed';return record
    ACTIVE=name
    return wait_ready(record)


def wait_ready(record):
    global ACTIVE
    name=record['container'];ACTIVE=name
    status('waiting_service',container=name)
    deadline=time.time()+3600
    while time.time()<deadline:
        d=state_of(name)
        if not d or not d['Running']:
            record['healthy']=False;record['failure']='container exited before health';return record
        logpath=ROOT/'logs'/name/'server.log'
        if logpath.exists():
            with logpath.open('rb') as stream:
                stream.seek(max(0,logpath.stat().st_size-262144));tail=stream.read().decode(errors='replace')
            if ('Subprocess scheduler_' in tail and 'crashed with exit code' in tail) or 'scheduler died during initialization' in tail:
                record['healthy']=False;record['failure']='scheduler crashed while HTTP router remained alive';return record
        try:
            http('/health',timeout=3)
            record['healthy']=True
            info=http('/get_server_info',timeout=30)
            (ROOT/'logs'/name/'ready_server_info.json').write_bytes(info)
            status('service_healthy',container=name);return record
        except Exception:time.sleep(10)
    record['healthy']=False;record['failure']='health timeout';return record


def suite(record,full=False):
    tag=record['tag'];out=ROOT/'results'/tag
    args=[sys.executable,'-B',str(SCRIPTS/'mtp_probe_suite.py'),'--container',record['container'],'--out',str(out)]
    if full:args+=['--full-short-groups']
    else:args+=['--only','sanity_chat,native_exact_1k256,native_exact_eos_1k256,native_longinput_1k1024']
    rc=command(args,ROOT/'logs'/f'{tag}_suite.log',timeout=7200)
    record['suite_rc']=rc;record['suite_status']=str(out/'status.json')
    if rc==0:
        args=[sys.executable,'-B',str(SCRIPTS/'mtp_probe_suite.py'),'--container',record['container'],
              '--out',str(ROOT/'results'/(tag+'_business')),'--business']
        record['business_rc']=command(args,ROOT/'logs'/(tag+'_business.log'),timeout=2400)
    return rc==0


def main(resume_graph_off=False,resume_hccl1000=False,resume_tuned=False,resume_dpnormal=False,resume_viable=False,resume_fuller=False,resume_after_ordered=False):
    global ACTIVE
    datasets=ROOT.parent/'Qchunksize/datasets/native_20261012'
    if not (resume_graph_off or resume_hccl1000 or resume_tuned or resume_dpnormal or resume_viable or resume_fuller or resume_after_ordered):
        ACTIVE='cyt-auto-mtp-publisher-v19-slim'
        status('waiting_primary_weight_probe',container=ACTIVE)
        p=ROOT/'results/publisher_v19/status.json';deadline=time.time()+7200
        while time.time()<deadline:
            if p.exists():
                d=json.loads(p.read_text())
                status('waiting_primary_weight_probe',primary_stage=d['stage'],primary_group=d.get('current_group'))
                if d.get('completed'):break
                if d['stage']=='failed':raise RuntimeError('primary weight probe failed: '+str(d.get('error')))
            time.sleep(20)
        else:raise RuntimeError('primary suite timeout')
        STATE['runs'].append({'tag':'publisher_v19','container':ACTIVE,'healthy':True,'suite_rc':0,'suite_status':str(p)})
        business=[sys.executable,'-B',str(SCRIPTS/'mtp_probe_suite.py'),'--container',ACTIVE,
                  '--out',str(ROOT/'results/publisher_v19_business'),'--business']
        rc=command(business,ROOT/'logs/publisher_v19_business.log',timeout=2400)
        if rc:raise RuntimeError('business probe did not complete')
        diagnostic=[sys.executable,'-B',str(SCRIPTS/'trajectory_check.py'),
                    '--source',str(ROOT/'results/publisher_v19'),
                    '--out',str(ROOT/'results/publisher_v19_trajectory')]
        rc=command(diagnostic,ROOT/'logs/publisher_v19_trajectory.log',timeout=2400)
        STATE['runs'][-1]['trajectory_rc']=rc
        shutil.copytree(ROOT/'logs'/ACTIVE/'native_chunk_datasets',datasets,dirs_exist_ok=True)
        stop()
    if not resume_after_ordered:
        base=['--image','quay.io/ascend/sglang:cann9.0.0-a3-v0.5.16','--loader-v0516',
              '--dp','2','--dp-attention','--context','180000','--mrr','64','--mem','0.90',
              '--deepep','auto','--fusion']
        if resume_dpnormal or resume_viable or resume_fuller:
            default={'healthy':True}
        elif resume_graph_off or resume_hccl1000 or resume_tuned:
            default=next(r for r in STATE['runs'] if r['tag']=='mtp-default-v16')
        else:
            default=launch('mtp-default-v16',base+['--graph'])
            if default['healthy']:suite(default)
            stop()
        if not default['healthy'] or default.get('suite_rc'):
            # Graph alone changes first, so a repeated failure can be attributed properly.
            if resume_hccl1000 or resume_tuned:
                offgraph=next(r for r in STATE['runs'] if r['tag']=='mtp-default-v16-graph-off')
            elif resume_graph_off:
                offgraph=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-default-v16-graph-off'))
            else:offgraph=launch('mtp-default-v16-graph-off',base)
            if offgraph['healthy']:suite(offgraph)
            if not (resume_hccl1000 or resume_tuned):stop()
            if not offgraph['healthy'] or offgraph.get('suite_rc'):
                # The captured error specifies 825MB versus 650MB. Try this one
                # variable before reducing concurrency or graph batches.
                if resume_tuned:
                    buffered=next(r for r in STATE['runs'] if r['tag']=='mtp-default-v16-hccl1000')
                elif resume_hccl1000:
                    buffered=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-default-v16-hccl1000'))
                else:buffered=launch('mtp-default-v16-hccl1000',base+['--graph','--env','HCCL_BUFFSIZE=1000'])
                if buffered['healthy']:suite(buffered)
                if not resume_tuned:stop()
                if not buffered['healthy'] or buffered.get('suite_rc'):
                    capacity_flags=base+['--mrr','8','--dispatch-tokens','64',
                                        '--graph','--graph-bs','1 2 4 8','--graph-max-bs','8',
                                        '--env','HCCL_BUFFSIZE=1536','--env','HCCL_OP_EXPANSION_MODE=AIV']
                    if resume_tuned:
                        tuned=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-v16-auto-cap64'))
                    else:tuned=launch('mtp-v16-auto-cap64',capacity_flags)
                    if tuned['healthy']:suite(tuned)
                    stop()
                    if not tuned['healthy'] or tuned.get('suite_rc'):
                        # Preserve capacity/Graph/communication settings, reserve more
                        # NPU memory by changing only the static fraction.
                        lowmem=launch('mtp-v16-auto-cap64-mem84',capacity_flags+['--mem','0.84'])
                        if lowmem['healthy']:suite(lowmem)
                        stop()
        if not resume_fuller:
            if not resume_viable:
                # A normal-mode DP2 case isolates DP attention from low-latency.
                if resume_dpnormal:
                    dpnormal=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-v19-dp2-normal'))
                else:dpnormal=launch('mtp-v19-dp2-normal',['--dp','2','--dp-attention','--context','180000','--mrr','8','--diagnostics'])
                if dpnormal['healthy']:suite(dpnormal)
                stop()
            # mem84 cannot hold weights+draft; mem90 leaves little dynamic HCCL space.
            viable_flags=base+['--mrr','8','--dispatch-tokens','64','--mem','0.87',
                              '--graph','--graph-bs','1 2 4 8','--graph-max-bs','8',
                              '--env','HCCL_OP_EXPANSION_MODE=AIV']
            if resume_viable:
                viable=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-v16-auto-cap64-mem87-hccl650'))
            else:viable=launch('mtp-v16-auto-cap64-mem87-hccl650',viable_flags)
            if viable['healthy']:suite(viable)
            stop()
            # If capacity64/650MB works at.87, isolate the static fraction at.90.
            # The earlier.90 tuned failure used1536MB, so it does not test this case.
            if viable['healthy'] and viable.get('suite_rc')==0:
                fuller=launch('mtp-v16-auto-cap64-mem90-hccl650',viable_flags+['--mem','0.90'])
                if fuller['healthy']:suite(fuller)
                stop()
        else:
            fuller=wait_ready(next(r for r in STATE['runs'] if r['tag']=='mtp-v16-auto-cap64-mem90-hccl650'))
            if fuller['healthy']:suite(fuller)
            stop()
        # Retain all assertions, move only the token probe to the producing stream.
        ordered=launch('mtp-v19-dp2-ordered-probe',['--dp','2','--dp-attention','--context','180000','--mrr','8',
                      '--diagnostics','--ordered-diagnostic'])
        if ordered['healthy']:suite(ordered)
        stop()
    # Isolate plan-stream overlap itself with the original probes retained.
    noplan=launch('mtp-v19-dp2-no-plan-lp-safe',['--dp','2','--dp-attention','--context','180000','--mrr','8',
                 '--diagnostics','--env','SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0'])
    if noplan['healthy']:suite(noplan)
    stop()
    return chunk_grid()


def chunk_grid():
    datasets=ROOT.parent/'Qchunksize/datasets/native_20261012'
    # ChunkSize MTP OFF is a separate task; all comparisons share repaired weights.
    # DP1 / target decode has one row per request; TP16 alignment needs bs16.
    # The earlier MTP DP2 list1/2/4/8 yielded an empty OFF-MTP capture list.
    fast_flags=['--mtp','off','--deepep','auto','--graph','--mrr','16','--graph-bs','1 2 4 8 16',
                '--graph-max-bs','16','--dispatch-tokens','64','--env','HCCL_BUFFSIZE=650',
                '--env','HCCL_OP_EXPANSION_MODE=AIV']
    chunk_record=launch('chunk-auto-hccl650-16384',fast_flags)
    chunk_flags=fast_flags
    if chunk_record['healthy']:
        sanity_args=[sys.executable,'-B',str(SCRIPTS/'mtp_probe_suite.py'),'--container',chunk_record['container'],
                     '--out',str(ROOT/'results/chunk_auto_sanity'),'--only','sanity_chat']
        rc=command(sanity_args,ROOT/'logs/chunk_auto_sanity.log',timeout=600)
        chunk_record['sanity_rc']=rc
    else:rc=1
    if rc:
        stop();chunk_flags=['--mtp','off','--deepep','normal','--mrr','16']
        chunk_record=launch('chunk-normal-hccl650-16384',chunk_flags)
        if not chunk_record['healthy']:raise RuntimeError('no stable ChunkSize service')
        sanity_args=[sys.executable,'-B',str(SCRIPTS/'mtp_probe_suite.py'),'--container',chunk_record['container'],
                     '--out',str(ROOT/'results/chunk_normal_sanity'),'--only','sanity_chat']
        rc=command(sanity_args,ROOT/'logs/chunk_normal_sanity.log',timeout=600)
        if rc:raise RuntimeError('ChunkSize semantic sanity failed')
    chunk_root=ROOT.parent/'Qchunksize/results/autonomous_20261012';chunk_root.mkdir(parents=True,exist_ok=True)
    # No injection timing control: ordinary fixed client concurrency, 2C/C requests.
    cells=[('w1_8k_1k',2),('w1_8k_1k',4),('w2_32k_1k',1),('w4_prefix32k_300',1),('w4_prefix32k_300',2)]
    for chunk in (16384,1024,4096,8192):
        if chunk!=16384:
            stop();chunk_record=launch('chunk-'+('auto' if '--graph' in chunk_flags else 'normal')+'-'+str(chunk),
                                      chunk_flags+['--chunk',str(chunk)])
            if not chunk_record['healthy']:
                raise RuntimeError('chunk restart failed; no configuration mixing')
        for scene,conc in cells:
            name=f'{scene}_chunk{chunk}_c{conc}';out=chunk_root/name
            args=[sys.executable,'-B',str(SCRIPTS/'chunk_collect.py'),'--dataset',str(datasets/(scene+'.json')),
                  '--out',str(out),'--concurrency',str(conc),'--chunk',str(chunk)]
            rc=command(args,ROOT/'logs'/(name+'.log'),timeout=7200)
            chunk_record.setdefault('cells',[]).append({'name':name,'rc':rc,'result':str(out/'result.json')})
            status('chunk_cell_done',cell=name,rc=rc)
            if rc:raise RuntimeError('invalid chunk cell; inspect preserved data before proceeding')
    stop()
    status('planned_grid_completed',completed=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume-graph-off',action='store_true');ap.add_argument('--resume-hccl1000',action='store_true');ap.add_argument('--resume-tuned',action='store_true');ap.add_argument('--resume-dpnormal',action='store_true');ap.add_argument('--resume-viable',action='store_true');ap.add_argument('--resume-fuller',action='store_true');ap.add_argument('--resume-after-ordered',action='store_true');ap.add_argument('--resume-chunk-aligned',action='store_true');args=ap.parse_args()
    if args.resume_chunk_aligned:
        saved=json.loads((ROOT/'overnight_status.json').read_text())
        assert saved['phase']=='ready_to_resume_chunk_aligned' and not saved.get('child_pid'),saved
        STATE=saved;STATE['runner_pid']=os.getpid();STATE['resumed_unix']=time.time()
    if args.resume_graph_off or args.resume_hccl1000 or args.resume_tuned or args.resume_dpnormal or args.resume_viable or args.resume_fuller or args.resume_after_ordered:
        saved=json.loads((ROOT/'overnight_status.json').read_text())
        resume_options=[
            ('resume_after_ordered','cyt-auto-mtp-v19-dp2-no-plan'),
            ('resume_fuller','cyt-auto-mtp-v16-auto-cap64-mem90-hccl650'),
            ('resume_viable','cyt-auto-mtp-v16-auto-cap64-mem87-hccl650'),
            ('resume_dpnormal','cyt-auto-mtp-v19-dp2-normal'),
            ('resume_tuned','cyt-auto-mtp-v16-auto-cap64'),
            ('resume_hccl1000','cyt-auto-mtp-default-v16-hccl1000'),
            ('resume_graph_off','cyt-auto-mtp-default-v16-graph-off')]
        expected=next(name for option,name in resume_options if getattr(args,option))
        assert saved['phase']=='waiting_service' and saved['container']==expected,saved
        assert not saved.get('child_pid'),saved
        STATE=saved;STATE['runner_pid']=os.getpid();STATE['resumed_unix']=time.time()
    try:
        if args.resume_chunk_aligned:chunk_grid()
        else:main(args.resume_graph_off,args.resume_hccl1000,args.resume_tuned,args.resume_dpnormal,args.resume_viable,args.resume_fuller,args.resume_after_ordered)
    except Exception as e:
        status('needs_conservative_review',error=repr(e))
        try:stop()
        except Exception as stop_error:status('needs_conservative_review',stop_error=repr(stop_error))
        raise
