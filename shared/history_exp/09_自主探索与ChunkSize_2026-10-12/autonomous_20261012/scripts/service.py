#!/usr/bin/env python3
"""Launch one owned experiment on node23; retain all stopped containers."""
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
WORK=ROOT.parents[1]
MODEL='/data_lib/data/models/glm-q'
DEFAULT_IMAGE='quay.io/ascend/sglang:v0.5.19-cann9.0.0-a3'


def docker(args,**kw):
    return subprocess.run(['sudo','-n','docker',*args],check=True,**kw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['start','stop','status'])
    ap.add_argument('--name',required=True);ap.add_argument('--image',default=DEFAULT_IMAGE)
    ap.add_argument('--repaired',action='store_true');ap.add_argument('--mtp',choices=['on','off'],default='on')
    ap.add_argument('--overlay',default='publisher')
    ap.add_argument('--dp',type=int,default=1);ap.add_argument('--ep',type=int,default=16)
    ap.add_argument('--dp-attention',action='store_true');ap.add_argument('--deepep',default='normal')
    ap.add_argument('--backend',default='deepep');ap.add_argument('--graph',action='store_true')
    ap.add_argument('--graph-bs',default='8 16 24 32');ap.add_argument('--graph-max-bs',type=int,default=32)
    ap.add_argument('--mem',type=float,default=.90);ap.add_argument('--context',type=int,default=131072)
    ap.add_argument('--mrr',type=int,default=8);ap.add_argument('--chunk',type=int,default=16384)
    ap.add_argument('--maxprefill',type=int,default=32768);ap.add_argument('--port',type=int,default=30080)
    ap.add_argument('--loader-v0516',action='store_true');ap.add_argument('--fusion',action='store_true')
    ap.add_argument('--dispatch-tokens',type=int);ap.add_argument('--diagnostics',action='store_true')
    ap.add_argument('--env',action='append',default=[])
    ap.add_argument('--deepep-fix',action='store_true')
    ap.add_argument('--ordered-diagnostic',action='store_true')
    ap.add_argument('--slim-draft',action='store_true')
    ap.add_argument('--extra',action='append',default=[])
    a=ap.parse_args()
    if not a.name.startswith('cyt-auto-'):raise RuntimeError('owned name prefix required')
    if a.action!='start':
        r=docker(['inspect',a.name],capture_output=True,text=True)
        d=json.loads(r.stdout)[0]
        if d['Config']['Labels'].get('owner')!='chengyetai':raise RuntimeError('not an owned container')
        if a.action=='stop':docker(['stop','-t','30',a.name]);return
        print(json.dumps(d['State']));return
    if a.chunk>a.maxprefill:raise ValueError('chunk exceeds maxprefill')
    info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    if info.count('No running processes found')!=8:raise RuntimeError('node23 is occupied; preserve current users')
    existing=subprocess.run(['sudo','-n','docker','inspect',a.name],capture_output=True)
    if existing.returncode==0:raise RuntimeError('name already exists; keep it and choose another name')
    out=ROOT/'logs'/a.name;out.mkdir(parents=True,exist_ok=False)
    (out/'npu_before.txt').write_text(info)
    for p in ('tmp','weight_audit','hf-cache'):(out/p).mkdir()
    env={'PYTHONDONTWRITEBYTECODE':'1','SGLANG_SET_CPU_AFFINITY':'1',
         'GLOO_SOCKET_IFNAME':'bond4','HCCL_SOCKET_IFNAME':'bond4','HCCL_IF_BASE_PORT':'50000',
         'HCCL_BUFFSIZE':'650','SGLANG_ENABLE_OVERPLAN_PLAN_STREAM':'1',
         'SGLANG_ENABLE_OVERLAP_PLAN_STREAM':'1','SGLANG_NPU_USE_MULTI_STREAM':'1',
         'STREAMS_PER_DEVICE':'32','PYTORCH_NPU_ALLOC_CONF':'expandable_segments:True',
         'SGLANG_ENABLE_FAST_INPUT_LOGPROBS':'0',
         'TRANSFORMERS_VERBOSITY':'error','HF_ENDPOINT':'https://hf-mirror.com',
         'HF_HOME':'/exp-logs/hf-cache','TMPDIR':'/exp-logs/tmp'}
    if a.dispatch_tokens:env['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']=str(a.dispatch_tokens)
    if a.diagnostics:env['SGLANG_ENABLE_ASYNC_ASSERT']='1'
    for setting in a.env:
        key,value=setting.split('=',1)
        if not key.replace('_','').isalnum() or key in ('HOME','CODEX_HOME','PYTHONPATH'):
            raise ValueError('unsupported environment key')
        env[key]=value
    if a.repaired:
        summary=json.loads((ROOT/'weights'/a.overlay/'summary.json').read_text())
        if not summary['complete']:raise RuntimeError('incomplete overlay')
        env.update(PYTHONPATH='/study/weights/loader',
                   SGLANG_READONLY_WEIGHT_MANIFEST='/study/weights/'+a.overlay+'/manifest.json',
                   SGLANG_READONLY_WEIGHT_MANIFEST_SHA256=summary['manifest_sha256'],
                   SGLANG_READONLY_WEIGHT_BASE='/models/glm-q',
                   SGLANG_READONLY_WEIGHT_AUDIT_DIR='/exp-logs/weight_audit')
    server=['python3','-m','sglang.launch_server','--tp-size','16','--dp-size',str(a.dp),
            '--ep-size',str(a.ep),'--nnodes','1','--node-rank','0','--dist-timeout','3600',
            '--model-path','/models/glm-q','--served-model-name','glm-5.2','--host','0.0.0.0',
            '--port',str(a.port),'--device','npu','--attention-backend','ascend','--dtype','bfloat16',
            '--quantization','modelslim','--trust-remote-code','--moe-a2a-backend',a.backend,
            '--context-length',str(a.context),'--chunked-prefill-size',str(a.chunk),
            '--max-prefill-tokens',str(a.maxprefill),'--max-running-requests',str(a.mrr),
            '--mem-fraction-static',str(a.mem),'--enable-cache-report','--enable-metrics',
            '--enable-metrics-for-all-schedulers','--tokenizer-worker-num','4','--watchdog-timeout','9000']
    if a.backend=='deepep':server+=['--deepep-mode',a.deepep]
    elif a.backend=='none':server+=['--ep-dispatch-algorithm','dynamic']
    if a.dp_attention:server+=['--enable-dp-attention','--enable-dp-lm-head']
    if not a.fusion:server+=['--disable-shared-experts-fusion']
    if a.graph:server+=['--cuda-graph-backend-decode','full','--cuda-graph-max-bs-decode',str(a.graph_max_bs),'--cuda-graph-bs-decode',*a.graph_bs.split()]
    else:server+=['--disable-cuda-graph']
    if a.mtp=='on':server+=['--speculative-algorithm','NEXTN','--speculative-draft-model-quantization','unquant',
                          '--speculative-num-steps','3','--speculative-eagle-topk','1','--speculative-num-draft-tokens','4']
    if a.slim_draft and a.mtp=='on':server+=['--speculative-draft-model-path','/models/mtp']
    for v in a.extra:server+=shlex.split(v)
    shell='set -eo pipefail\nunset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING\n'
    shell+='source /usr/local/Ascend/ascend-toolkit/set_env.sh\nsource /usr/local/Ascend/nnal/atb/set_env.sh\n'
    shell+='\n'.join('export '+k+'='+shlex.quote(v)+(':"${PYTHONPATH:-}"' if k=='PYTHONPATH' else '') for k,v in env.items())+'\n'
    shell+=shlex.join(['python3','-c','import tbe; print("[preflight] CANN tbe import OK", flush=True)'])+'\n'
    shell+=shlex.join(server)+' 2>&1 | tee /exp-logs/server.log\n'
    (out/'launch.sh').write_text(shell)
    (out/'config.json').write_text(json.dumps({**vars(a),'node':'23','env':env,'server_command':server,
                                             'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    cmd=['run','-d','--name',a.name,'--net','host','--ipc','host','--shm-size','16g','--pids-limit','-1',
         '--label','owner=chengyetai','--label','experiment=autonomous_mtp_chunk']
    for i in range(16):cmd+=['--device',f'/dev/davinci{i}']
    for dev in ('davinci_manager','devmm_svm','hisi_hdc'):cmd+=['--device','/dev/'+dev]
    for path in ('/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/etc/ascend_install.info','/var/queue_schedule'):
        cmd+=['-v',path+':'+path+':ro']
    cmd+=['-v',MODEL+':/models/glm-q:ro','-v',str(ROOT)+':/study:ro','-v',str(out)+':/exp-logs']
    if a.slim_draft:cmd+=['-v',str(ROOT/'weights/draft_view')+':/models/mtp:ro']
    if a.loader_v0516:
        p=WORK/'current_exp/Q1MTPExp/scripts/load_model_utils_v0516.py'
        cmd+=['-v',str(p)+':/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:ro']
    if a.ordered_diagnostic:
        if a.image!=DEFAULT_IMAGE or not a.diagnostics:raise ValueError('ordered probe is for audited v19 diagnostics only')
        for filename in ('eagle_worker_common.py','eagle_worker_v2.py'):
            p=ROOT/'patches/ordered_diagnostic'/filename
            cmd+=['-v',str(p)+':/sgl-workspace/sglang/python/sglang/srt/speculative/'+filename+':ro']
    if a.deepep_fix:
        p=WORK/'history_exp/03_ChunkSize单机_2026-09/chunk_manual_20260927/patches/deepep_fix_20260927/deep_ep'
        cmd+=['-v',str(p)+':/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep:ro']
    cmd+=['--entrypoint','bash',a.image,'-lc',shell]
    result=docker(cmd,capture_output=True,text=True)
    (out/'container_id.txt').write_text(result.stdout)
    print(json.dumps({'name':a.name,'logs':str(out),'container':result.stdout.strip()}))


if __name__=='__main__':main()
