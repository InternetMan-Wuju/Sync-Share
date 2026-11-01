#!/usr/bin/env python3
"""Plot long-decode output rate by token progress and the instantaneous acceptance gauge.

Reads archived original SSE only; every point covers approximately 1024 newly
observed tokens. A partial 32K response remains explicitly partial.
"""
import argparse
import csv
import gzip
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def steps(path,mode,block=1024):
    rows=[];last=None; target=block
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            raw=item.get('raw_line','')
            if not raw.startswith('data:') or raw[5:].strip()=='[DONE]':continue
            obj=json.loads(raw[5:]);m=obj.get('meta_info',{})
            c=m.get('completion_tokens')
            if not isinstance(c,int) or c<1:continue
            now={'t':item['recv_perf_ns'],'c':c,'ver':m.get('spec_verify_ct'),
                 'accepted':m.get('spec_num_correct_drafts'),'proposed':m.get('spec_num_proposed_drafts')}
            if last is None:
                last=now
            if c<target:continue
            dt=(now['t']-last['t'])/1e9;delta=c-last['c']
            dv=now['ver']-last['ver'] if isinstance(now['ver'],int) and isinstance(last['ver'],int) else 0
            accepted=now['accepted']-last['accepted'] if isinstance(now['accepted'],int) and isinstance(last['accepted'],int) else 0
            proposed=now['proposed']-last['proposed'] if isinstance(now['proposed'],int) and isinstance(last['proposed'],int) else 0
            if dt>0 and delta>0:
                rows.append({'mode':mode,'output_tokens_at_window_end':c,'window_output_tokens':delta,
                     'window_seconds':dt,'token_per_second':delta/dt,
                     'mtp_window_accept_length':delta/dv if dv else '',
                     'mtp_window_draft_accept_rate':accepted/proposed if proposed else ''})
            last=now;target=c+block
    return rows


def main(root):
    out=root/'analysis';out.mkdir(exist_ok=True)
    rows=[]
    for mode in ('on','off'):
        p=root/'decode'/mode/'request_0000/measurement/sse_raw.jsonl.gz'
        if p.exists():rows+=steps(p,mode)
    if not rows:return
    with (out/'decode_progress.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    for mode,color in [('off','#4079b0'),('on','#d98222')]:
        x=[r for r in rows if r['mode']==mode]
        if x:axs[0].plot([r['output_tokens_at_window_end'] for r in x],[r['token_per_second'] for r in x],'.-',color=color,label=f'MTP {mode.upper()}')
    gauges=[]
    metric_file=root/'decode/on/metrics_raw.jsonl.gz'
    if metric_file.exists():
        zero=json.loads((root/'decode/schedule.json').read_text())['start_perf_s']
        with gzip.open(metric_file,'rt',encoding='utf-8') as f:
            for line in f:
                sample=json.loads(line)
                for row in sample.get('raw_prometheus','').splitlines():
                    if row.startswith('sglang:spec_accept_length{') and 'tp_rank="0"' in row:
                        gauges.append(((sample['perf_ns']/1e9)-zero,float(row.rsplit(' ',1)[1])))
                        break
    if gauges:
        axs[1].plot([g[0] for g in gauges],[g[1] for g in gauges],'-',color='#d98222',label='Prometheus instantaneous acceptance gauge (not whole-request mean)')
    axs[0].set_ylabel('Observed output rate (token/s)')
    axs[0].set_xlabel('Cumulative observed output tokens (approximately 1024-token windows)')
    axs[1].set_ylabel('Acceptance length (instantaneous gauge)')
    axs[1].set_xlabel('Seconds since the 45-minute scenario started')
    axs[1].set_ylim(bottom=1)
    for ax in axs:ax.legend();ax.grid(alpha=.3)
    fig.suptitle('1K input / 32K target output: observed partial responses; lower panel is not cumulative',fontsize=12)
    fig.savefig(out/'decode_progress.png',dpi=170)
    print(out/'decode_progress.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);main(p.parse_args().root)
