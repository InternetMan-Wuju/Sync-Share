#!/usr/bin/env python3
"""Plot measured MTP acceptance, throughput and client latencies from summary JSON."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);root=ap.parse_args().root
scenes=['decode','mixed','prefix'];labels=['1K / 32K target\n45-min window, may be partial','8K / 1K','32K / 300\n~60% prefix cache']
data={s:json.loads((root/'analysis'/f'{s}_summary.json').read_text()) for s in scenes}
fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
x=np.arange(3);width=.32
for ax,key,title,ylabel in [(axs[0,1],'window_output_tokens_per_s','Output throughput in the 45-minute window','Tokens / second, including warmup gaps'),
                            (axs[1,0],'mean_ttft_s','Client TTFT mean (requests with output inside window)','Seconds'),
                            (axs[1,1],'mean_observed_tpot_ms','Observed TPOT mean (within-window outputs)','Milliseconds / token')]:
    for mode,offset,color in [('off',-width/2,'#487fad'),('on',width/2,'#dc8b31')]:
        vals=[data[s].get(mode,{}).get(key) for s in scenes]
        bars=ax.bar(x+offset,[v if isinstance(v,(int,float)) else 0 for v in vals],width,color=color,label='MTP '+mode.upper())
        for bar,val in zip(bars,vals):
            if val is not None:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{val:.2f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,labels,fontsize=9);ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.legend()
accept=[data[s].get('on',{}).get('weighted_accept_length') for s in scenes]
bars=axs[0,0].bar(x,[v if v is not None else 0 for v in accept],.52,color='#dc8b31')
for bar,val,scene in zip(bars,accept,scenes):
    coverage=data[scene].get('on',{}).get('acceptance_output_coverage_ratio')
    if val is not None:
        text=f'{val:.3f}' + (f'\ncoverage {coverage*100:.2f}%' if coverage is not None else '')
        axs[0,0].text(bar.get_x()+bar.get_width()/2,bar.get_height(),text,ha='center',va='bottom',fontsize=9)
axs[0,0].set_xticks(x,labels,fontsize=9);axs[0,0].set_title('MTP ON: weighted acceptance from terminal counters',fontsize=11)
axs[0,0].set_ylabel('Output tokens / verification, including bonus');axs[0,0].set_ylim(0,5);axs[0,0].grid(axis='y',alpha=.2)
fig.suptitle('MTP OFF node23 / MTP ON node25; same canonical weight bytes; Chunk 16384; Graph actually OFF',fontsize=12)
fig.savefig(root/'analysis'/'summary.png',dpi=170)
print(root/'analysis'/'summary.png')
