#!/usr/bin/env python3
"""Read saved raw SSE data and draw per-token ITL against staggered arrivals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load_group


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    root=ap.parse_args().root
    fig,axs=plt.subplots(2,1,figsize=(14,9),sharey=True,constrained_layout=True)
    colors={'A':'#2878b5','B':'#e18b21','C':'#3b9c6b','D':'#8f67a9'}
    for ax,chunk in zip(axs,(8192,1024)):
        path=root/f'node_23/chunk_{chunk}/bench'
        data,ticks,dispatch=load_group(path)
        zero=dispatch['A']['dispatch_perf_ns']
        for name in 'ABCD':
            frames=ticks[name]
            xs=[(frames[i][1]-zero)/1e9 for i in range(1,len(frames))]
            gaps=[(frames[i][1]-frames[i-1][1])/1e9 for i in range(1,len(frames))]
            ax.plot(xs,gaps,'.',markersize=2.4,alpha=.65,color=colors[name],label=f'{name} ITL')
        for incoming in 'BCD':
            start=(dispatch[incoming]['dispatch_perf_ns']-zero)/1e9
            first=(data[incoming]['first_token_perf_ns']-zero)/1e9
            ax.axvspan(start,first,color=colors[incoming],alpha=.14,
                        label=f'{incoming} request -> first token')
            ax.axvline(start,color=colors[incoming],alpha=.8,linestyle='--',linewidth=.8)
        ax.set_title(f'Chunk {chunk}: same 32K inputs / 300 output, A->B->C->D')
        ax.set_ylabel('Client ITL (seconds)')
        ax.set_xlabel('Seconds since A submission (client monotonic clock)')
        ax.set_ylim(0, 10.3)  # Same scale for direct 8K/1K spike comparison.
        ax.axhline(.5,color='black',linewidth=.7,alpha=.4)
        ax.grid(alpha=.25)
        ax.legend(loc='upper right',fontsize=8,ncol=2)
    fig.suptitle('ITL plotted at the later token arrival; a gap spanning an injection window may appear just to its right', fontsize=11)
    fig.savefig(root/'analysis'/'itl_timeline.png',dpi=165)
    print(root/'analysis'/'itl_timeline.png')

if __name__=='__main__':
    main()
