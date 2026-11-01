#!/usr/bin/env python3
"""Export static scientific plots of completed records only."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/home/chengyetai/workspace/current_exp/autonomous_20261012/figures/mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from report_labels import service_label, workload_label

_cjk_font = Path('/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf')
if _cjk_font.exists():
    font_manager.fontManager.addfont(str(_cjk_font))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(_cjk_font)).get_name(), 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK=ROOT.parent/'Qchunksize'


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    files=sorted((ROOT/'results').glob('*/*/result.json'))
    values=[]
    for p in files:
        if p.parent.name=='sanity_chat':continue
        d=json.loads(p.read_text())
        if any(v.get('failures') for v in d.get('validation',[])) or any(r.get('error') or not r.get('stream_done') for r in d['requests']):continue
        for r in d['requests']:
            if r.get('error') or not r['meta'].get('spec_verify_ct'):continue
            cycle=r.get('output_shape',{}).get('periodic_suffix')
            values.append({'run':p.parent.parent.name,'group':p.parent.name,'index':r['index'],
                           'AL':r['meta']['spec_accept_length'],'repeat_fraction':cycle['fraction'] if cycle else 0})
    if values:
        fig,ax=plt.subplots(figsize=(15,6))
        groups=sorted(set(v['run'] for v in values));colors=plt.get_cmap('tab10')
        for i,group in enumerate(groups):
            rows=[v for v in values if v['run']==group]
            ax.scatter([v['repeat_fraction'] for v in rows],[v['AL'] for v in rows],label=service_label(group),
                       color=colors(i%10),alpha=.7,s=32)
        ax.set(xlabel='精确循环尾段占输出的比例',ylabel='每轮验证平均输出token数（AL）',ylim=(.95,4.1))
        ax.axhline(2.11,color='gray',ls='--',lw=1,label='参考AL 2.11（条件不一致）')
        ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.02,.5));ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/'mtp_AL_and_repetition.png',dpi=180);fig.savefig(out/'mtp_AL_and_repetition.pdf');plt.close(fig)
        (out/'mtp_plot_points.json').write_text(json.dumps(values,indent=2))
    evidence=CHUNK/'ChunkSize负载与阶段分析.json'
    if evidence.exists():
        data=json.loads(evidence.read_text());groups=data['groups']
        if groups:
            fig,axes=plt.subplots(len(groups),3,figsize=(13,3.2*len(groups)),squeeze=False)
            for i,g in enumerate(groups):
                rows=sorted(g['candidates'],key=lambda r:r['chunk']);x=np.arange(len(rows));labels=[str(r['chunk']) for r in rows]
                for j,(key,label) in enumerate([('throughput_mean','整套服务每秒输出token数'),('ttft_p50_mean_s','首token等待时间中位数（秒）'),('tpot_p50_mean_ms','每token生成时间中位数（毫秒）')]):
                    y=[r[key] for r in rows];axes[i,j].bar(x,y,color='#4678a3');axes[i,j].set_xticks(x,labels);axes[i,j].set_ylabel(label)
                    axes[i,j].set_title(workload_label(g['scene']).replace('，','\n')+'\n客户端并发'+str(g['concurrency']),fontsize=10);axes[i,j].set_xlabel('每块输入token数');axes[i,j].grid(axis='y',alpha=.2)
                    if j==0:
                        lower=[r['throughput_mean']-r['throughput_min'] for r in rows]
                        upper=[r['throughput_max']-r['throughput_mean'] for r in rows]
                        axes[i,j].errorbar(x,y,yerr=[lower,upper],fmt='none',ecolor='black',capsize=3)
            fig.tight_layout();fig.savefig(out/'chunk_performance.png',dpi=180);fig.savefig(out/'chunk_performance.pdf');plt.close(fig)
            fig,axes=plt.subplots(len(groups),1,figsize=(10,3.1*len(groups)),squeeze=False)
            components=[('server_before_forward_s','服务器接收至首次计算','#8094a8'),
                        ('server_prefill_span_s','输入处理时间','#e2aa57'),
                        ('server_decode_and_finish_s','生成输出至结束','#5f9e8f'),
                        ('client_minus_server_e2e_s','客户端与服务器时间残差','#ae7a9d')]
            for i,g in enumerate(groups):
                chunks=sorted(r['chunk'] for r in g['candidates']);x=np.arange(len(chunks));bottom=np.zeros(len(chunks))
                cells=[c for c in data['cells'] if c['scene']==g['scene'] and c['concurrency']==g['concurrency']
                       and c['aggregate']['complete'] and c['aggregate']['cache_condition_valid']]
                for key,label,color in components:
                    heights=[np.mean([c['aggregate']['timestamp_breakdown_mean_s'][key] for c in cells if c['chunk_requested']==chunk]) for chunk in chunks]
                    axes[i,0].bar(x,heights,bottom=bottom,label=label,color=color);bottom+=heights
                totals=[np.mean([c['aggregate']['e2e_s_mean'] for c in cells if c['chunk_requested']==chunk]) for chunk in chunks]
                axes[i,0].plot(x,totals,'ko',ms=4,label='客户端全程平均耗时')
                axes[i,0].set_xticks(x,[str(c) for c in chunks]);axes[i,0].set_ylabel('平均耗时（秒）');axes[i,0].set_xlabel('每块输入token数')
                axes[i,0].set_title(workload_label(g['scene'])+'\n客户端并发'+str(g['concurrency']));axes[i,0].grid(axis='y',alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
            fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'chunk_E2E_breakdown.png',dpi=180)
            fig.savefig(out/'chunk_E2E_breakdown.pdf');plt.close(fig)
    print('plots exported to',out)


if __name__=='__main__':main()
