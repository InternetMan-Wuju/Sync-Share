#!/usr/bin/env python3
"""Offline event-aligned Chunk comparison for the single four-request run.

Reads raw SSE through the checked parser in ../prefill_comparison_20260928.
An injection event is a within-round observation, never an independent repeat.
The shaded interval uses only client monotonic timestamps: dispatch -> first
output of the new request. Server Prefill is reported as a duration, not mapped
onto that client timeline. No online service requests are made.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median
import csv
import hashlib
import importlib.util
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'chunk_redo_20260928/results/20260928_043035'
DOC = ROOT / 'doc/ChunkSize_单轮Prefill注入干扰分析_2026-09-28.md'
PARSER = OUT.parent / 'prefill_comparison_20260928/plot_report.py'
spec = importlib.util.spec_from_file_location('checked_chunk_parser', PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
CHUNKS = (8192, 1024)
COLORS = {8192:'#2563A6', 1024:'#DF7D24'}
STAGES = parser.STAGES
PAIRS = [('A','B'),('B','C'),('C','D')]
FIGURES = []
INPUTS = set()


def read_json(path):
    INPUTS.add(path)
    return json.loads(path.read_text())


def csv_out(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def nearest(values,q):
    return sorted(values)[math.ceil(len(values)*q)-1]


def chart_style(ax,axis='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis,color='#E4EAF0',linewidth=.7)


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    FIGURES.append((name,fig))


def derive():
    with ThreadPoolExecutor(max_workers=4) as pool:
        parsed=list(pool.map(parser.load_request,[(c,n) for c in CHUNKS for n in 'ABCD']))
    INPUTS.update(parser.INPUT_FILES)
    records=[v[0] for v in parsed]
    data={(r['chunk'],r['request']):r for r in records}
    ticks={(r['chunk'],r['request']):t for r,t,g in parsed}
    raw={}
    dispatch={}
    for c in CHUNKS:
        bench=RUN/f'node_23/chunk_{c}/bench'
        dispatch[c]={d['name']:d for d in read_json(bench/'dispatches.json')}
        assert set(dispatch[c])==set('ABCD')
        plan=read_json(bench/'plan.json')
        assert plan['rule']=='A20->B; B20->C; after A completes and C20->D; max three active; four total'
        for name in 'ABCD':
            raw[(c,name)]=read_json(bench/f'requests/{name}/result.json')
        config=read_json(RUN/f'node_23/chunk_{c}/before_bench/server_info.json')
        assert config['chunked_prefill_size']==c
        assert config['enable_mixed_chunk'] is False and config['prefill_decode_interval']==0
        assert config['speculative_algorithm'] is None
        assert config['cuda_graph_config']['decode']['backend']=='disabled'
    for name in 'ABCD':
        assert data[(8192,name)]['input_sha256']==data[(1024,name)]['input_sha256']
    events=[];traces={};interval_rows=[]
    for c in CHUNKS:
        for index,(old,new) in enumerate(PAIRS,1):
            t0=dispatch[c][new]['dispatch_perf_ns']
            assert not any(t0 < d['dispatch_perf_ns'] <= t0 + 14_000_000_000
                           for d in dispatch[c].values())
            t1=ticks[(c,new)][0]
            duration=(t1-t0)/1e9
            source_times=ticks[(c,old)]
            before=[(b-a)/1e9 for a,b in zip(source_times,source_times[1:]) if b<=t0][-20:]
            intersections=[(a,b,(b-a)/1e9) for a,b in zip(source_times,source_times[1:]) if a<t1 and b>t0]
            arrived=sum(t0<t<=t1 for t in source_times)
            active=[name for name in 'ABCD' if name!=new and
                    raw[(c,name)]['started_perf_ns']<t0<raw[(c,name)]['ended_perf_ns']]
            decoding=[name for name in active if ticks[(c,name)][0]<=t0]
            assert len(active)==len(decoding)==(1 if index==1 else 2)
            assert old in decoding and arrived==2 and len(intersections)==3
            baseline_speed=len(before)/sum(before)
            first_after=next(t for t in source_times if t>t1)
            r=data[(c,new)];o=data[(c,old)]
            event=dict(event=f'E{index}',chunk=c,observer=old,new_request=new,
                       existing_decode_count=len(decoding),existing_decode_requests=','.join(decoding),
                       dispatch_perf_ns=t0,new_first_perf_ns=t1,window_s=duration,
                       dispatch_to_client_start_ms=(raw[(c,new)]['started_perf_ns']-t0)/1e6,
                       new_queue_ms=r['queue_s']*1000,new_prefill_s=r['prefill_s'],
                       new_ttft_s=r['ttft_s'],new_tpot_ms=r['tpot_ms'],new_e2e_s=r['e2e_s'],
                       new_prefill_speed_tok_s=r['prefill_speed_tok_s'],
                       prior_itl_n=len(before),prior_itl_mean_ms=mean(before)*1000,
                       prior_itl_median_ms=median(before)*1000,
                       prior_reference_speed_tok_s=baseline_speed,
                       overlap_interval_n=len(intersections),overlap_max_itl_s=max(v[2] for v in intersections),
                       received_old_tokens_in_window=arrived,
                       old_window_receive_rate_tok_s=arrived/duration,
                       receive_rate_vs_prior_reference_pct=(arrived/duration/baseline_speed-1)*100,
                       old_first_output_after_new_first_s=(first_after-t1)/1e9,
                       old_whole_request_tpot_ms=o['tpot_ms'])
            events.append(event)
            # Keep integer-clock evidence in CSV and relative values in plotting JSON.
            offsets=[(t-t0)/1e9 for t in source_times]
            count_at_zero=sum(t<=t0 for t in source_times)
            trace={'arrival_offsets_s':offsets,'count_at_zero':count_at_zero,
                   'gap_start_s':[(a-t0)/1e9 for a,b in zip(source_times,source_times[1:])],
                   'gap_end_s':[(b-t0)/1e9 for a,b in zip(source_times,source_times[1:])],
                   'gap_seconds':[(b-a)/1e9 for a,b in zip(source_times,source_times[1:])],
                   'window_s':duration}
            traces[f'{c}-E{index}']=trace
            for n,(a,b) in enumerate(zip(source_times,source_times[1:]),1):
                interval_rows.append(dict(chunk=c,event=f'E{index}',observer=old,
                    from_output_token=n,to_output_token=n+1,start_offset_s=(a-t0)/1e9,
                    end_offset_s=(b-t0)/1e9,itl_s=(b-a)/1e9,
                    intersects_injection_window=a<t1 and b>t0))
    # Cross-check previously archived interference numbers, not generated charts.
    old_csv=RUN/'analysis/interference.csv';INPUTS.add(old_csv)
    with old_csv.open() as f: archived=list(csv.DictReader(f))
    for e in events:
        match=next(x for x in archived if int(x['chunk'])==e['chunk'] and x['decode_request']==e['observer'])
        assert abs(float(match['overlap_max_ms'])/1000-e['overlap_max_itl_s'])<1e-8
        assert int(match['overlap_count'])==e['overlap_interval_n']
    stats=[]
    for role,names in [('all_four','ABCD'),('new_incoming','BCD')]:
        for c in CHUNKS:
            rr=[data[(c,n)] for n in names]
            for key in ['ttft_s','tpot_ms']:
                values=[r[key] for r in rr]
                stats.append(dict(scope=role,chunk=c,metric=key,n=len(values),mean=mean(values),
                                  p95=nearest(values,.95),p99=nearest(values,.99),method='nearest_rank'))
    checks=dict(status='PASS',rounds_per_chunk=1,requests_per_chunk=4,total_raw_requests=8,
                injection_events_per_chunk=3,matched_event_positions=3,
                all_outputs_300=True,all_raw_output_ids_match=True,all_requests_http200=True,
                same_inputs_between_chunks=True,active_decode_counts=[1,2,2],
                old_tokens_received_per_injection_window=2,
                matches_archived_overlap_max_itl=True,
                local_reference_intervals=[19,19,20],
                independent_round_replicates=1)
    csv_out('requests.csv',records)
    csv_out('injection_events.csv',events)
    csv_out('aligned_itl_intervals.csv',interval_rows)
    csv_out('latency_summary.csv',stats)
    (OUT/'aligned_traces.json').write_text(json.dumps(traces,ensure_ascii=False)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'checks':checks,'source_run':str(RUN),
        'window':'client dispatch -> first output of newly injected request',
        'prior_reference':'up to 20 complete observer ITL intervals ending before injection',
        'statistics':stats,'events':events},ensure_ascii=False,indent=2)+'\n')
    return records,data,ticks,raw,dispatch,events,traces,stats,checks


def event_timeline(events,traces):
    fig=plt.figure(figsize=(14,15.2))
    grid=fig.add_gridspec(3,2,left=.075,right=.965,top=.865,bottom=.105,
                          hspace=.46,wspace=.18)
    fig.suptitle('新 Prefill 到来时，已有 Decode 发生了什么？',fontsize=20,fontweight='bold',y=.98)
    fig.text(.5,.948,'09-28 单轮数据 · 每档 4 请求 · 三次注入位置分别对齐 · 32K 输入 / 300 输出',
             ha='center',fontsize=11.8,color='#536579')
    fig.legend(handles=[Patch(facecolor='#F7D893',alpha=.6,label='新请求发起 → 新请求首 token（客户端窗口）'),
                        Line2D([0],[0],color='#475569',ls='--',label='发起时刻 t=0')],
               loc='upper center',bbox_to_anchor=(.5,.932),ncol=2,frameon=False,fontsize=10.5)
    for index,(old,new) in enumerate(PAIRS,1):
        for col,c in enumerate(CHUNKS):
            e=next(x for x in events if x['chunk']==c and x['event']==f'E{index}')
            tr=traces[f'{c}-E{index}'];t1=e['window_s']
            cell=grid[index-1,col].subgridspec(2,1,height_ratios=[1.08,1],hspace=.12)
            top=fig.add_subplot(cell[0])
            bottom=fig.add_subplot(cell[1],sharex=top)
            ts=np.array(tr['arrival_offsets_s']);counts=np.arange(1,len(ts)+1)-tr['count_at_zero']
            for ax in [top,bottom]:
                ax.axvspan(0,t1,color='#F7D893',alpha=.43,zorder=0)
                ax.axvline(0,color='#475569',ls='--',lw=1)
                ax.axvline(t1,color='#B88425',ls=':',lw=1.1)
                ax.set_xlim(-5,14)
                chart_style(ax,'both')
            top.step(ts,counts,where='post',color=COLORS[c],lw=1.6)
            top.set_ylim(-12,21)
            top.set_yticks([-10,0,10,20])
            top.set_title(f'E{index}  {old} 遇到 {new} · Chunk {c//1024}K · 原有 Decode {e["existing_decode_count"]} 条',
                          loc='left',fontsize=11.4,pad=11)
            top.text(.025,.94,f'新请求 Prefill {e["new_prefill_s"]:.3f}s；窗口 {t1:.3f}s',
                     transform=top.transAxes,va='top',fontsize=9.5,
                     bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':2.5})
            top.text(.98,.08,'窗口内旧请求仅收到 2 token',ha='right',transform=top.transAxes,
                     fontsize=9.5,color=COLORS[c],fontweight='bold')
            top.tick_params(labelbottom=False)
            if col==0:top.set_ylabel('旧请求累计输出\n相对 t=0（token）',fontsize=9.5)
            for a,b,gap in zip(tr['gap_start_s'],tr['gap_end_s'],tr['gap_seconds']):
                if b<-5 or a>14:continue
                overlap=a<t1 and b>0
                bottom.hlines(gap,max(-5,a),min(14,b),colors=COLORS[c],
                              lw=2.2 if overlap else 1.1,alpha=1 if overlap else .62)
                if -5<=b<=14:bottom.plot(b,gap,'o',color=COLORS[c],ms=2.1)
            bottom.axhline(e['prior_itl_median_ms']/1000,color='#66768A',ls=':',lw=.8)
            biggest=max([(g,a,b) for a,b,g in zip(tr['gap_start_s'],tr['gap_end_s'],tr['gap_seconds'])
                         if a<t1 and b>0])
            g,a,b=biggest
            bottom.text((a+b)/2,g+.32,f'{g:.3f} s',ha='center',color=COLORS[c],
                        fontsize=10.5,fontweight='bold')
            bottom.set_ylim(0,10.8)
            bottom.set_yticks([0,3,6,9])
            bottom.set_xticks([-5,0,5,10,14])
            bottom.set_xlabel('相对新请求发起时间（s）',fontsize=9.5)
            if col==0:bottom.set_ylabel('旧请求 ITL（s）',fontsize=9.5)
    fig.text(.075,.025,
        '上图阶梯横段 = 没有新输出到达。下图每条横线表示一个完整 ITL，横向覆盖这次停顿的起止时间。\n'
        '黄色区域包含排队、Prefill 和交付；服务端 Prefill 仅列持续时间，未强行映射到客户端时钟。\n'
        '越过黄色右边界的 ITL 仍计入相交窗口最大值。三次注入属于同一轮，不能当作三次独立重复。',
        fontsize=10.3,color='#607286',linespacing=1.7)
    save(fig,'01_注入对齐_输出进度与ITL')


def comparison_bars(events):
    fig,axes=plt.subplots(2,2,figsize=(13.4,9.2))
    fig.subplots_adjust(left=.09,right=.97,top=.81,bottom=.15,hspace=.47,wspace=.23)
    fig.suptitle('逐次注入：新请求延迟与旧请求停顿',fontsize=19,fontweight='bold',y=.975)
    fig.text(.5,.928,'每个柱子是一次注入观测；E1 原有 1 条 Decode，E2/E3 原有 2 条 Decode',
             ha='center',fontsize=11,color='#536579')
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=f'Chunk {c//1024}K') for c in CHUNKS],
               loc='upper center',bbox_to_anchor=(.5,.906),ncol=2,frameon=False)
    items=[('new_prefill_s','A. 新请求服务端 Prefill','用时（s）；越低越好',12,3),
           ('new_ttft_s','B. 新请求客户端 TTFT','用时（s）；越低越好',12,3),
           ('overlap_max_itl_s','C. 旧请求在相交窗口内的最大 ITL','停顿（s）；越低越好',12,3),
           ('old_window_receive_rate_tok_s','D. 受干扰窗口交付速率（含等待）','窗口内收到的旧输出 token/s',.43,3)]
    x=np.arange(3)
    for ax,(key,title,label,limit,digits) in zip(axes.flat,items):
        for j,c in enumerate(CHUNKS):
            vals=[e[key] for e in events if e['chunk']==c]
            bars=ax.bar(x+(-.18 if j==0 else .18),vals,width=.33,color=COLORS[c])
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,val+limit*.025,f'{val:.{digits}f}',
                        ha='center',fontsize=9.5,fontweight='bold',color=COLORS[c])
        ax.set_title(title,loc='left',pad=12)
        ax.set_xticks(x,['E1 A遇B','E2 B遇C','E3 C遇D'])
        ax.set_ylim(0,limit);ax.set_ylabel(label);chart_style(ax)
    fig.text(.09,.035,
        '窗口 = 新请求发起至其首 token；两档三个窗口中，观察的旧请求都恰好收到 2 token。\n'
        'D 图的速率差由窗口时长不同产生，表示这段时间的交付体验，不能当作独立的 Decode 算子吞吐。',
        fontsize=10.3,color='#607286',linespacing=1.7)
    save(fig,'02_逐事件_Prefill与Decode干扰')


def breakdown_means(data):
    keys=['queue_s','prefill_s','residual_s','after_first_s','e2e_s']
    rows=[]
    for c in CHUNKS:
        row={'chunk':c,'request_count':4,'included_requests':'A/B/C/D'}
        row.update({key:mean(data[(c,n)][key] for n in 'ABCD') for key in keys})
        assert abs(sum(row[key] for key,label,color in STAGES)-row['e2e_s'])<1e-8
        rows.append(row)
    return rows


def breakdown(data):
    fig,ax=plt.subplots(figsize=(13.4,4.6))
    fig.subplots_adjust(left=.095,right=.97,top=.78,bottom=.155)
    fig.suptitle('E2E Breakdown',fontsize=19,fontweight='bold',y=.975)
    display_labels={
        'queue_s':'Queue(几乎没有)',
        'prefill_s':'Prefill',
        'residual_s':'Mid(IPC主导)',
        'after_first_s':'Decode',
    }
    # Display labels follow the requested presentation; Mid remains an
    # unassigned residual. All four requests contribute equally to each mean.
    fig.legend(handles=[Patch(facecolor=color,label=display_labels[key])
                        for key,label,color in STAGES],
               loc='upper center',bbox_to_anchor=(.5,.914),ncol=4,
               frameon=False,fontsize=10.5,columnspacing=2)
    rows=breakdown_means(data)
    csv_out('e2e_breakdown_means.csv',rows)
    total_x=185
    for y,r in enumerate(rows):
        left=0
        for key,label,color in STAGES:
            val=r[key]
            ax.barh(y,val,left=left,height=.42,color=color)
            if key=='prefill_s':
                ax.text(.5,y-.29,f'{val:.3f}s',ha='left',va='bottom',
                        fontsize=10.5,color=color,fontweight='bold')
            elif key=='after_first_s':
                ax.text(left+val/2,y,f'{val:.3f}s',ha='center',va='center',
                        fontsize=11.5,color='white',fontweight='bold')
            left+=val
        ax.text(total_x,y,f'{r["e2e_s"]:.3f}s',ha='right',va='center',
                fontsize=11,color='#334155',fontweight='bold')
    ax.text(total_x,-.59,'Total',ha='right',va='bottom',fontsize=10,color='#475569')
    ax.set_yticks(range(2),[f'Chunk{r["chunk"]//1024}k' for r in rows])
    ax.set_ylim(1.55,-.8);ax.set_xlim(0,188)
    ax.set_xticks(range(0,176,25))
    ax.set_xlabel('E2E (s)')
    chart_style(ax,'x')
    save(fig,'03_新请求_E2E与Prefill分解')


def speeds(events):
    fig,axes=plt.subplots(2,1,figsize=(13.3,9.8))
    fig.subplots_adjust(left=.1,right=.965,top=.825,bottom=.13,hspace=.48)
    fig.suptitle('Prefill 处理速度，以及旧请求在注入期间的输出速度',fontsize=18.5,fontweight='bold',y=.975)
    fig.text(.5,.933,'上：服务端 Prefill 阶段；下：客户端在指定窗口收到的输出',
             ha='center',fontsize=11,color='#536579')
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=f'Chunk {c//1024}K') for c in CHUNKS]+
                       [Line2D([0],[0],marker='D',mfc='white',mec='#475569',color='none',label='注入前局部参考')],
               loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False,fontsize=10)
    x=np.arange(3)
    for j,c in enumerate(CHUNKS):
        rr=[e for e in events if e['chunk']==c];xs=x+(-.18 if j==0 else .18)
        vals=[e['new_prefill_speed_tok_s'] for e in rr]
        bars=axes[0].bar(xs,vals,width=.33,color=COLORS[c])
        for bar,val in zip(bars,vals):
            axes[0].text(bar.get_x()+bar.get_width()/2,val+45,f'{val:.1f}',ha='center',
                         fontsize=10,color=COLORS[c],fontweight='bold')
        vals=[e['old_window_receive_rate_tok_s'] for e in rr]
        bars=axes[1].bar(xs,vals,width=.33,color=COLORS[c])
        for bar,val in zip(bars,vals):
            axes[1].text(bar.get_x()+bar.get_width()/2,val+.06,f'{val:.3f}',ha='center',
                         fontsize=10,color=COLORS[c],fontweight='bold')
        refs=[e['prior_reference_speed_tok_s'] for e in rr]
        axes[1].scatter(xs,refs,marker='D',s=40,facecolor='white',edgecolor=COLORS[c],zorder=4,lw=1.4)
        for xx,val in zip(xs,refs):
            axes[1].text(xx,val+.075,f'{val:.2f}',ha='center',fontsize=9,color=COLORS[c])
    axes[0].set_title('A. 新请求 Prefill：13,184 个未命中输入 token / 服务端 Prefill 秒数',loc='left',pad=12)
    axes[0].set_ylim(0,2750);axes[0].set_ylabel('输入 token/s；越高越好')
    axes[1].set_title('B. 含停顿的窗口交付速率；注入前约 2 token/s（空心菱形）',loc='left',pad=12)
    axes[1].set_ylim(0,2.7);axes[1].set_ylabel('窗口内收到的输出 token/s（含等待）')
    for ax in axes:
        ax.set_xticks(x,['E1：A遇B','E2：B遇C','E3：C遇D']);chart_style(ax)
    fig.text(.1,.025,
        '空心菱形 = 注入前最近至多 20 个完整 ITL 的平均速率（实际为 19/19/20 个间隔）。\n'
        '它是同一请求的局部参考，不是另跑的无注入对照；柱子与菱形的观察窗口和统计方式不同。',
        fontsize=10.2,color='#607286',linespacing=1.7)
    save(fig,'04_Prefill速度与窗口Decode速度')


def round_overview(raw,dispatch):
    fig,axes=plt.subplots(2,1,figsize=(13.4,7.2),sharex=True)
    fig.subplots_adjust(left=.075,right=.97,top=.84,bottom=.095,hspace=.38)
    fig.suptitle('四请求时序',fontsize=18.5,fontweight='bold',y=.975)
    fig.legend(handles=[Patch(facecolor='#E4AB42',label='开始请求 → 首 token'),
                        Patch(facecolor='#718BB5',label='首 token → 请求结束')],
               loc='upper center',bbox_to_anchor=(.5,.937),ncol=2,frameon=False)
    for ax,c in zip(axes,CHUNKS):
        zero=dispatch[c]['A']['dispatch_perf_ns']
        for y,name in enumerate('ABCD'):
            r=raw[(c,name)]
            start=(r['started_perf_ns']-zero)/1e9
            first=(r['first_token_perf_ns']-zero)/1e9
            end=(r['ended_perf_ns']-zero)/1e9
            ax.barh(y,first-start,left=start,height=.52,color='#E4AB42')
            ax.barh(y,end-first,left=first,height=.52,color='#718BB5')
            ax.text(end+2,y,f'{end:.1f}s',va='center',fontsize=9,color='#475569')
        ax.set_yticks(range(4),list('ABCD'));ax.set_ylim(3.6,-.9)
        ax.set_title(f'Chunk{c//1024}k',loc='left',fontsize=11.5,pad=12)
        ax.set_xlim(0,350);chart_style(ax,'x')
        ax.set_xlabel('自 A 发起以来（s）')
    save(fig,'05_单轮四请求时序')


def write_report(records,data,events,stats,checks):
    asset='../chunk_redo_20260928/analysis/injection_focus_20260928'
    src='../chunk_redo_20260928/results/20260928_043035'
    s=[
        '# ChunkSize 单轮实验：以新 Prefill 注入时刻观察 Decode 干扰','',
        '整理日期：**2026-09-28（北京时间）**。数据来自今天的 `20260928_043035`：'
        '**Chunk 8K、1K 各一轮四条，合计八条正式请求**。本页按用户确认的单轮数据分析，'
        '重点是新 Prefill 到来时，新请求自身与正在输出的旧请求分别发生什么。','',
        '**速度口径先澄清：图中的约 0.3 token/s 是含长停顿的短窗口交付速率，不是正常 Decode 速度。** '
        '本轮注入前旧请求约 2.0–2.2 token/s；约 6.1 秒窗口只收到 2 token，才得到约 0.33 token/s。','',
        '**观察结论：本轮 8K 仍然更合适。** 三次注入中，新请求的 Prefill 都更短；'
        '旧请求在 8K 下通常出现两段约 3 秒的停顿，1K 下出现一段约 9.3–9.6 秒的长停顿。'
        '但两档在各自“注入→新首 token”窗口里，所观察的旧请求都只交付 **2 个输出 token**：'
        '8K 窗口约 6.1 秒，1K 约 10 秒。应同时看停顿形状、窗口长度和输出进度，不能只报一个最大 ITL。','',
        '## 1. 本轮怎么发生：三个事件不是三轮','',
        '| 注入事件 | 新请求 | 主要观察的旧请求 | 注入前已有 Decode | 触发条件 |',
        '|---|---|---|---|---|',
        '| E1：A 遇 B | B | A | A，共 1 条 | A 已输出至少 20 token，仍在 Decode |',
        '| E2：B 遇 C | C | B | A、B，共 2 条 | B 已输出至少 20 token |',
        '| E3：C 遇 D | D | C | B、C，共 2 条 | A 已完成，C 已输出至少 20 token 且仍在 Decode |','',
        '每条总输入 32,768，实际命中 19,584（59.765625%），实际新计算 **13,184 token**，输出 300。'
        '同一节点 23，TP16/EP16，MTP OFF，Graph 实际关闭，`enable_mixed_chunk=False`、`prefill_decode_interval=0`。'
        '两档复用同一输入，先跑 8K 后跑 1K。以上状态按归档配置核对。','',
        f'计划与实际发起时刻：[8K]({src}/node_23/chunk_8192/bench/dispatches.json)、'
        f'[1K]({src}/node_23/chunk_1024/bench/dispatches.json)；'
        f'实际配置：[8K]({src}/node_23/chunk_8192/before_bench/server_info.json)、'
        f'[1K]({src}/node_23/chunk_1024/before_bench/server_info.json)。','',
        '三个事件的背景 Decode 数不同，并且前一个事件会影响后续时间和状态；'
        '**比较应以相同事件位置的 8K 对 1K 为主，不把它们当作三次独立重复实验。**','',
        '## 2. 主图：把新请求等待与旧请求输出对齐','',
        f'![按三次注入对齐的旧请求输出进度与 ITL]({asset}/01_注入对齐_输出进度与ITL.png)','',
        '读图方法：','',
        '1. 每行对应一个注入事件，左列 8K、右列 1K；**横坐标 0 是新请求发起时刻**。',
        '2. 黄色区间从发起至客户端收到新请求首 token，包含客户端发送、服务端处理和交付，'
        '**不是单独测得的纯 Prefill 时段**。服务端 Prefill 只在图内列持续时间，没有映射到客户端时钟。',
        '3. 上半图是旧请求相对 t=0 的输出进度；阶梯长时间水平，表示这段时间没有新输出到达。'
        '负数表示注入前的输出位置，不是负的生成数量。',
        '4. 下半图一条水平线表示一个完整 ITL，线段横向覆盖该间隔的开始和结束，纵坐标为持续秒数。'
        '长横线代表一次长停顿，不是连续测到了多次相同停顿。',
        '5. 全部面板使用相同时间范围和 ITL 纵轴；注入前后局部范围为 −5～14 秒，'
        '没有把后续另一次新请求发起混进本次展示窗口。','',
        '**最值得注意的形状：**8K 的两段约 3 秒停顿之间还有一次输出到达；'
        '1K 在开始阶段短暂输出后，经历一段约 9–10 秒的连续停顿。'
        '最长间隔可能跨过黄色右边界，因此其完整长度包括新请求首 token 到达之后的一小段时间。','',
        '## 3. 逐事件指标：新请求和旧请求分开统计','',
        f'![逐事件 Prefill、TTFT、最大 ITL 与窗口输出速率]({asset}/02_逐事件_Prefill与Decode干扰.png)','',
        '| 事件 | Chunk | 新请求 Prefill（s） | 新请求 TTFT（s） | 发起→新首 token 窗口（s） | 旧请求相交最大 ITL（s） | 窗口旧输出数 | 窗口旧输出速率（token/s） |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for index in range(1,4):
        for c in CHUNKS:
            e=next(x for x in events if x['event']==f'E{index}' and x['chunk']==c)
            s.append(f'| {e["event"]} {e["observer"]}遇{e["new_request"]} | {c//1024}K | {e["new_prefill_s"]:.3f} | {e["new_ttft_s"]:.3f} | {e["window_s"]:.3f} | {e["overlap_max_itl_s"]:.3f} | {e["received_old_tokens_in_window"]} | {e["old_window_receive_rate_tok_s"]:.3f} |')
    s += ['', '三个指标不要混淆：','',
        '- **新请求 TTFT**从实际客户端请求开始计时；**注入窗口**从控制器提交请求计时。'
        '两者相差约 20–21 ms，属于控制器提交至实际请求开始的时间差；表中没有把两者写成一个值。',
        '- **相交最大 ITL**取所有与注入窗口相交的完整旧请求输出间隔中的最大值，'
        '不把跨边界间隔裁短。它不是整个旧请求的最大值；旧请求可能还遭遇后续注入。',
        '- **窗口输出速率**为 `窗口内实际收到的旧输出 token / 窗口秒数`。'
        '本轮分子均为 2，速率差直接来自窗口时长差，不能把它当作独立测得的纯 Decode 算子吞吐。','',
        '### 3.1 每个相交间隔，保留原始停顿结构','',
        '| 事件 | Chunk | 注入窗口相交的三个完整 ITL（s，按时间顺序） | 新首 token 后等到旧请求再输出（s） |',
        '|---|---|---|---:|',
    ]
    trace=json.loads((OUT/'aligned_traces.json').read_text())
    for index in range(1,4):
        for c in CHUNKS:
            e=next(x for x in events if x['event']==f'E{index}' and x['chunk']==c)
            t=trace[f'{c}-E{index}']
            vals=[g for a,b,g in zip(t['gap_start_s'],t['gap_end_s'],t['gap_seconds']) if a<e['window_s'] and b>0]
            s.append(f'| E{index} | {c//1024}K | '+ ' / '.join(f'{v:.3f}' for v in vals)+f' | {e["old_first_output_after_new_first_s"]:.3f} |')
    s += ['', '这些完整间隔可能包含窗口前、窗口后时间，因此不能将三个 ITL 直接相加后称作“窗口内的停顿总时间”。','',
        '## 4. TTFT/TPOT 的均值与 P95/P99：保留两种请求口径','',
        '**全四条 A/B/C/D**便于对照上一版；**后来注入的 B/C/D**更贴近本次问题。'
        '旧请求已经进入 Decode 时，它原先的 TTFT 已发生；新 Prefill 对它的影响主要看上面的局部 ITL 和输出进度。','',
        '| 请求口径 | 指标 | Chunk | n | 均值 | P95 | P99 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for scope,title in [('all_four','全部 A/B/C/D'),('new_incoming','后来 B/C/D')]:
        for metric,label,digits in [('ttft_s','TTFT（s）',3),('tpot_ms','TPOT（ms/token）',2)]:
            for c in CHUNKS:
                q=next(x for x in stats if x['scope']==scope and x['metric']==metric and x['chunk']==c)
                s.append(f'| {title} | {label} | {c//1024}K | {q["n"]} | {q["mean"]:.{digits}f} | {q["p95"]:.{digits}f} | {q["p99"]:.{digits}f} |')
    s += ['', 'TPOT = `(客户端 E2E − TTFT)/(300−1)`，是整条请求首 token 后的平均值，'
        '可能包含不止一次新请求注入的影响。**这里没有把“窗口秒数 / 收到的 2 个 token”叫 TPOT。**','',
        'P95/P99 按最近秩法 `ceil(n×p)` 取值；n=3 或 4 时二者均为样本最大值，'
        '**不能视为稳定的业务尾延迟**。B/C/D 的三个样本也不等于三次独立实验。','',
        '## 5. 全部四请求的平均 E2E Breakdown','',
        '**A/B/C/D 全部纳入，每个 Chunk 各 4 条请求；四个阶段分别取算术均值，图中只画 8K、1K 两条。** '
        'Total 为四条请求 E2E 的算术均值，不是整轮墙钟时长。','',
        f'![包括 A/B/C/D 的 Chunk 8K 和 1K 平均 E2E Breakdown]({asset}/03_新请求_E2E与Prefill分解.png)','',
        '图例按展示要求简写为 Queue(几乎没有)、Prefill、Mid(IPC主导)、Decode。'
        '**Mid 的数值仍是首 token 前未归因差值，IPC 主导是展示假设，尚未通过独立计时验证；'
        'Decode 的数值仍含首 token 后的等待、调度与传输。**','',
        '| 阶段 | 数据与解释 |', '|---|---|',
        '| Queue | 最终 SSE 的 `queue_time`，是服务端报告字段 |',
        '| Prefill | `prefill_finished_time − forward_entry_time`，服务端阶段区间，不是独立纯算子耗时 |',
        '| 首 token 前未归因 | `TTFT − Queue − Prefill`，不能命名为实测 IPC |',
        '| 首 token 后至结束 | `E2E − TTFT`，含生成、其他 Prefill 阻塞、调度、传输和收尾，不是纯 NPU Decode |','',
        '四段按定义相加为 E2E；这是使用服务端区间与客户端时长的**账面分解**，'
        '不是已经证明各阶段严格互斥的硬件执行时间线。图中顺序不说明所有未归因开销都发生在 Prefill 之后。','',
        '| Chunk | 请求数 | Queue 均值（ms） | Prefill 均值（s） | Mid 均值（ms） | Decode 均值（s） | Total 均值（s） |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in breakdown_means(data):
        s.append(f'| {r["chunk"]//1024}K | 4 | {r["queue_s"]*1000:.3f} | {r["prefill_s"]:.3f} | {r["residual_s"]*1000:.3f} | {r["after_first_s"]:.3f} | {r["e2e_s"]:.3f} |')
    s += ['', f'图表均值可复核：[e2e_breakdown_means.csv]({asset}/e2e_breakdown_means.csv)。']
    s += ['', '## 6. Prefill 速度与注入期间的 Decode 输出速度','',
        f'![新请求 Prefill 速度及旧请求注入前参考与窗口交付速度]({asset}/04_Prefill速度与窗口Decode速度.png)','',
        '- **Prefill 速度**：`13184 / 新请求服务端 Prefill 秒数`，分子只含实际未命中输入。',
        '- **旧请求窗口速度**：`窗口内实际收到的输出 token / 注入窗口秒数`，分母含整个新请求首 token 等待。',
        '- **注入前局部参考**：选旧请求在注入前已经完整结束的最近至多 20 个 ITL，'
        '速率为 `间隔数 / 这些间隔的时长之和`；E1/E2/E3 实际分别有 19/19/20 个。'
        '**它不是另跑的无注入对照**，与窗口交付速率的观察时长和计算方式也不同。','',
        '| 事件 | Chunk | 新请求 Prefill（token/s） | 注入前参考 ITL 数 | 注入前参考速度（token/s） | 注入窗口旧输出速度（token/s） |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for index in range(1,4):
        for c in CHUNKS:
            e=next(x for x in events if x['event']==f'E{index}' and x['chunk']==c)
            s.append(f'| E{index} | {c//1024}K | {e["new_prefill_speed_tok_s"]:.2f} | {e["prior_itl_n"]} | {e["prior_reference_speed_tok_s"]:.3f} | {e["old_window_receive_rate_tok_s"]:.3f} |')
    s += ['', '图中空心菱形在约 2 token/s，窗口柱形只有约 0.19–0.33 token/s，'
        '直接体现新 Prefill 到来时旧请求交付输出变稀疏。它不意味着 Decode 算子本身只剩这点计算速度。','',
        '### 6.1 完整请求的 Decode 平均速度仍可作为补充','',
        '为与原来的 TPOT 图对应，下面保留 `1000/TPOT(ms)`，即 `299/(E2E−TTFT)`。'
        '这是完整请求的客户端观察速度，不能全部归因到某一次注入。','',
        '| 请求 | 8K 整请求首 token 后速度（token/s） | 1K 整请求首 token 后速度（token/s） |',
        '|---|---:|---:|',
    ]
    for n in 'ABCD':
        s.append(f'| {n} | {data[(8192,n)]["decode_speed_tok_s"]:.4f} | {data[(1024,n)]["decode_speed_tok_s"]:.4f} |')
    s += ['', '## 7. 放回整轮时间线，检查背景条件','',
        f'![两档单轮四请求的生命周期与三次注入]({asset}/05_单轮四请求时序.png)','',
        '这张图用于解释样本结构：E1 只有 A 在 Decode；E2 已有 A/B；E3 时 A 已结束，仍有 B/C。'
        '因此 E1 是最直接的“一个 Decode 遇到一个新 Prefill”观察，E2/E3 是已有两个 Decode 时的补充。'
        '同一事件位置在两档的已有请求数量一致，但两档的实际发起时刻、输出上下文进度并不完全相同。','',
        '## 8. 本轮能得出的判断与边界','',
        '1. **8K 的新请求 Prefill 更短，旧请求最长连续停顿也更短。** 三个事件位置都给出相同方向。',
        '2. **平均 TPOT 会稀释短时干扰。** 这次判断更应看按注入时刻对齐的输出进度、完整 ITL 与新请求 Prefill。',
        '3. **两个约 3 秒停顿与一个约 9.6 秒停顿，要连同整个等待窗口理解。** '
        '本轮两档窗口内都只收到 2 个旧输出；8K 更早结束这段等待，不代表它没有 Prefill 阻塞。',
        '4. **只有每档一轮。** 不能推断重复实验的方差或稳定 P99；三次注入有相关性，不能扩充成三轮。',
        '5. 本轮没有独立的 Decode-only 控制组；注入前参考不是无干扰反事实。'
        '随机 token、强制 300 输出、Graph 关闭、mixed chunk 关闭限制了外推范围。',
        '6. A/C/D 在两档的最终输出 ID 不同，只有 B 完全相同；同输入与输出长度相同不等于输出内容完全相同。','',
        '### 8.1 一般应选什么调度方式','',
        '若目标是在线对话的输出流畅性，我建议优先保障已有 Decode，再用剩余 token/计算预算安排 Prefill chunk，'
        '或使用能将 Decode 与小段 Prefill 混合的调度。这不是机械地规定每一块后只能执行一次 Decode；'
        '应同时约束已有请求 ITL、新请求 TTFT 与整体吞吐。若负载主要是批量输入处理、几乎没有活跃 Decode，'
        '较大的 Prefill 批次或更偏 Prefill 的策略可以作为另一种目标下的选择。此处是调优建议，未在本轮做交错调度 A/B。','',
        '### 8.2 A 是否完全不动：调度、执行与交付要分开','',
        '归档配置是 `enable_mixed_chunk=False`、`prefill_decode_interval=0`。固定镜像的 '
        '[scheduler.py](../mtp_full_20260928/results/20260928_061159/source_snapshot/served_image_sources/scheduler.py) '
        '在约 3305 行优先选择存在的新 Prefill batch，约 3486 行继续处理未完成的 chunked 请求。'
        '这解释了为什么不能期待每个 1K 块后自动安排旧请求 Decode。','',
        '但不能把它加强成“从 B 发起到首 token 的整个窗口，A 连一个 token 都不会动”。'
        '原始 SSE 在每个窗口内确实收到观察对象的 2 个 token；同时 `disable_overlap_schedule=False`，'
        '源码约 1832 行的 overlap loop 会交错提交当前 batch 与处理前一 batch 结果。'
        '因此窗口内到达的 token 可能涉及已提交计算、结果队列与交付延迟，不能仅按到达时刻认定它在该时刻执行 Decode。'
        '现有日志 `decode_log_interval=40`，也不是逐 Decode step 的完整轨迹；不能凭中间缺少 Decode 日志就证明完全没有执行。','',
        '### 8.3 1K 为什么更慢：已确认的变化与尚未定位的根因','',
        '四条请求的服务日志均确认：8K 是 `8192 + 4992`，共 2 个 Prefill batch；'
        '1K 是 `12 × 1024 + 896`，共 13 个 Prefill batch。新计算总量都为 13,184 token。'
        '模型配置摘要与当时归档一致，模型有 78 层，MoE 路由专家数为 256、每 token 选择 8 个。'
        f'原始批次证据和配置核对保存在 [scheduler_evidence.json]({asset}/scheduler_evidence.json)。','',
        '**两个原因链条应分开：调度优先级解释旧请求为何长时间没有连续输出；'
        'Prefill 执行效率解释相同输入为何总用时不同。** 更多小批次意味着更多次跨模型层的批次执行；'
        '固定的调度、kernel launch、同步/通信成本以及小矩阵效率损失，是需要验证的候选原因。'
        'Graph 实际关闭、TP16/EP16、小规模每专家 token 批量，使这些方向值得优先检查，'
        '但本轮没有 kernel/通信/CPU 的 profiler breakdown，不能给任何一项分配确定的秒数或宣称根因已定位。','',
        '只按新注入的 B/C/D 看，1K Prefill 为 9.525–9.803 秒，8K 为 5.637–5.659 秒，'
        '耗时约为 1.68–1.74 倍；全四条平均为 1.728 倍，即多约 72.8%，不是精确两倍。'
        '1K 不是把整份输入重算十三遍；每个 chunk 只推进其新增输入，但要走相应的模型计算与调度路径。','',
        '### 8.4 0.3 token/s 为什么看起来离谱','',
        '以 E1 为例，8K 的窗口为 6.117528 秒、观察的 A 收到 2 token，故为 `2 / 6.117528 = 0.326929 token/s`；'
        '1K 为 `2 / 10.285293 = 0.194452 token/s`。这个数是真实窗口交付率，但其中大部分时间是等待。'
        '注入前参考约 2.0–2.2 token/s，完整请求首 token 后均速约 1.84–2.12 token/s；'
        '三个量不能混用，任何一个单请求值也不是多请求服务的总输出吞吐。','',
        '约 2 token/s 的基线是否还有优化空间是另一个问题。当前是 78 层 MoE、TP16/EP16、MTP OFF、'
        'Graph disabled、DeepEP normal、1–3 条活跃请求；这些是可确认的运行条件，'
        '并不能单凭配置判断哪项吃掉了最多时间。需要用无注入基线和算子/通信/调度分析来定位，'
        '不能把这个低速直接说成硬件正常上限，或保证单开某个开关就会达到某个速度。','',
        '## 9. 文件、导出与复算','',
        f'五张图均有 PNG、SVG，并汇总成 [五页 PDF]({asset}/ChunkSize_单轮注入干扰_五图.pdf)。','',
        '| 图 | PNG | SVG |','|---|---|---|',
    ]
    for name,label in [('01_注入对齐_输出进度与ITL','主图：按注入对齐'),
                       ('02_逐事件_Prefill与Decode干扰','逐事件指标'),
                       ('03_新请求_E2E与Prefill分解','四请求平均 E2E / Prefill'),
                       ('04_Prefill速度与窗口Decode速度','阶段速度'),
                       ('05_单轮四请求时序','整轮时序')]:
        s.append(f'| {label} | [PNG]({asset}/{name}.png) | [SVG]({asset}/{name}.svg) |')
    s += ['', f'数值明细：[注入事件 CSV]({asset}/injection_events.csv)、[全部请求 CSV]({asset}/requests.csv)、'
        f'[TTFT/TPOT 汇总]({asset}/latency_summary.csv)、[按注入对齐的全部 ITL]({asset}/aligned_itl_intervals.csv)、'
        f'[统计与校验]({asset}/summary.json)。','',
        f'生成代码：[plot_injection_report.py]({asset}/plot_injection_report.py)，'
        f'复用[原始 SSE 校验解析器](../chunk_redo_20260928/analysis/prefill_comparison_20260928/plot_report.py)。'
        f'输入文件与代码摘要见 [input_manifest.json]({asset}/input_manifest.json)。','',
        '```bash\npython3 /home/chengyetai/workspace/current_exp/chunk_redo_20260928/analysis/injection_focus_20260928/plot_injection_report.py\n```','',
        '本次离线复核八条原始 SSE 与最终输出 ID；每条 300 个逐 token 事件，HTTP 200，'
        '实际命中 19,584，输出长度完整；跨档输入 SHA 一致。'
        '六个注入窗口均核查已有 Decode 数、实际到达 token 数、边界相交间隔和归档最大 ITL，结果一致。'
        '脚本读取归档数据生成图表，不调用在线服务。','',
    ]
    DOC.write_text('\n'.join(s),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records,data,ticks,raw,dispatch,events,traces,stats,checks=derive()
    font_file=parser.configure_fonts()
    event_timeline(events,traces)
    comparison_bars(events)
    breakdown(data)
    speeds(events)
    round_overview(raw,dispatch)
    with PdfPages(OUT/'ChunkSize_单轮注入干扰_五图.pdf') as pdf:
        for name,fig in FIGURES:pdf.savefig(fig,bbox_inches='tight')
    for name,fig in FIGURES:plt.close(fig)
    write_report(records,data,events,stats,checks)
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(INPUTS)]
    (OUT/'input_manifest.json').write_text(json.dumps({'run_id':RUN.name,
        'font':font_file,'inputs':files,
        'code':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in [Path(__file__).resolve(),PARSER]]},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(DOC),'figures':[name for name,fig in FIGURES],'checks':checks},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
