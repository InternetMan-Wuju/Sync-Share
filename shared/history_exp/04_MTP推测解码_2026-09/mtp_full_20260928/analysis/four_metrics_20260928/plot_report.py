#!/usr/bin/env python3
"""Read archived MTP streams; report acceptance, output throughput, TTFT, TPOT.

Window throughput and latency retain the original strict 2700s cutoff policy.
Acceptance is a ratio of cumulative counters, not an average of live gauges.
No model/server calls are made. Figures contain only titles, axes, and data.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'mtp_full_20260928/results/20260928_061159'
DOC=ROOT/'doc/MTP_接受长度_吞吐_TTFT_TPOT对比_2026-09-28.md'
SCENES=['decode','mixed','prefix']
LABELS=['1K / 32K目标','8K / 1K','32K / 300\n缓存59.77%']
NAMES={'decode':'1K 输入 / 32K 目标输出','mixed':'8K 输入 / 1K 输出','prefix':'32K 输入 / 300 输出，缓存命中 59.77%'}
LENGTHS={'decode':(1024,32768),'mixed':(8192,1024),'prefix':(32768,300)}
COLORS={'off':'#2563A6','on':'#DF7D24'}
INPUTS=set();FIGURES=[]


def load(p):
    INPUTS.add(p)
    return json.loads(p.read_text())


def quantile(v,q):
    return sorted(v)[math.ceil(len(v)*q)-1] if len(v)>=3 else None


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def inspect_request(task):
    scene,mode,r,stop=task
    folder=RUN/scene/mode/f'request_{r["case_index"]:04d}'/'measurement'
    path=folder/'sse_raw.jsonl.gz';INPUTS.add(path)
    window_n=0;first=None;last=None;full_n=0;done=0;last_ids=None
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            e=json.loads(line);raw=e.get('raw_line','')
            if not raw.startswith('data:'):continue
            raw=raw[5:].strip()
            if raw=='[DONE]':done+=1;continue
            obj=json.loads(raw);m=obj.get('meta_info') or {};n=m.get('completion_tokens')
            if not isinstance(n,int):continue
            full_n=max(full_n,n);last_ids=obj.get('output_ids')
            if e['recv_perf_ns']<=stop and n>window_n:
                window_n=n
                if first is None:first=e['recv_perf_ns']
                last=e['recv_perf_ns']
    assert r['http_status']==200 and r['error'] is None
    assert full_n==r['actual_output']
    ids=load(folder/'output_ids_final_or_partial.json')
    assert ids==last_ids and len(ids)==full_n
    assert r['prompt_tokens']==LENGTHS[scene][0] and r['requested_output']==LENGTHS[scene][1]
    assert r['cached_tokens']==(19584 if scene=='prefix' else 0)
    assert r['input_sha256']==r['expected_input_sha256']
    if first is not None:assert abs((first-r['started_perf_ns'])/1e9-r['ttft_s'])<1e-8
    boundary=r['started_perf_ns']>=stop-3_000_000_000 and full_n<r['requested_output']
    valid_latency=first is not None and not boundary
    tpot=(last-first)/1e6/(window_n-1) if first is not None and window_n>1 else None
    meta=r['meta_info_last'];verify=meta.get('spec_verify_ct') or 0
    accepted=meta.get('spec_num_correct_drafts') or 0;proposed=meta.get('spec_num_proposed_drafts') or 0
    if verify:
        assert proposed==verify*3
        hist=meta['spec_correct_drafts_histogram']
        assert sum(hist)==verify and sum(i*n for i,n in enumerate(hist))==accepted
        assert abs(full_n/verify-meta['spec_accept_length'])<1e-8
    return dict(scene=scene,mode=mode,index=r['case_index'],input_sha256=r['input_sha256'],
        target_output=r['requested_output'],actual_output=full_n,window_output=window_n,
        complete=full_n==r['requested_output'] and done==1,
        boundary_tail=boundary,latency_included=valid_latency,
        ttft_s=r['ttft_s'],window_tpot_ms=tpot,
        generation_span_s=(last-first)/1e9 if first is not None and window_n>1 else 0,
        verify_count=verify,accepted_drafts=accepted,proposed_drafts=proposed,
        observed_tpot_ms=r['observed_tpot_ms'],e2e_s=r['e2e_s'])


def derive():
    tasks=[];summaries={};schedules={}
    for scene in SCENES:
        sched=load(RUN/scene/'schedule.json');assert sched['duration_s']==2700
        schedules[scene]=sched
        for mode in ['off','on']:
            rows=load(RUN/scene/mode/'completed_or_partial_results.json')
            tasks += [(scene,mode,r,int(sched['stop_perf_s']*1e9)) for r in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(inspect_request,tasks))
    assert len(rows)==83
    summary_rows=[];percentiles=[]
    for scene in SCENES:
        summaries[scene]={}
        archived=load(RUN/'analysis'/f'{scene}_summary.json')
        for mode in ['off','on']:
            rr=[r for r in rows if r['scene']==scene and r['mode']==mode]
            lat=[r for r in rr if r['latency_included']]
            acc=[r for r in rr if r['verify_count']>0]
            verify=sum(r['verify_count'] for r in acc)
            tokens=sum(r['window_output'] for r in rr)
            accept_tokens=sum(r['actual_output'] for r in acc)
            covered=sum(r['window_output'] for r in acc)
            generated=sum(max(0,r['window_output']-1) for r in lat)
            gen_seconds=sum(r['generation_span_s'] for r in lat)
            stats=dict(scene=scene,mode=mode,window_seconds=2700,requests=len(rr),
                complete_requests=sum(r['complete'] for r in rr),
                partial_requests=sum(not r['complete'] for r in rr),
                latency_sample_count=len(lat),observed_tokens=tokens,
                window_output_tokens_per_s=tokens/2700,
                observed_generation_tokens_per_s=generated/gen_seconds,
                mean_ttft_s=mean(r['ttft_s'] for r in lat),
                mean_observed_tpot_ms=mean(r['window_tpot_ms'] for r in lat),
                spec_verify_ct=verify,acceptance_covered_output_tokens=accept_tokens,
                acceptance_covered_window_tokens=covered,
                acceptance_output_coverage_ratio=covered/tokens if mode=='on' else None,
                weighted_accept_length=accept_tokens/verify if verify else None,
                weighted_draft_accept_rate=sum(r['accepted_drafts'] for r in acc)/sum(r['proposed_drafts'] for r in acc) if verify else None)
            for key,value in stats.items():
                if key in archived[mode] and value is not None:
                    assert abs(value-archived[mode][key])<1e-7,(scene,mode,key,value,archived[mode][key])
            summaries[scene][mode]=stats;summary_rows.append(stats)
            for key,label in [('ttft_s','TTFT(s)'),('window_tpot_ms','TPOT(ms/token)')]:
                vals=[r[key] for r in lat]
                percentiles.append(dict(scene=scene,mode=mode,metric=label,n=len(vals),mean=mean(vals),
                    p50=quantile(vals,.5),p90=quantile(vals,.9),p95=quantile(vals,.95),p99=quantile(vals,.99)))
    paired=[]
    archived_pair=load(RUN/'analysis/paired_summary.json')
    for scene in SCENES:
        modes={mode:{r['index']:r for r in rows if r['scene']==scene and r['mode']==mode} for mode in ['off','on']}
        shared=sorted(modes['off'].keys() & modes['on'].keys())
        pairs=[]
        for i in shared:
            a,b=modes['off'][i],modes['on'][i]
            assert a['input_sha256']==b['input_sha256']
            if a['complete'] and b['complete']:pairs.append((a,b))
        assert len(pairs)==archived_pair[scene]['complete_pairs']
        if pairs:
            stat=dict(scene=scene,n=len(pairs),off_ttft_s=mean(a['ttft_s'] for a,b in pairs),
                      on_ttft_s=mean(b['ttft_s'] for a,b in pairs),
                      off_tpot_ms=mean(a['observed_tpot_ms'] for a,b in pairs),
                      on_tpot_ms=mean(b['observed_tpot_ms'] for a,b in pairs))
            for mode in ['off','on']:
                for key in ['ttft_s','tpot_ms']:
                    assert abs(stat[f'{mode}_{key}']-archived_pair[scene][f'complete_pair_{mode}_mean_{key}'])<1e-8
            paired.append(stat)
    write_csv('summary.csv',summary_rows);write_csv('requests.csv',rows)
    write_csv('latency_percentiles.csv',percentiles);write_csv('paired_latencies.csv',paired)
    result={'source_run':str(RUN),'checks':{'status':'PASS','raw_requests':83,'source_scenes':3,
            'all_outputs_match_final_ids':True,'all_derived_means_match_archived_summaries':True,
            'all_same_input_complete_pairs_verified':True},'groups':summaries,
            'paired_complete_latencies':paired,'percentiles':percentiles}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def fonts():
    p=Path('/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf')
    font_manager.fontManager.addfont(str(p))
    name=font_manager.FontProperties(fname=str(p)).get_name()
    plt.rcParams.update({'font.family':['DejaVu Sans',name],'font.size':10.5,
        'axes.unicode_minus':False,'axes.titleweight':'bold','axes.labelcolor':'#334155',
        'axes.edgecolor':'#D3DBE4','text.color':'#172B42','xtick.color':'#475569',
        'ytick.color':'#475569','figure.facecolor':'white','axes.facecolor':'white',
        'svg.fonttype':'path','pdf.fonttype':42})
    warnings.filterwarnings('error',message='Glyph .* missing from font')


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E4EAF0',lw=.7)
    ax.set_xticks(np.arange(3),LABELS,fontsize=10.2)


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    FIGURES.append((name,fig))


def paired_bars(ax,groups,key,limit,digits):
    x=np.arange(3)
    for j,mode in enumerate(['off','on']):
        vals=[groups[s][mode][key] for s in SCENES]
        bars=ax.bar(x+(-.18 if j==0 else .18),vals,width=.33,color=COLORS[mode])
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,v+limit*.023,f'{v:.{digits}f}',
                    ha='center',va='bottom',fontsize=10.5,fontweight='bold',color=COLORS[mode])
    ax.set_ylim(0,limit);style(ax)


def plot_accept_throughput(groups):
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.1))
    fig.subplots_adjust(left=.075,right=.97,top=.78,bottom=.17,wspace=.24)
    fig.suptitle('MTP 接受长度与输出吞吐',fontsize=19,fontweight='bold',y=.98)
    fig.legend(handles=[Patch(facecolor=COLORS[m],label=f'MTP {m.upper()}') for m in ['off','on']],
               loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=11)
    ax=axes[0];values=[groups[s]['on']['weighted_accept_length'] for s in SCENES]
    bars=ax.bar(np.arange(3),values,width=.48,color=COLORS['on'])
    for bar,v in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,v+.09,f'{v:.3f}',ha='center',fontweight='bold',color=COLORS['on'],fontsize=11)
    ax.set_ylim(0,4.65);ax.set_title('接受长度（MTP ON）',loc='left',fontsize=12,pad=13)
    ax.set_ylabel('输出 token / verify（含 bonus）');style(ax)
    paired_bars(axes[1],groups,'window_output_tokens_per_s',10.1,3)
    axes[1].set_title('输出吞吐',loc='left',fontsize=12,pad=13)
    axes[1].set_ylabel('窗口总输出 / 2700s（token/s）')
    for x,s in enumerate(SCENES):
        a,b=groups[s]['off'],groups[s]['on'];ratio=b['observed_tokens']/a['observed_tokens']
        axes[1].text(x,max(a['window_output_tokens_per_s'],b['window_output_tokens_per_s'])+1.05,
                     f'{ratio:.2f}×',ha='center',fontweight='bold',fontsize=11,color='#334155')
    save(fig,'01_接受长度与输出吞吐')


def plot_latencies(groups):
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.1))
    fig.subplots_adjust(left=.075,right=.97,top=.78,bottom=.17,wspace=.24)
    fig.suptitle('MTP TTFT / TPOT',fontsize=19,fontweight='bold',y=.98)
    fig.legend(handles=[Patch(facecolor=COLORS[m],label=f'MTP {m.upper()}') for m in ['off','on']],
               loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=11)
    paired_bars(axes[0],groups,'mean_ttft_s',7.6,3)
    axes[0].set_title('TTFT mean',loc='left',fontsize=12,pad=13);axes[0].set_ylabel('首 token 等待（s）')
    paired_bars(axes[1],groups,'mean_observed_tpot_ms',560,2)
    axes[1].set_title('TPOT mean',loc='left',fontsize=12,pad=13);axes[1].set_ylabel('平均每输出 token 时间（ms/token）')
    save(fig,'02_TTFT与TPOT')


def document(result):
    g=result['groups'];asset='../mtp_full_20260928/analysis/four_metrics_20260928'
    source='../mtp_full_20260928/results/20260928_061159'
    s=['# MTP：接受长度、对应吞吐、TTFT、TPOT','',
       '整理日期：**2026-09-28**。本页仅围绕这四项比较 MTP OFF / ON，'
       '数据为 `20260928_061159` 的三种负载，各模式每场景测量 **2,700 秒（45 分钟）**。'
       '图表展示各场景汇总值，不逐条铺开请求。','',
       '**主要结果：**三个场景 ON 接受长度为 **3.998 / 3.459 / 2.767**，'
       '对应窗口输出吞吐为 OFF 的 **3.706× / 3.148× / 2.224×**。'
       'TPOT 均明显降低；32K 输入 / 300 输出场景的 TTFT 则从 **5.396 s 升到 6.411 s**。','',
       '## 1. 接受长度与输出吞吐','',
       f'![MTP 接受长度与输出吞吐]({asset}/01_接受长度与输出吞吐.png)','',
       '| 每请求输入 / 目标输出 | ON 接受长度 | OFF 吞吐（token/s） | ON 吞吐（token/s） | ON / OFF |',
       '|---|---:|---:|---:|---:|']
    for scene in SCENES:
        a,b=g[scene]['off'],g[scene]['on']
        s.append(f'| {NAMES[scene]} | {b["weighted_accept_length"]:.3f} | {a["window_output_tokens_per_s"]:.3f} | {b["window_output_tokens_per_s"]:.3f} | {b["observed_tokens"]/a["observed_tokens"]:.3f}× |')
    s += ['', '**接受长度是“每次 verify 平均推进多少输出 token”，包含 bonus token。** '
          '本配置草稿步骤为 3、验证槽位为 4；接受长度接近 4 表示一次验证通常推进接近 4 个 token。'
          'OFF 没有这个 MTP 指标，所以用“—”，没有人为填 0 或 1。','',
          '接受长度高，可以减少推进同样输出长度所需的验证次数；但它不等于实际加速倍数。'
          '草稿与验证计算、Prefill、缓存预热及请求间隙都有成本。三个场景的接受长度与吞吐倍率同向变化，'
          '只是这批不同负载的对应观察，不能据此证明接受长度独自决定吞吐。','',
          '## 2. TTFT 与 TPOT','',
          f'![MTP OFF 与 ON 的 TTFT 和 TPOT 均值]({asset}/02_TTFT与TPOT.png)','',
          '| 场景 | OFF TTFT（s） | ON TTFT（s） | ON TTFT 变化 | OFF TPOT（ms） | ON TPOT（ms） | TPOT 降幅 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        a,b=g[scene]['off'],g[scene]['on']
        s.append(f'| {NAMES[scene]} | {a["mean_ttft_s"]:.3f} | {b["mean_ttft_s"]:.3f} | {(b["mean_ttft_s"]/a["mean_ttft_s"]-1)*100:+.2f}% | {a["mean_observed_tpot_ms"]:.2f} | {b["mean_observed_tpot_ms"]:.2f} | {(1-b["mean_observed_tpot_ms"]/a["mean_observed_tpot_ms"])*100:.2f}% |')
    s += ['', '以上是各模式在整个窗口中实际获得的有效请求样本的算术均值，含有足够输出的尾部截断请求。'
          'ON 更快，覆盖的输入样本更多，因此两模式的样本构成并不完全相同。','',
          '**相同输入且双方都完整输出的补充核对：**','',
          '| 场景 | 完整配对数 | OFF / ON TTFT 均值（s） | OFF / ON TPOT 均值（ms） |',
          '|---|---:|---:|---:|']
    for p in result['paired_complete_latencies']:
        s.append(f'| {NAMES[p["scene"]]} | {p["n"]} | {p["off_ttft_s"]:.3f} / {p["on_ttft_s"]:.3f} | {p["off_tpot_ms"]:.2f} / {p["on_tpot_ms"]:.2f} |')
    s += ['', '8K/1K 的全窗口 TTFT 是 ON 小幅降低，但同输入完整配对为 **3.706 → 3.754 s**，'
          '因此不能宣布该负载下 MTP 稳定改善 TTFT。两种统计都支持后续生成的 TPOT 明显降低。'
          '1K/32K 没有双方完成目标长度的请求，不提供完整请求配对结论。','',
          '## 3. 三个场景的 OFF / ON 表','',
          '下面明确区分“每请求目标输出”和“45 分钟内全部测量请求合计输出”，吞吐分子使用后者。','']
    for scene in SCENES:
        a,b=g[scene]['off'],g[scene]['on'];isl,osl=LENGTHS[scene]
        length_label={1024:'1K',8192:'8K',32768:'32K'}
        il=length_label[isl];ol=length_label.get(osl,str(osl))
        s += [f'### {NAMES[scene]}','', '| **指标** | **MTP OFF** | **MTP ON** |', '| :---: | :---: | :---: |',
              f'| 每请求输入 / 目标输出 | {il} / {ol} | {il} / {ol} |',
              '| 客户端并发 | 1 | 1 |','| 测量墙钟 | 2700s | 2700s |',
              f'| 完整 / 截断请求数 | {a["complete_requests"]} / {a["partial_requests"]} | {b["complete_requests"]} / {b["partial_requests"]} |',
              f'| 窗口总输出 | {a["observed_tokens"]:,} token | {b["observed_tokens"]:,} token |',
              f'| 接受长度（含 bonus） | — | {b["weighted_accept_length"]:.3f} |',
              f'| 输出吞吐 | {a["window_output_tokens_per_s"]:.3f} tok/s | {b["window_output_tokens_per_s"]:.3f} tok/s |',
              f'| 吞吐相对 OFF | 1.000× | {b["observed_tokens"]/a["observed_tokens"]:.3f}× |',
              f'| TTFT mean | {a["mean_ttft_s"]:.3f}s | {b["mean_ttft_s"]:.3f}s |',
              f'| TPOT mean | {a["mean_observed_tpot_ms"]:.2f}ms | {b["mean_observed_tpot_ms"]:.2f}ms |',
              f'| 有效延迟样本数 | {a["latency_sample_count"]} | {b["latency_sample_count"]} |',
              f'| 接受计数覆盖窗口输出 | — | {b["acceptance_output_coverage_ratio"]*100:.2f}% |','']
    s += ['## 4. 四项指标的精确定义','',
          '| 指标 | 计算方法 | 说明 |','|---|---|---|',
          '| 接受长度 | 有终态累计计数的请求：`Σ实际输出token / Σspec_verify_ct` | 按 verify 次数加权，含 bonus；不是平均瞬时 Gauge，也不是简单平均每请求接受长度 |',
          '| 输出吞吐 | 截止前实际收到的测量输出 token / 2700秒 | 分母含清缓存、前缀预热、Prefill 和请求间隙；分子不含预热请求输出 |',
          '| TTFT | 请求开始到首次输出到达客户端 | 各有效请求的算术均值 |',
          '| TPOT | `(窗口内最后新增输出到达−首次输出到达)/(窗口输出token−1)` | 先按请求计算，再对有效请求求均值；MTP 一帧可包含多个 token，不能拿帧间隔代替逐 token ITL |','',
          '接受计数的实际分子/分母：','',
          '| 场景 | 有接受计数的输出 / verify | 窗口输出计数覆盖 |', '|---|---:|---:|']
    for scene in SCENES:
        r=g[scene]['on']
        s.append(f'| {NAMES[scene]} | {r["acceptance_covered_output_tokens"]:,} / {r["spec_verify_ct"]:,} | {r["acceptance_covered_window_tokens"]:,} / {r["observed_tokens"]:,} |')
    s += ['', '8K/1K 的尾部截断请求缺最终累计接受计数，所以接受长度覆盖 97.66% 的窗口输出；'
          '这部分输出仍计入实际吞吐，没有伪造它的接受统计。截止后才到达的 token 不进入窗口吞吐。'
          '截止前最后 3 秒才开始且未完成的极短补发请求不进入延迟均值。','',
          '为避免把两种“速度”混用，生成期间的观察速率另列如下，**主图使用上面的 45 分钟输出吞吐**：','',
          '| 场景 | OFF 生成期间速率（token/s） | ON 生成期间速率（token/s） |', '|---|---:|---:|']
    for scene in SCENES:
        s.append(f'| {NAMES[scene]} | {g[scene]["off"]["observed_generation_tokens_per_s"]:.3f} | {g[scene]["on"]["observed_generation_tokens_per_s"]:.3f} |')
    s += ['', '此补充速率为各请求 `输出数−1` 总和 / 首末输出间隔之和，排除了请求之间的清缓存和预热等时间；'
          '因此与窗口吞吐不同，也不等于“请求 TPOT 均值的倒数”。','',
          '## 5. 解释结果时必须保留的条件','',
          '- **1K/32K 的 32K 是目标。** 45 分钟内两端均未完成 32,768 输出，完整请求数为 0；'
          '每端仅 1 条有意义的长请求延迟样本，另有截止附近的极短补发请求。不能把窗口统计称为完整 32K 输出成绩。',
          '- **1K/32K 首条输出为周期 3 的重复 token。** 原始输出重复 `[168,111,238]`，'
          '接近 4 的接受长度不能代表真实自然语言请求。',
          '- **每节点仅 1 条请求在途。** OFF 固定节点 23，ON 固定节点 25，尚未交换节点复测；'
          '窗口产出不能替代高并发服务容量。',
          '- **固定 Chunk=16384、Graph 实际关闭。** 与之前 Chunk8K/1K 的实验条件不同，'
          '不能把这里的 MTP 倍率直接乘到此前 Chunk 优化收益上。',
          '- **输出质量等价尚未验证。** 同输入完整配对也存在输出 ID 不一致；'
          '本页报告速度和接受计数，不据此宣称质量完全相同。',
          '- 本页按场景分别汇总，不把三种不同输入、输出与缓存负载混成一个“整体 MTP 加速倍数”。','',
          f'配置和证据：[实验清单]({source}/experiment_manifest.json)、'
          f'[输出模式审计]({source}/analysis/output_patterns.json)、'
          f'[配对逐请求记录]({source}/analysis/paired_requests.csv)。','',
          '## 6. 文件与复算','',
          f'[两页 PDF]({asset}/MTP_四项指标对比.pdf)；图像均提供 PNG 和 SVG：','',
          '| 图 | PNG | SVG |','|---|---|---|',
          f'| 接受长度与输出吞吐 | [PNG]({asset}/01_接受长度与输出吞吐.png) | [SVG]({asset}/01_接受长度与输出吞吐.svg) |',
          f'| TTFT / TPOT | [PNG]({asset}/02_TTFT与TPOT.png) | [SVG]({asset}/02_TTFT与TPOT.svg) |','',
          f'派生数据：[汇总 CSV]({asset}/summary.csv)、[逐请求 CSV]({asset}/requests.csv)、'
          f'[延迟分位数 CSV]({asset}/latency_percentiles.csv)、[完整配对延迟]({asset}/paired_latencies.csv)、'
          f'[校验 JSON]({asset}/summary.json)。分位数采用最近秩法；n<3 时留空，其余小样本分位数也仅供描述。','',
          f'生成代码：[plot_report.py]({asset}/plot_report.py)；[输入和代码 SHA256]({asset}/input_manifest.json)。','',
          '```bash\npython3 /home/chengyetai/workspace/current_exp/mtp_full_20260928/analysis/four_metrics_20260928/plot_report.py\n```','',
          '本次从 **83 条原始 SSE** 按 45 分钟截止时刻重算输出数、TTFT 和 TPOT，'
          '核对最终输出 ID、缓存命中与接受直方图，并核对相同输入完整配对。'
          '主要统计均与原归档 summary 一致。脚本只读归档，不调用在线服务。','']
    DOC.write_text('\n'.join(s),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=derive();fonts();plot_accept_throughput(result['groups']);plot_latencies(result['groups'])
    with PdfPages(OUT/'MTP_四项指标对比.pdf') as pdf:
        for name,fig in FIGURES:pdf.savefig(fig,bbox_inches='tight')
    for name,fig in FIGURES:plt.close(fig)
    document(result)
    manifest={'source_run':str(RUN),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(INPUTS)]}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(DOC),'figures':[n for n,f in FIGURES],'checks':result['checks']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
