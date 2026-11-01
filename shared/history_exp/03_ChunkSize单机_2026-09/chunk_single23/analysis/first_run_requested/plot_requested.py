#!/usr/bin/env python3
"""Charts/tables requested for the 2026-09-26 Chunk benchmark, archived data only."""
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('/home/chengyetai/workspace/current_exp/chunk_single23/results/20260926_145836')
ASSETS = Path(__file__).resolve().parent
DOC = Path('/home/chengyetai/workspace/current_exp/doc/ChunkSize首轮TTFT_TPOT_E2E_ITL_速度对比_2026-09-26.md')
CHUNKS = (8192,1024)
COLORS = {8192:'#3176af',1024:'#dd6b3d'}
CASES = [
 ('cold', '54.6K 冷态','54.6K\ncold'),
 ('solo', '54.6K 暖态（独立）','54.6K\nwarm'),
 ('with_decode', '54.6K 暖态（与 Decode 同时）','54.6K\n+ decode'),
 ('length32','32K 暖态（目标 32K 总输入）','32K\nwarm (target)'),
 ('length16','16K 暖态','16K\nwarm'),
 ('decode_alone','背景 Decode（单独）','Decode\nalone'),
 ('decode_mixed','背景 Decode（注入长 Prefill）','Decode\n+ prefill'),
]
LABELS = {key: label for key,label,_ in CASES}
TICKS = {key: tick for key,_,tick in CASES}
ORDER = [key for key,_,_ in CASES]
LONG = ORDER[:5]
FIG1_ORDER = ('length32',)  # User's target: 32K total input, ~60% token cache hit, 300 output.

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def categorized(chunk):
    folder=BASE/f'chunk_{chunk}'
    full, smoke=read(folder/'full.jsonl'),read(folder/'smoke.jsonl')
    select=lambda role, prefix:[r for r in full if r.get('role')==role and r.get('stage','').startswith(prefix)]
    return {
        'cold':select('long_cold','cold'),
        'solo':[r for r in smoke if r.get('role')=='long_warm'],
        'with_decode':select('long_warm','mixed_'),
        'length32':select('long_warm','length_32768'),
        'length16':select('long_warm','length_16384'),
        'decode_alone':select('decode','baseline_'),
        'decode_mixed':select('decode','mixed_'),
    }
DATA={c:categorized(c) for c in CHUNKS}

def phases(r):
    m=r['meta_info']
    q=float(m['queue_time'])
    prefill=float(m['prefill_finished_time'])-float(m['forward_entry_time'])
    unattributed=float(r['ttft_s'])-q-prefill
    tail=float(r['e2e_s'])-float(r['ttft_s'])
    if min(q,prefill,unattributed,tail)<0:
        raise ValueError('negative phase duration: '+r['stage'])
    return q,prefill,unattributed,tail

def rate(r):
    _,prefill,_,_=phases(r)
    return (r['prompt_tokens']-r['cached_tokens'])/prefill

def client_gaps(r):
    e=r['events']
    if any(b['completion_tokens']-a['completion_tokens']!=1 for a,b in zip(e,e[1:])):
        raise ValueError('SSE completion counter is not one token per event')
    return [1000*(b['t']-a['t']) for a,b in zip(e,e[1:])]

def mean(rows,fn):
    return statistics.mean(fn(r) for r in rows)

def nearest_rank(values,q):
    values=sorted(values)
    return values[math.ceil(q*len(values))-1]

plt.rcParams.update({'font.size':10,'axes.titlesize':12,'savefig.dpi':165,'figure.facecolor':'white'})
def common(ax,order=ORDER):
    ax.set_xticks(range(len(order)),[TICKS[x] for x in order],fontsize=8)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)

# Fig 1: target workload only, one stacked bar for each Chunk.
fig,ax=plt.subplots(figsize=(12.5,4.4),constrained_layout=True)
phase_labels=['Queue (reported)','Prefill (server timestamps)','Unattributed before first token',
              'After first token (client time)']
phase_colors=['#c8af59','#54a57f','#a38bc2','#4478a8']
ypos=[]
ynames=[]
endpoints=[]
for i,key in enumerate(FIG1_ORDER):
    for sub,c in enumerate(CHUNKS):
        y=i*2.3+sub*.82
        rs=DATA[c][key]
        segments=[mean(rs,lambda r,k=k:phases(r)[k]) for k in range(4)]
        left_seconds=0.0
        for j,(label,color,part) in enumerate(zip(phase_labels,phase_colors,segments)):
            ax.barh(y,part,left=left_seconds,height=.68,color=color,
                    label=label if i==sub==0 else None)
            left_seconds+=part
        ypos.append(y)
        ynames.append(f'32K warm | Chunk {c//1024}K')
        endpoints.append((y,left_seconds))
for y,total in endpoints:
    ax.text(total+3,y,f'{total:.1f}s',va='center',fontsize=8)
ax.set_yticks(ypos,ynames,fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0,max(t for _,t in endpoints)*1.18)
ax.set_xlabel('client E2E (seconds): segments follow real time left to right')
ax.set_title('32K warm: 32768 total input, 19584 cached, 300 output (one request per Chunk)')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,fontsize=9)
fig.savefig(ASSETS/'01_e2e_breakdown.png');plt.close(fig)

# Fig 2: per-request SSE event gaps for background Decode, only.
itl_keys=['decode_alone','decode_mixed']
fig,axs=plt.subplots(2,1,figsize=(9,7.8),constrained_layout=True)
statnames=['P50','P95','P99']
for i,key in enumerate(itl_keys):
    for c,shift in ((8192,-.18),(1024,.18)):
        rs=DATA[c][key];v=[x for r in rs for x in client_gaps(r)]
        for k,(s,quant) in enumerate(zip(statnames,(.5,.95,.99))):
            x=i*4+k+shift
            axs[0].bar(x,nearest_rank(v,quant),width=.35,color=COLORS[c],label=f'{c:,}' if i==k==0 else None)
        maxima=[max(client_gaps(r))/1000 for r in rs]
        axs[1].bar(i+shift,statistics.mean(maxima),width=.35,color=COLORS[c])
        for j,x in enumerate(maxima):
            axs[1].scatter(i+shift+(j-.5)*.065,x,marker='_',s=100,c='#243242',zorder=4)
axs[0].set_xticks([i*4+k for i in range(2) for k in range(3)],
                   [f'{"alone" if i==0 else "+ prefill"}\n{p}' for i in range(2) for p in statnames])
axs[0].set(ylabel='client ITL (ms)',title='Per-token client arrival gaps pooled from two 900-token requests')
axs[0].legend(title='Chunk tokens')
axs[1].set_xticks(range(2),['Decode alone','Decode + prefill'])
axs[1].set(ylabel='mean of per-request longest ITL (s)',title='Worst pause for each Decode request; short ticks = two individual runs')
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.savefig(ASSETS/'02_itl_bars.png');plt.close(fig)

# Figures 3/4: true server-phase token count/time versus client post-first rate.
for name,order,get,unit,title in [
 ('03_prefill_speed_bars.png',LONG,rate,'uncached tokens / server prefill seconds (tok/s)',
  'Prefill phase speed: server forward-entry to prefill-finished'),
 ('04_decode_speed_bars.png',ORDER,lambda r:1000/r['tpot_ms'],
  'client post-first output rate (tok/s)', 'Decode speed: inverse of client per-request TPOT')]:
    fig,ax=plt.subplots(figsize=(11 if len(order)==7 else 9,4.5),constrained_layout=True)
    for i,key in enumerate(order):
        for c,shift in ((8192,-.18),(1024,.18)):
            rs=DATA[c][key];values=[get(r) for r in rs]
            avg=statistics.mean(values)
            ax.bar(i+shift,avg,width=.34,color=COLORS[c],label=f'{c:,}' if i==0 else None)
            ax.text(i+shift,avg*1.014,f'{avg:.0f}' if name.startswith('03_') else f'{avg:.2f}',
                    fontsize=8,ha='center',va='bottom')
            for j,val in enumerate(values):
                ax.scatter(i+shift+(j-(len(values)-1)/2)*.068,val,marker='_',s=85,color='#283543',zorder=4)
    common(ax,order);ax.set(ylabel=unit,title=title)
    ax.legend(title='Chunk tokens',loc='upper left',bbox_to_anchor=(1.01,1),borderaxespad=0)
    ax.set_ylim(top=ax.get_ylim()[1]*1.08)
    fig.savefig(ASSETS/name);plt.close(fig)

# Archive exact grouped values used by the Markdown tables / charts.
with (ASSETS/'breakdown_per_request.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['chunk','case','stage','role','ttft_s','tpot_ms','e2e_s',
        'queue_s','server_prefill_s','first_token_unattributed_s','after_first_token_s',
        'prefill_tok_s','decode_tok_s','client_itl_max_ms'])
    for c in CHUNKS:
        for key in ORDER:
            for r in DATA[c][key]:
                q,p,o,t=phases(r)
                w.writerow([c,key,r['stage'],r['role'],r['ttft_s'],r['tpot_ms'],r['e2e_s'],q,p,o,t,
                    rate(r) if key in LONG else '',1000/r['tpot_ms'],max(client_gaps(r)) if len(r['events'])>1 else ''])

doc=['# 首轮 Chunk Size 对比：TTFT、TPOT、E2E、ITL 和速度', '',
 '数据：2026-09-26，23 节点单机 Chunk 8192 与 1024；[原始结果](../chunk_single23/results/20260926_145836/)、[生成脚本](../chunk_single23/analysis/first_run_requested/plot_requested.py)。这里只重算存档，不向在线服务发请求。', '',
 '需求澄清：**32K 指总输入长度**，输出 300，约 60% token 缓存命中。图中的 **32K warm** 才符合这个长度/缓存场景：前缀预热 19661 token，实测命中 19584/32768 ≈ 59.77%；8K/1K 各只有一条独立暖态请求。54.6K cold/warm/+decode 来自此前将 32K 误当作公共前缀长度的设计，**不是目标 32K 场景**；目标场景尚无 32K + Decode 干扰样本。', '',
 '以下逐个指标给出结果。各场景每种 Chunk 只有 **1 或 2 条请求**；P95/P99 按 nearest-rank 算法计算，所以在 n=1 或 2 时**都等于该组最大值，不能解读为真实尾延迟**。表内均值是组内请求算术均值，未混合不同角色。历史工作负载使用随机 token ID、`ignore_eos=True`，没有真实回复的语义验收；这些图只用于理解首轮数据。', '',
 '## 1. TTFT 与 TPOT：均值、P95、P99', '',
 'TTFT：客户端发送请求至首个输出 token；TPOT：`(E2E − TTFT) / (实际输出 token 数 − 1)`。单位见表头。长请求 54.6K 的缓存命中在暖态为 32,768/54,614 ≈60%；背景 Decode 为 1,024 输入/900 输出。暖态独立取冒烟记录；“与 Decode 同时”只取两轮注入时的长请求。', '']
for field,unit in [('ttft_s','s'),('tpot_ms','ms/token')]:
    doc += [f'### {"TTFT" if field=="ttft_s" else "TPOT"}（{unit}）','',
            '| 场景 | 8K n | 8K 均值 | 8K P95 | 8K P99 | 1K n | 1K 均值 | 1K P95 | 1K P99 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in ORDER:
        row=[LABELS[key]]
        for c in CHUNKS:
            arr=[r[field] for r in DATA[c][key]]
            row+= [str(len(arr)),f'{statistics.mean(arr):.2f}',f'{nearest_rank(arr,.95):.2f}',f'{nearest_rank(arr,.99):.2f}']
        doc.append('| '+' | '.join(row)+' |')
    doc.append('')

doc += ['## 2. E2E（s）及 breakdown','',
 '![E2E 与四段账面分解](../chunk_single23/analysis/first_run_requested/01_e2e_breakdown.png)','',
 '按每条请求原始字段计算，再对同场景取均值。四段**账面上严格相加为客户端 E2E**：', '',
 '1. **Queue**：服务端 `meta_info.queue_time`，确实记录了，单位转换为 ms。',
 '2. **Prefill 阶段**：服务端 `prefill_finished_time − forward_entry_time`；表示从进入 forward 到首阶段结束的服务器时间，未进一步拆算子计算与调度。',
 '3. **首 token 前未归因**：客户端 TTFT 减前两项，包括计时起点差、请求收发、转发、首输出发送等；**没有 IPC 单独耗时，不能把这一整段叫 IPC**。',
 '4. **首 token 后至结束**：客户端 `E2E − TTFT`，包含连续 Decode、输出传输等，**不能说全是 NPU Decode**。', '',
 '图 1 **只画符合目标的 32K 暖态请求**：上条 Chunk 8K，下条 Chunk 1K，均为单条请求；Queue → Prefill → 首 token 前未归因 → 首 token 后依次相接；条末是 E2E。Queue 与未归因时间相对整段输出很短，按真实比例肉眼难辨，具体数值见下表。其他历史负载不进入图 1。','',
 '| 场景 | Chunk | n | E2E(s) | Queue(ms) | Prefill(s) | 首 token 前未归因(ms) | 首 token 后(s) |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for key in FIG1_ORDER:
    for c in CHUNKS:
        rs=DATA[c][key]
        q,p,o,t=(mean(rs,lambda r,j=j:phases(r)[j]) for j in range(4))
        doc.append(f'| {LABELS[key]} | {c} | {len(rs)} | {mean(rs,lambda r:r["e2e_s"]):.2f} | {q*1000:.2f} | {p:.2f} | {o*1000:.1f} | {t:.2f} |')
doc += ['',
 '### 图 1 为什么 Prefill 差很多，而 E2E 差很少？', '',
 '这里每个 Chunk 只有 **一条** 32K 暖态请求，二者实际输入均为 32768 token、缓存命中均为 19584 token、实际输出均为 300 token。8K 的服务端 Prefill 阶段为 5.754 s，1K 为 9.138 s（1K 是 8K 的 **1.59 倍**，不是接近两倍的 54.6K 案例）；客户端 TTFT 分别为 5.769 s、9.152 s。', '',
 '- **8K**：E2E = TTFT 5.769 s + 首 token 后 142.888 s = **148.657 s**。',
 '- **1K**：E2E = TTFT 9.152 s + 首 token 后 140.851 s = **150.003 s**。', '',
 '因此 1K 的首 token 前多了约 **3.383 s**，但首 token 后少了约 **2.037 s**，相抵后 E2E 只多 **1.346 s**。后段包含 **299 个输出 token**，每 token 平均相差约 **6.814 ms**；这是客户端 TPOT 差异乘 299 所得。每条请求记录了 300 次逐 token SSE，结束时间和最后一次 SSE 相差不足 0.2 ms；客户端 E2E 与服务端 `e2e_latency` 仅相差约 5 ms。现有记录在加法和实际 token 数上自洽。', '',
 '**口径限制**：后段是客户端“首 token 到请求结束”，含 Decode 与输出传输，**不是单独测得的 NPU Decode 计算耗时**；单条请求的约 1.4% TPOT 波动累计到 299 个间隔就有约 2 秒。每种 Chunk 只有 1 条，且这批逐请求记录**没有保存生成 token ID**，无法检查两种 Chunk 的实际生成序列是否一致；不能证明 1K 的 Decode 稳定更快，也不能因为 E2E 接近就说 Prefill 没受影响。要评价稳定性须重复相同 32K 工作负载、保存逐 token 输出并检查尾延迟与输出停顿。', '']
doc += ['','## 3. ITL：背景 Decode 的条形图','',
 '![Decode ITL 条形图](../chunk_single23/analysis/first_run_requested/02_itl_bars.png)','',
 '相邻 SSE 事件的 `completion_tokens` 在这 8 条背景 Decode 请求中每次只增加 1；故可以计算**客户端观察到的**逐 token 输出间隔。上图 P50/P95/P99 汇合每种 Chunk、每种状态的两条请求（合计 1,798 个间隔）；下图先计算每条请求最久停顿，再画两条的均值，并用短横线标出各自原值。请求级样本仍只有两条。','',
 '| Chunk | Decode 状态 | ITL P50(ms) | ITL P95(ms) | ITL P99(ms) | 每请求最长 ITL(s) |',
 '|---:|---|---:|---:|---:|---|']
for c in CHUNKS:
    for key in ('decode_alone','decode_mixed'):
        rs=DATA[c][key];g=[x for r in rs for x in client_gaps(r)]
        maxima=[max(client_gaps(r))/1000 for r in rs]
        doc.append(f'| {c} | {LABELS[key]} | {nearest_rank(g,.5):.1f} | {nearest_rank(g,.95):.1f} | {nearest_rank(g,.99):.1f} | '+', '.join(f'{v:.3f}' for v in maxima)+' |')
doc += ['','平均 TPOT 接近时仍可能有单次长停顿；这里报告的 ITL 是**客户端收到 SSE 的间隔**，不是服务端 NPU 逐步执行计时。','',
 '## 4. Prefill 速度与 Decode 速度：两种 Chunk 柱状图','',
 '![Prefill 阶段速度柱状图](../chunk_single23/analysis/first_run_requested/03_prefill_speed_bars.png)','',
 'Prefill：对每条**长请求**计算 `(prompt_tokens − cached_tokens) / (prefill_finished_time − forward_entry_time)`，再取该场景算术均值。这里用服务端记录的 Prefill 阶段区间，**不再用 TTFT 冒充 Prefill 用时**；区间内部可能含非算子开销，不能叫纯 NPU 计算速度。冷/暖/与背景并发分别比较，短横线是单次值。', '',
 '![Decode 客户端速度柱状图](../chunk_single23/analysis/first_run_requested/04_decode_speed_bars.png)','',
 'Decode：每请求先用 `1000 / TPOT(ms)` 得到首 token 后平均输出 tok/s，再取同场景算术均值；该速率包含客户端输出传输，不是设备瞬时 Decode 速度。长请求和背景 Decode 的目标输出分别为 300、900，图中按场景分开；短横线是单次值。', '',
 '## 5. 目前数据的边界','',
 '- TTFT/TPOT 的请求级 P95/P99 无统计意义：长请求每场景 1–2 条，背景 Decode 每状态 2 条。ITL 分位数是**间隔级**，不等于请求级尾延迟。',
 '- 现有计时没有独立 IPC、NPU 纯算子时间、首 token 后 Decode 与传输的分离；E2E 图把不可拆项明确归为“未归因”和“首 token 后”。',
 '- 首轮没有通过真实中文提示词做语义验收，不能凭随机 token 输出长度认定模型输出正常。图表用来回看这次实验，正式性能结论需要新一轮语义检查和更多重复。',
 '- 本页不把短窗口的 1–2 条请求当成稳态并发吞吐；持续吞吐测量需另做。', '',
 '逐条结果与阶段分解可查 [breakdown_per_request.csv](../chunk_single23/analysis/first_run_requested/breakdown_per_request.csv)。', '']
# Check the interpretation of the two speed figures against each original request.
insert_at=doc.index('## 5. 目前数据的边界')
comparison=['### 图 3 与图 4：逐条件核查', '',
 '图 3 的 Prefill 阶段速度是先对每条请求分别计算，再取该场景的均值。下表的比值是 `8K 均值 / 1K 均值`；两组同场景的 `prompt_tokens`、`cached_tokens` 和目标输出数相同。', '',
 '| 长输入场景 | n/Chunk | 8K Prefill (tok/s) | 1K Prefill (tok/s) | 8K/1K |',
 '|---|---:|---:|---:|---:|']
for key in LONG:
    a=mean(DATA[8192][key],rate)
    b=mean(DATA[1024][key],rate)
    comparison.append(f'| {LABELS[key]} | {len(DATA[8192][key])} | {a:.1f} | {b:.1f} | {a/b:.2f}× |')
comparison += ['',
 '例如 54.6K 暖态独立请求，两组都是 `54,614 − 32,768 = 21,846` 个未命中 token、300 输出；服务端 Prefill 阶段时间分别约 **9.90 s / 19.10 s**。分子一致，1.93 倍的速率比来自这一阶段耗时之差。更小的 Chunk 对应更多分块轮次是可能原因之一，但这批时间戳不能单独证明调度、算子和传输各自贡献。', '',
 '图 4 应把长请求和背景 Decode 分开读。下面只取最能检查 Decode 干扰的两组 **1,024 输入 / 900 输出背景请求**（各 Chunk 每组 n=2）；“差异”是 `(1K/8K − 1) × 100%`。', '',
 '| 背景 Decode | 8K 平均速度 (tok/s) | 1K 平均速度 (tok/s) | 1K 对 8K | 8K 两轮最长 ITL(s) | 1K 两轮最长 ITL(s) |',
 '|---|---:|---:|---:|---|---|']
for key in ('decode_alone','decode_mixed'):
    a=mean(DATA[8192][key],lambda r:1000/r['tpot_ms'])
    b=mean(DATA[1024][key],lambda r:1000/r['tpot_ms'])
    stalls=[]
    for c in CHUNKS:
        stalls.append(', '.join(f'{max(client_gaps(r))/1000:.3f}' for r in DATA[c][key]))
    comparison.append(f'| {LABELS[key]} | {a:.3f} | {b:.3f} | {(b/a-1)*100:+.2f}% | {stalls[0]} | {stalls[1]} |')
comparison += ['',
 '**核查结论**：8K 的服务端 Prefill 阶段速率在这些场景为 1K 的 **1.59–1.93 倍**，其中“54.6K 暖态独立”接近两倍；这是相同负载下的历史观测，只有 1–2 条请求/条件，不能解释为精确、稳定的计算性能增益。背景请求的**平均 Decode 速度**在两种 Chunk 间接近（单独约 2.11 tok/s、混合约 2.02–2.03 tok/s）；但注入长 Prefill 时，客户端**最长 ITL** 在 1K 两轮为 15.596/17.821 s，在 8K 为 6.976/7.008 s。故只能说“平均 Decode 速度未见明显差异”，**不能说 Chunk Size 对 Decode 无影响**。', '',
 '两组服务日志均为 `enable_mixed_chunk=False`；因此“减小 Chunk 就一定会在块间穿插 Decode”的前提在本实验也不成立。[原实验调度条件](ChunkSize单机实验_2026-09-26.md)。', '',
 '一致性核对：各背景 Decode 请求中的服务端 `meta_info.decode_throughput` 与按客户端 `1000/TPOT` 重算的速度最大差约 `0.000004 tok/s`；这验证了两份记录的口径在本批数据中一致，**不是独立的更多次性能样本**。', '']
doc[insert_at:insert_at]=comparison
DOC.write_text('\n'.join(doc))
print(DOC)
