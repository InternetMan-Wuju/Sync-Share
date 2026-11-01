#!/usr/bin/env python3
"""Plot archived 2026-09-26 Chunk comparison; no server or model access."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/chengyetai/workspace/current_exp/chunk_single23/results/20260926_145836')
OUT = Path(__file__).resolve().parent
COLORS = {8192: '#2879b9', 1024: '#dc663b'}
LABELS = {
    'cold': '54.6K\ncold',
    'smoke': '54.6K\nwarm solo',
    'long_mixed': '54.6K\nwarm + decode',
    'length_32768': '32K\nwarm',
    'length_16384': '16K\nwarm',
    'baseline': 'Decode\nalone',
    'mixed_decode': 'Decode\n+ prefill',
}
ORDER = list(LABELS)
LONG_ORDER = ORDER[:5]

def load(chunk):
    folder = ROOT / f'chunk_{chunk}'
    data = {}
    for name in ('full', 'smoke'):
        data[name] = [json.loads(line) for line in (folder / f'{name}.jsonl').read_text().splitlines()]
    data['timeline'] = [json.loads(line) for line in (folder / 'metrics_timeline.jsonl').read_text().splitlines()]
    return data

DATA = {size: load(size) for size in COLORS}

def groups(data):
    full, smoke = data['full'], data['smoke']
    select = lambda role, prefix: [r for r in full if r.get('role') == role and r.get('stage', '').startswith(prefix)]
    return {
        'cold': select('long_cold', 'cold'),
        'smoke': [r for r in smoke if r.get('role') == 'long_warm'],
        'long_mixed': select('long_warm', 'mixed_'),
        'length_32768': select('long_warm', 'length_32768'),
        'length_16384': select('long_warm', 'length_16384'),
        'baseline': select('decode', 'baseline_'),
        'mixed_decode': select('decode', 'mixed_'),
    }

GROUPS = {c: groups(data) for c, data in DATA.items()}

def style(ax, labels=ORDER):
    ax.set_xticks(range(len(labels)), [LABELS[s] for s in labels], fontsize=8)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

def dots(ax, keys, get, **kwargs):
    for i, key in enumerate(keys):
        for size, shift in ((8192, -.17), (1024, .17)):
            v = [get(r) for r in GROUPS[size][key]]
            v = [x for x in v if x is not None and math.isfinite(x)]
            for j, x in enumerate(v):
                jitter = (j-(len(v)-1)/2)*.075
                ax.scatter(i+shift+jitter, x, s=40, color=COLORS[size], edgecolor='white', linewidth=.45,
                           zorder=3, label=(f'{size:,}' if i == 0 and j == 0 else None), **kwargs)
    style(ax, keys)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white', 'savefig.dpi': 165})
fig, axs = plt.subplots(3, 1, figsize=(11, 11.5), constrained_layout=True)
dots(axs[0], ORDER, lambda r: r.get('ttft_s'))
axs[0].set(ylabel='TTFT (s)', title='Request latency: each point is one request')
axs[0].legend(title='Chunk tokens', ncol=2)
dots(axs[1], ORDER, lambda r: r.get('tpot_ms'))
axs[1].set(ylabel='TPOT (ms/token)', title='Per-request mean TPOT (not tail ITL)')
# E2E = TTFT + remaining generation; draw mean of measured requests in each small group.
for i, key in enumerate(ORDER):
    for size, shift in ((8192, -.17), (1024, .17)):
        rs = GROUPS[size][key]
        if not rs:
            continue
        first = np.mean([r['ttft_s'] for r in rs])
        tail = np.mean([r['e2e_s'] - r['ttft_s'] for r in rs])
        axs[2].bar(i+shift, first, width=.32, color=COLORS[size], alpha=.9)
        axs[2].bar(i+shift, tail, width=.32, bottom=first, color=COLORS[size], alpha=.37)
        for j,r in enumerate(rs):
            axs[2].scatter(i+shift+(j-(len(rs)-1)/2)*.075, r['e2e_s'], marker='_', s=110,
                           color='#263744', zorder=4)
style(axs[2])
axs[2].set(ylabel='E2E (s)', title='E2E: solid TTFT + pale after-first-token; ticks = individual E2E')
fig.savefig(OUT / '01_latency.png')
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
dots(axs[0,0], LONG_ORDER, lambda r: (r['prompt_tokens']-r['cached_tokens'])/r['ttft_s'] if r['ttft_s'] else None)
axs[0,0].set(ylabel='uncached tokens / TTFT (tok/s)', title='Prefill proxy; includes queue / transfer / first decode')
axs[0,0].legend(title='Chunk tokens')
dots(axs[0,1], ORDER, lambda r: 1000/r['tpot_ms'] if r.get('tpot_ms') else None)
axs[0,1].set(ylabel='post-first output rate (tok/s)', title='Decode speed derived from per-request TPOT')
# Burst wall time: from earliest request start to last completion; exclude warmups and flush.
BURSTS = defaultdict(dict)
for size in COLORS:
    g = GROUPS[size]
    for key in ORDER:
        if key in ('baseline', 'mixed_decode', 'long_mixed'):
            continue
        for r in g[key]:
            BURSTS[size][key + ('_0' if key == 'smoke' else '')] = [r]
    for j in (0,1):
        for typ, stage in (('baseline', f'baseline_{j}'), ('mixed', f'mixed_{j}')):
            reqs = [r for r in DATA[size]['full'] if r.get('stage') == stage and r.get('role') in ('decode', 'long_warm')]
            BURSTS[size][f'{typ}_{j}'] = reqs

burst_csv = []
for size, cases in BURSTS.items():
    for stage, rows in cases.items():
        start = min(r['started_at'] for r in rows)
        finish = max(r['started_at']+r['e2e_s'] for r in rows)
        span = finish - start
        burst_csv.append({'chunk':size,'stage':stage,'requests':len(rows), 'output_tokens':sum(r['output_tokens'] for r in rows),
                          'window_s':span,'output_tok_s':sum(r['output_tokens'] for r in rows)/span,'req_s':len(rows)/span})

for i,key in enumerate(('cold','smoke','long_mixed','length_32768','length_16384','baseline','mixed_decode')):
    prefixes = {'cold':['cold'], 'smoke':['smoke_0'], 'long_mixed':['mixed_0','mixed_1'],
                'length_32768':['length_32768'],'length_16384':['length_16384'],
                'baseline':['baseline_0','baseline_1'], 'mixed_decode':['mixed_0','mixed_1']}[key]
    for size, shift in ((8192,-.17),(1024,.17)):
        for j,stage in enumerate(prefixes):
            row=next(x for x in burst_csv if x['chunk']==size and x['stage']==stage)
            offset=(j-(len(prefixes)-1)/2)*.075
            for ax,metric in ((axs[1,0],'output_tok_s'),(axs[1,1],'req_s')):
                ax.scatter(i+shift+offset, row[metric],s=40,color=COLORS[size],edgecolor='white',linewidth=.45)
for ax in axs[1,:]: style(ax)
axs[1,0].set(ylabel='output tokens / wall time (tok/s)', title='Short active windows only (1 or 2 requests)')
axs[1,1].set(ylabel='requests / wall time (req/s)', title='Short active windows only; no sustained QPS test')
fig.savefig(OUT / '02_rates_and_short_windows.png')
plt.close(fig)

with (OUT / 'short_windows.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=burst_csv[0].keys())
    writer.writeheader();writer.writerows(burst_csv)

def gaps(row):
    ev=row['events']
    if not all(b['completion_tokens']-a['completion_tokens']==1 for a,b in zip(ev,ev[1:])):
        return []
    return [(ev[i]['t'], ev[i]['t']-ev[i-1]['t']) for i in range(1,len(ev))]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for size, shift in ((8192,-.17),(1024,.17)):
    for key,i in (('baseline',0),('mixed_decode',1)):
        for j,r in enumerate(GROUPS[size][key]):
            gs=gaps(r)
            axs[0,0].scatter(i+shift+(j-.5)*.08, max(x for _,x in gs), s=52,color=COLORS[size],
                             edgecolor='white',linewidth=.4,label=(f'{size:,}' if key=='baseline' and j==0 else None))
            axs[0,1].scatter(i+shift+(j-.5)*.08,np.quantile([x for _,x in gs],.99)*1000,
                             s=52,color=COLORS[size],edgecolor='white',linewidth=.4)
axs[0,0].set_xticks([0,1], ['Decode alone','Decode + prefill']);axs[0,0].set(ylabel='max client ITL (s)',title='One long interruption can hide in mean TPOT')
axs[0,0].legend(title='Chunk tokens')
axs[0,1].set_xticks([0,1], ['Decode alone','Decode + prefill']);axs[0,1].set(ylabel='P99 client ITL (ms)',title='P99 of 899 intervals per request; two requests/group')
for ax in axs[0,:]: ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
for size,ax in ((8192,axs[1,0]),(1024,axs[1,1])):
    full=DATA[size]['full']; mixed=next(r for r in full if r.get('stage')=='mixed_0' and r.get('role')=='decode')
    base=next(r for r in full if r.get('stage')=='baseline_0' and r.get('role')=='decode')
    control=next(r for r in full if r.get('stage')=='mixed_0' and r.get('role')=='control' and 'injected_at' in r)
    injection_elapsed=control['injected_at']-mixed['started_at']
    for r,label,color in ((base,'baseline', '#6b7785'),(mixed,'prefill injected',COLORS[size])):
        xs=[t-injection_elapsed for t,v in gaps(r)]; ys=[v for t,v in gaps(r)]
        keep=[(x,y) for x,y in zip(xs,ys) if -2<=x<=30]
        ax.scatter([x for x,y in keep],[y for x,y in keep], s=14,color=color,label=label,alpha=.8)
    ax.axvline(0,color='black',ls=':',lw=1)
    ax.set(xlabel='seconds relative to long-prefill injection',ylabel='client ITL (s)',
           title=f'Chunk {size:,}: first paired round')
    ax.grid(alpha=.2);ax.legend(loc='upper right');ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT / '03_client_itl_decode_interference.png')
plt.close(fig)

fig,axs=plt.subplots(2,1,figsize=(11,7.2),constrained_layout=True)
dots(axs[0], LONG_ORDER, lambda r: r['cached_tokens']/r['prompt_tokens']*100 if r.get('prompt_tokens') else None)
axs[0].set(ylabel='cached / prompt (%)',title='Actual request-level token cache hit ratio')
axs[0].legend(title='Chunk tokens')
for i,key in enumerate(LONG_ORDER):
    for size,shift in ((8192,-.17),(1024,.17)):
        timeline=DATA[size]['timeline']
        for j,r in enumerate(GROUPS[size][key]):
            observed=[s['kv_available_tokens'] for s in timeline if 'kv_available_tokens' in s and
                      r['started_at']<=s['at']<=r['started_at']+r['e2e_s']]
            if observed:
                axs[1].scatter(i+shift+(j-(len(GROUPS[size][key])-1)/2)*.075,min(observed),
                               s=42,color=COLORS[size],edgecolor='white',linewidth=.45)
style(axs[1],LONG_ORDER)
axs[1].set(ylabel='min sampled free KV tokens',title='Rank-0 KV pool; sampled about every 3 seconds')
fig.savefig(OUT / '04_cache_and_kv_context.png')
plt.close(fig)

# Derived per-request values. These are analysis products, not extra measurements.
with (OUT/'per_request.csv').open('w',newline='') as f:
    fields=['chunk','group','stage','role','input_tokens','cached_tokens','output_tokens',
            'ttft_s','tpot_ms','e2e_s','client_itl_max_s','client_itl_p99_s',
            'prefill_proxy_tok_s','decode_postfirst_tok_s']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for size in COLORS:
        for key,rows in GROUPS[size].items():
            for r in rows:
                gl=[gap for _,gap in gaps(r)]
                writer.writerow({'chunk':size,'group':key,'stage':r['stage'],'role':r['role'],
                    'input_tokens':r['prompt_tokens'],'cached_tokens':r['cached_tokens'],
                    'output_tokens':r['output_tokens'],'ttft_s':r['ttft_s'],
                    'tpot_ms':r['tpot_ms'],'e2e_s':r['e2e_s'],
                    'client_itl_max_s':max(gl) if gl else '',
                    'client_itl_p99_s':np.quantile(gl,.99) if gl else '',
                    'prefill_proxy_tok_s':(r['prompt_tokens']-r['cached_tokens'])/r['ttft_s'] if key in LONG_ORDER else '',
                    'decode_postfirst_tok_s':1000/r['tpot_ms']})
print('saved to',OUT)
