#!/usr/bin/env python3
"""CPU-only catalog of measured MTP workloads; never send model requests."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path('/home/chengyetai/workspace/current_exp/Q1MTPExp')
HIST = ROOT.parent.parent/'history_exp'
ROWS = []


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def request_stats(records):
    accepted = [r for r in records if not r.get('error') and r.get('meta', {}).get('spec_verify_ct')]
    if not accepted:
        return {}
    count = sum(r['completion_tokens'] for r in accepted)
    verify = sum(r['meta']['spec_verify_ct'] for r in accepted)
    correct = sum(r['meta']['spec_num_correct_drafts'] for r in accepted)
    proposed = sum(r['meta']['spec_num_proposed_drafts'] for r in accepted)
    hs = [r['meta'].get('spec_correct_drafts_histogram') or r['meta'].get('spec_accept_histogram') for r in accepted]
    positions = None
    if all(h and sum(h) == r['meta']['spec_verify_ct'] for h, r in zip(hs, accepted)):
        hist = [sum(h[i] if i < len(h) else 0 for h in hs) for i in range(4)]
        positions = [sum(hist[i:])/sum(hist) for i in (1, 2, 3)]
    for r in accepted:
        h = r['meta'].get('spec_correct_drafts_histogram') or r['meta'].get('spec_accept_histogram')
        if h:
            assert sum(h) == r['meta']['spec_verify_ct']
            assert sum(i*v for i,v in enumerate(h)) == r['meta']['spec_num_correct_drafts']
        assert r['meta']['spec_num_proposed_drafts'] == 3*r['meta']['spec_verify_ct']
    prompt = sum(r['meta'].get('prompt_tokens', 0) for r in accepted)
    return {'AL':count/verify, 'draft_rate':correct/proposed, 'P1_P2_P3':positions,
            'requests_measured':len(accepted), 'output_total':count,
            'input_lengths':sorted({r['meta'].get('prompt_tokens') for r in accepted}),
            'cache_hit_ratio':sum(r['meta'].get('cached_tokens', 0) for r in accepted)/prompt if prompt else None,
            'verify_ct':verify,'correct_drafts':correct,'proposed_drafts':proposed}


def add_current(path, category, label, dataset, node='23', note=''):
    data = read(path)
    requests = data['requests']
    row = {'category':category,'date':'2026-10-12','label':label,'node':node,'mode':'ON',
           'dataset':dataset,'concurrency':data['concurrency'], 'output_target':data.get('max_tokens',data.get('target_output_tokens')),
           'source_file':str(path),'notes':note, **request_stats(requests)}
    agg = data['aggregate']
    if not agg:
        row.update(status='failed',requests_measured=0,notes=note+'；未得到有效MTP测量')
    else:
        row['status'] = 'completed'
        row['TTFT_s'] = mean([r.get('ttft_s') for r in requests if not r.get('error')])
        row['E2E_s'] = mean([r.get('e2e_s') for r in requests if not r.get('error')])
        row['TPOT_ms'] = mean([r.get('tpot_ms', (r['e2e_s']-r['ttft_s'])*1000/(r['completion_tokens']-1)
                                if r.get('e2e_s') is not None and r.get('ttft_s') is not None and r['completion_tokens']>1 else None)
                               for r in requests if not r.get('error')])
        wall = agg.get('wall_s')
        row['output_tok_s'] = row['output_total']/wall if wall else None
        row['wall_s'] = wall
    ROWS.append(row)


def build_rows():
    evidence = read(ROOT/'audit_20261012/acceptance_evidence.json')
    preview = HIST/'04_MTP推测解码_2026-09/mtp_preview25/results/20260926_181500'
    for mode in ('on','off'):
        request_path = preview/mode/'requests.jsonl'
        measured = [json.loads(line) for line in request_path.read_text().splitlines()
                    if line.strip()]
        measured = [r for r in measured if r.get('role') == 'measurement']
        for scene, label in (('mixed','8K/1K'),('decode','1K/2048'),('long','总32K/300，缓存60%')):
            path = preview/mode/(scene+'_summary.json')
            data = read(path)
            scene_requests = [r for r in measured if r['scenario'].rsplit('_',1)[0] == scene]
            assert len(scene_requests) == data['n']
            acc = next(r['aggregate'] for r in evidence['history_20260926'] if r['mode']==mode and r['scene']==scene)
            ROWS.append({'category':'历史9月26日','date':'2026-09-26','label':label,'mode':mode.upper(),
                         'node':'25','dataset':'自写独立随机ID','concurrency':1,'requests_measured':data['n'],
                         'input_lengths':[data['input_tokens']],'output_target':data['output_tokens'],
                         'AL':acc['AL'],'draft_rate':acc['draft_accept_rate'],'P1_P2_P3':acc['P1_P2_P3'],
                         'TTFT_s':data['ttft_s']['mean'],'TPOT_ms':data['tpot_ms']['mean'],
                         'E2E_s':mean([r.get('e2e_s') for r in scene_requests]),
                         'measurement_source_file':str(request_path),
                         'output_tok_s':data['output_tok_s']['mean'],'throughput_method':'单请求输出/耗时的均值',
                         'cache_hit_ratio':data['actual_cache_ratio']['mean'],'source_file':str(path),
                         'status':'completed','notes':'单机同节点ON→OFF串行；1K/2048为缩短测试；完整输出ID未保存，不能复算周期尾段'})
    full = HIST/'04_MTP推测解码_2026-09/mtp_full_20260928/results/20260928_061159/analysis'
    for scene,label in (('mixed','8K/1K'),('decode','1K/32K目标，45分钟截断'),('prefix','总32K/300，缓存60%')):
        path=full/(scene+'_summary.json')
        data=read(path)
        for mode in ('on','off'):
            d=data[mode]
            acc=next(r['aggregate'] for r in evidence['history_20260928'] if r['mode']==mode and r['scene']==scene)
            ROWS.append({'category':'历史9月28日','date':'2026-09-28','label':label,'mode':mode.upper(),
                         'node':'25' if mode=='on' else '23','dataset':'自写独立随机ID','concurrency':1,
                         'requests_measured':d['complete_requests']+d['partial_requests'],
                         'requests_with_acceptance':acc['accepted_stat_requests'],
                         'input_lengths':[8192 if scene=='mixed' else 1024 if scene=='decode' else 32768],
                         'output_target':1024 if scene=='mixed' else 32768 if scene=='decode' else 300,
                         'output_total':d['observed_tokens'],'AL':acc['AL'],'draft_rate':acc['draft_accept_rate'],
                         'P1_P2_P3':acc['P1_P2_P3'],'acceptance_output_coverage':d['acceptance_output_coverage_ratio'],
                         'TTFT_s':d['mean_ttft_s'],'TPOT_ms':d['mean_observed_tpot_ms'],
                         'output_tok_s':d['window_output_tokens_per_s'],'throughput_method':'2700秒窗口总输出/窗口长度',
                         'cache_hit_ratio':mean(d['actual_cached_ratios']),'source_file':str(path),
                         'status':'window_finished','notes':f"完整{d['complete_requests']}、截断{d['partial_requests']}；输出总量与有接受统计的输出覆盖量不同；包含窗口边界记录"})
    for r in evidence['native_20261011']:
        raw=read(r['file'])
        ROWS.append({'category':'10月11日原生工具','date':'2026-10-11','label':r['dataset_name']+'：变长，上限8K/256',
                     'mode':'ON','node':'23','dataset':'SGLang '+r['dataset_name'], 'AL':r['accept_length'],
                     'requests_measured':r['completed'],'output_total':r['total_output_tokens'],
                     'source_file':r['file'],'status':'completed','notes':r['limit'],
                     'input_total':r['total_input_tokens'],'random_range_ratio':0,
                     'output_tok_s':raw.get('output_throughput') or raw.get('output_token_throughput')})
    for folder in ('q1mtp_20261012_135627','q1mtp_20261012_141630'):
        for path in sorted((ROOT/'results'/folder).glob('*.json')):
            data=read(path)
            if 'aggregate' not in data:continue
            real=path.name.startswith('real')
            add_current(path,'10月12日早期数据对比',path.stem,
                        'ShareGPT截取文本ID（无完整chat模板）' if real else '自写独立随机ID',
                        note='早期客户端仅存输出摘要；位置直方图缺失')
    for path in sorted((ROOT/'results/scenarios').glob('*.json')):
        add_current(path,'10月12日旧四组',path.stem.split('_mtpon')[0], '自写独立随机ID',
                    note='w3实际仅2048输出；请求数量未按文档倍率；除w4外缺位置直方图')
    for path in sorted((ROOT/'results/posrate').glob('*.json')):
        add_current(path,'10月12日逐位置补测',path.stem.rsplit('_',1)[0], '自写独立随机ID',
                    note='独立补测轮次；w3实际仅2048输出；完整output_ids未保存')
    short=ROOT/'results/native_randomids_20261012_1844'
    for name, ds in (('native_exact_ids','SGLang原生ID直发'),('uniform_legacy','自写独立随机ID'),('native_default_text','SGLang原生默认文本，重新分词')):
        add_current(short/(name+'.json'),'10月12日原生短复测',name,ds,
                    note='每组2条，完整输出保存；默认文本实际输入1106/1088，并非固定1024')
    latest=read(ROOT/'results/latest_native_full.json')
    out=Path(latest['out'])
    for name in ('w1_8k_1k','w2_32k_1k','w3_1k_32k','w4_prefix32k_300'):
        add_current(out/name/'result.json','10月12日原生完整重跑',name,
                    'SGLang原生共享前缀/后缀拼接' if name.startswith('w4') else 'SGLang原生ID直发',
                    note='完整长度；w1墙钟自请求时间范围恢复；服务与AboutModels存在配置差异')
    for folder in ('sweep','sweep2'):
        for path in sorted((ROOT/'results'/folder).glob('*.probe.json')):
            add_current(path,'配置短探针',path.name.split('_')[0],'真实短问询，34输入/32输出',
                        note='仅1条短请求验证功能；没有位置直方图；Graph在normal路径实际关闭')
    smoke=read(preview/'on/initial_smoke_085.json')
    meta=smoke['meta_info'];hist=meta['spec_correct_drafts_histogram']
    ROWS.append({'category':'初始短探针','date':'2026-09-26','label':'1K/32，MEM0.85','mode':'ON','node':'25',
                 'dataset':'自写独立随机ID','concurrency':1,'requests_measured':1,'input_lengths':[1024],
                 'output_target':32,'output_total':32,'AL':32/meta['spec_verify_ct'],
                 'draft_rate':meta['spec_num_correct_drafts']/meta['spec_num_proposed_drafts'],
                 'P1_P2_P3':[sum(hist[i:])/sum(hist) for i in (1,2,3)],
                 'TTFT_s':smoke['ttft_s'],'TPOT_ms':smoke['tpot_ms'],'E2E_s':smoke['e2e_s'],
                 'output_tok_s':32/smoke['e2e_s'],'source_file':str(preview/'on/initial_smoke_085.json'),
                 'status':'completed','notes':'仅32输出，不能外推长输出或真实业务'})


def num(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def source(row):
    p=row['source_file']
    return f'[原始结果]({p})'


def acceptance_table(rows):
    lines=['| 日期/轮次 | 场景 | 输入方式 | 实际输入/目标输出 | 并发/请求 | AL | 草稿接受率 | P1 / P2 / P3 | 来源 |',
           '|---|---|---|---|---|---|---|---|---|']
    rounds = {'历史9月26日':'历史短测','历史9月28日':'历史窗口测',
              '10月11日原生工具':'原生工具','10月12日早期数据对比':'早期对比',
              '10月12日旧四组':'旧四组','10月12日逐位置补测':'逐位置补测',
              '10月12日原生短复测':'原生短复测','10月12日原生完整重跑':'原生完整重跑',
              '配置短探针':'配置探针','初始短探针':'初始探针'}
    for r in rows:
        lengths=r.get('input_lengths')
        workload=('/'.join(map(str,lengths))+' → '+str(r.get('output_target'))) if lengths else r['label']
        positions=' / '.join(num(v,4) for v in r['P1_P2_P3']) if r.get('P1_P2_P3') is not None else '—'
        conc=f"{r.get('concurrency','—')}/{r.get('requests_measured','—')}"
        lines.append(f"| {r['date'][5:]} {rounds[r['category']]} | {r['label']} | {r['dataset']} | {workload} | {conc} | {num(r.get('AL'))} | {num(r.get('draft_rate'),4)} | {positions} | {source(r)} |")
    return '\n'.join(lines)


def main():
    build_rows()
    path=ROOT/'MTP全部实验汇总_20261012.md'
    lines=['# MTP全部实验汇总（截至2026-10-12完整重跑结束）','',
           '汇总本地历史归档和当前实验中有数值结果的工作负载及配置短探针。没有发送请求或启动服务。每行对应保存的测量轮次；重复轮次保留，不复制其他轮次的逐位置率。聊天验收、前缀预热和配置捕获内部请求不作为独立workload计数。','',
           'AL=有接受计数的实际输出token总数/verify总数；草稿接受率=correct/proposed。P1/P2/P3是无条件位置概率，只在完整直方图可用时计算。缺失写“—”；不是0，也不从AL反推。历史OFF没有接受统计，单列性能。', '',
           '## 1. 最新四组完整重跑','',acceptance_table([r for r in ROWS if r['category']=='10月12日原生完整重跑']), '',
           '12条请求全部完成。长输出完整32768token；只3种输出ID，99.997%为两token周期，AL≈4。缓存组实测59.765625%。原生random-ids输入仍可导致退化，不能当正常业务接受率。','',
           '## 2. 本次旧四组及逐位置补测','',acceptance_table([r for r in ROWS if r['category'] in ('10月12日旧四组','10月12日逐位置补测')]),'',
           '上述w3行的实际输出目标为2048，不能当完整1K/32K。旧四组w2/w3与逐位置补测即使AL相同，也不能将补测P1/P2/P3回填到缺直方图的旧记录。旧轮并发4的w1发12条、并发1的w2/w3发2条、w4发4条，均未按AboutModels当前倍率；最新重跑为8/1/1/2。','',
           '## 3. 原生random-ids短复测','',acceptance_table([r for r in ROWS if r['category']=='10月12日原生短复测']), '',
           '每组仅2条；默认text先decode再重新分词，实际输入1106/1088，与直接ID内容也不同。这是输入差异，不能解释为HTTP字段本身改变MTP算法。','',
           '## 4. 早期随机输入与ShareGPT截取对比','',acceptance_table([r for r in ROWS if r['category']=='10月12日早期数据对比' and r['status']=='completed']), '',
           '这些文件虽然标random，实际是自写逐token独立均匀采样，不能标成SGLang原生random-ids。real是截取ShareGPT对话文本转ID，未用完整聊天模板，不代表正式业务数据集评测。32K/256早期并发4尝试未得到有效测量，列在失败记录中。','',
           '1K/1024两组沿用了前一组输入，实际缓存命中不同：random为87.50%，real为65.625%；其他上述256输出组命中为0。因此两组延迟和吞吐不能作为同缓存条件的输入对照。','',
           '## 5. 9月历史MTP ON实验','',acceptance_table([r for r in ROWS if r['category'] in ('历史9月26日','历史9月28日') and r['mode']=='ON']), '',
           '9月26日为25单机ON→OFF，每场景2条；decode实际2048输出。9月28日为23 OFF、25 ON各单机对照，每场景2700秒：ON主要长请求观察到21184输出，未完成32768目标；表中请求数包含极短的窗口边界记录。8K组接受计数覆盖97.66%的窗口输出；因此AL及逐位置率对应有接受计数的子集，不能把窗口17826输出直接除verify。其他两个ON场景接受输出覆盖100%。','',
           '## 6. 10月11日原生benchmark记录','',acceptance_table([r for r in ROWS if r['category']=='10月11日原生工具']), '',
           '两组各8条，总输入38412、总输出1134；range_ratio=0，属于变长，不能称固定8K/256。AL来自服务累计avg_spec_accept_length，可能包含预热/历史请求；逐位置率和本workload的干净草稿计数无法恢复，保留为原工具报告值。','',
           '## 7. 配置验证短探针','',acceptance_table([r for r in ROWS if r['category'] in ('配置短探针','初始短探针')]), '',
           'C/D/E/F/G分别验证v0.5.16 normal+MEM0.92+Graph关、v0.5.16 normal+MEM0.88+Graph关、v0.5.16 normal+MEM0.92+Graph请求开、v0.5.19 normal+MEM0.92+Graph请求开、v0.5.20 normal+MEM0.92+Graph请求开。实际normal下Graph全部关闭；34/32只是功能验证，不能报成完整workload。','',
           '## 8. 性能完整表（含历史OFF）','',
           '| 轮次 | 场景 | MTP | 总输出tok/s | TTFT均s | TPOT均ms | E2E均s | 实际缓存命中 | 来源 |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in ROWS:
        if r.get('status')=='failed':continue
        lines.append(f"| {r['category']} | {r['label']} | {r['mode']} | {num(r.get('output_tok_s'))} | {num(r.get('TTFT_s'))} | {num(r.get('TPOT_ms'))} | {num(r.get('E2E_s'))} | {num(100*r['cache_hit_ratio'],2)+'%' if r.get('cache_hit_ratio') is not None else '—'} | {source(r)} |")
    lines += ['', '吞吐均不除以NPU：9月26日是单请求输出/耗时的均值，9月28日是45分钟窗口吞吐，10月12日通常是客户端整组输出/墙钟。它们口径不同，不能直接比较倍率。旧客户端TPOT按(E2E−TTFT)/(输出−1)求请求均值；最新完整重跑用首末新增ID时间，E2E另保留流结束开销。w1最新吞吐墙钟在CPU恢复自最早请求开始到最晚请求结束，未包含线程池启动和最终汇总。','',
              '9月26日E2E从对应ON/OFF目录下requests.jsonl的measurement记录直接取均值，排除flush和prefix_prime。部分其他历史结果未存独立E2E均值，本表不由TTFT+平均TPOT反推；9月28日完整配对E2E保存在历史核心指标表。10月11日性能或延迟字段未纳入表中的“—”不是不存在，是未采用为跨轮统一指标。','',
              '## 9. 未形成有效MTP测量的尝试','',
              '| 尝试 | 结果 | 是否有可用AL/P1/P2/P3 |',
              '|---|---|---|',
              '| v0.5.18，DP2+DP attention，auto，Graph开/关两次 | LowLatency Combine/Dispatch错误 | 无 |',
              '| A：v0.5.16 auto、Graph开、MEM0.88 | LowLatency Combine tiling错误 | 无 |',
              '| B：v0.5.16 auto、Graph关、MEM0.88 | LowLatency Dispatch tiling错误 | 无 |',
              '| H：v0.5.20 auto、Graph开、MEM0.92 | LowLatency相关错误 | 无 |',
              '| ROUTE_A：v0.5.19、DP2/EP1、DeepEP关 | loader错误/DP flags遗漏，未形成有效测量 | 无 |',
              '| 23+20双机TP32/EP32：ON初始、ON改通信参数、OFF | 三次均aclnnNotifyDispatch错误 | 无；OFF也失败，不能全归因于MTP |',
              '| 早期32K/256、并发4 | 结果aggregate为空，未形成有效测量 | 无 |',
              '| native_full聊天预检首次 | 客户端模板返回格式处理错误，HTTP400；修复后正式四组完成 | 不算一次新workload |',
              '| native_full w1首次汇总 | 缺尾部零桶导致汇总异常；CPU补零恢复，原请求全部完成 | 原计数有效，未重发，不重复计数 |','',
              '配置细节与原始启动记录见[MTP配置排列组合与AboutModels对照](MTP配置排列组合与AboutModels对照_20261012.md)。早期loader重试等不是新配置收益实验，不将重复启动算作新的算法方案。','',
              '## 10. 读表结论','',
              '原生输入口径已经纠正，完整32K输出也已跑完；但合成random-ids仍触发长循环，AL≈4不能代表真实业务。8K组循环也可能伴随低草稿接受率，不能把所有高低都归结为循环。ShareGPT截取与原生合成数据不是同一种输入。', '',
              '表中轮次的镜像、MEM、DP、Graph、并发、输入和统计覆盖各不相同；这是一份实测记录清单，不是同条件MTP收益榜单。当前完整重跑仅ON，没有同配置OFF。关于权重、输出精度和参考接受率的进一步定位，参见[MTP问题定位](MTP问题定位_20261012.md)及[MTP接受率复核](MTP接受率复核_20261012.md)，本汇总不以计数校验代替精度证明。','',
              f'机器可读明细：[JSON](MTP全部实验汇总_20261012.json)、[CSV](MTP全部实验汇总_20261012.csv)。共{len(ROWS)}条数值/失败workload记录；其中{sum(r.get("status")!="failed" for r in ROWS)}条有结果、{sum(r.get("status")=="failed" for r in ROWS)}条早期负载无有效结果。失败配置表单独统计，不伪造数值。']
    path.write_text('\n'.join(lines)+'\n')
    (ROOT/'MTP全部实验汇总_20261012.json').write_text(json.dumps({'scope':'saved local workloads and functional probes; exclude chat/warmup/internal capture as independent benchmarks', 'rows':ROWS},ensure_ascii=False,indent=2))
    fields=sorted({key for row in ROWS for key in row})
    with (ROOT/'MTP全部实验汇总_20261012.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print('rows',len(ROWS),'written',path)
    for category in dict.fromkeys(r['category'] for r in ROWS):
        print(category,sum(r['category']==category for r in ROWS))


if __name__=='__main__':
    main()
