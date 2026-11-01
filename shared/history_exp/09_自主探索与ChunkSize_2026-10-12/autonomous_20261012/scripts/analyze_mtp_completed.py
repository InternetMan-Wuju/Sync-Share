#!/usr/bin/env python3
"""Offline evidence by trajectory type, with explicit limits on EOS inference."""
from report_labels import humanize_report
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')


def stats(records):
    valid=[x for x in records if not x.get('error') and x['meta'].get('spec_verify_ct')]
    v=sum(x['meta']['spec_verify_ct'] for x in valid)
    h=[0]*4
    for x in valid:
        for i,count in enumerate(x['meta'].get('spec_correct_drafts_histogram') or []):h[i]+=count
    return {'requests':len(records),'verified_requests':len(valid),'verify_count':v,
            'output_count':sum(x['completion_tokens'] for x in records),
            'AL':sum(x['completion_tokens'] for x in valid)/v if v else None,
            'positions':[sum(h[i:])/v for i in (1,2,3)] if v else None}


def fmt(value):
    return '—' if value is None else f'{value:.3f}'


def main():
    groups=[]
    for p in sorted((ROOT/'results').glob('*/*/result.json')):
        info_path=p.parent.parent/'server_info.json'
        if info_path.exists() and not json.loads(info_path.read_text()).get('speculative_algorithm'):continue
        d=json.loads(p.read_text());records=d['requests']
        if any(v.get('failures') for v in d.get('validation',[])) or any(x.get('error') or not x.get('stream_done') for x in records):continue
        cycle=[];other=[]
        for x in records:
            shape=x.get('output_shape',{}).get('periodic_suffix')
            (cycle if shape and shape['fraction']>=.5 else other).append(x)
        groups.append({'run':p.parent.parent.name,'group':p.parent.name,'source':str(p),
                       'all':stats(records),'majority_periodic':stats(cycle),'remaining':stats(other),
                       'ignore_eos':d['ignore_eos'],
                       'eos_requests':sum(x.get('first_eos_position') is not None for x in records),
                       'tokens_after_eos':sum(x.get('tokens_after_first_eos',0) for x in records),
                       'length_finished':sum((x['meta'].get('finish_reason') or {}).get('type')=='length' for x in records)})
    audits=[]
    for p in (ROOT/'results').rglob('summary.json'):
        try:d=json.loads(p.read_text())
        except Exception:continue
        if 'prefill_checks' not in d:continue
        checks=d['prefill_checks'];comp=[row for x in checks for row in x.get('comparisons',[])]
        rechecks=d.get('mismatch_prefill_rechecks',[]);pinned=defaultdict(list)
        for r in rechecks:
            if not r.get('error'):
                pinned[(r['group'],r['index'],r['position'],r.get('routed_dp_rank'))].append(r['token'])
        recheck_pairs=[{'group':k[0],'index':k[1],'position':k[2],'routed_dp_rank':k[3],
                        'tokens':v,'stable':len(set(v))==1} for k,v in pinned.items() if len(v)>=2]
        audits.append({'source':str(p),'completed':d.get('completed',False),
                       'mismatch_prefill_rechecks':rechecks,'repeated_prefix_pairs':recheck_pairs,
                       'unstable_prefix_pairs':sum(not x['stable'] for x in recheck_pairs),
                       'checks':len(checks),'aligned_checks':sum(bool(x.get('alignment_valid')) for x in checks),
                       'tokens_checked':len(comp),'mismatches':sum(not x['match'] for x in comp),
                       'reported_nonfinite':sum(not x['reported_values_finite'] for x in comp),
                       'errors':[x.get('error') for x in checks if x.get('error')],
                       'windows':d.get('windows',[])})
    lines=['# MTP轨迹分组与主模型核对','',
           '每条请求的终态计数在CPU复算。精确周期尾段占输出至少50%归入主要循环；其余不自动等同于正常业务。EOS前后没有原始逐步草稿计数，不拆分虚构接受率。','',
           '| 轮次 | 输入组 | 总AL | 主要循环数/全部 | 循环AL | 其余AL | 首次EOS后仍生成token | 达长度上限数 |',
           '|---|---|---|---|---|---|---|---|']
    for g in groups:
        lines.append(f"| {g['run']} | {g['group']} | {fmt(g['all']['AL'])} | {g['majority_periodic']['requests']}/{g['all']['requests']} | {fmt(g['majority_periodic']['AL'])} | {fmt(g['remaining']['AL'])} | {g['tokens_after_eos']} | {g['length_finished']} |")
    lines+=['','## 独立主模型prefill核对','',
            '把保存的MTP输出作为已知输入，按同一前缀计算主模型top2并逐token核对。只有返回token IDs能完整对齐时才比较。不同前向/批量可能有数值差异，应结合top1-top2间隔判断；核对有限样本不能证明全模型准确率。','',
            '| 数据 | 完成 | 对齐/检查 | 比较token | 不一致 | 非有限 | 不稳定前缀对照数 | 客户端/接口错误 |',
            '|---|---|---|---|---|---|---|---|']
    for a in audits:
        lines.append(f"| [原始核对]({a['source']}) | {a['completed']} | {a['aligned_checks']}/{a['checks']} | {a['tokens_checked']} | {a['mismatches']} | {a['reported_nonfinite']} | {a['unstable_prefix_pairs']} | {'; '.join(a['errors'])} |")
    lines+=['','完整聊天的允许EOS与强制输出对照，如果两组都达到256长度上限，不能由相同AL宣称EOS毫无影响；这只是窗口内没有正常结束。','']
    (ROOT/'MTP轨迹分组与主模型核对.md').write_text(humanize_report('\n'.join(lines)))
    (ROOT/'MTP轨迹分组与主模型核对.json').write_text(json.dumps({'groups':groups,'audits':audits},ensure_ascii=False,indent=2))
    print('trajectory groups',len(groups),'prefill audits',len(audits))


if __name__=='__main__':main()
