#!/usr/bin/env python3
"""Publish conclusions only after completed experiments and offline validation."""
from report_labels import humanize_report
import datetime
import json
import statistics
from collections import defaultdict
from pathlib import Path
from audit_mtp_workload_coverage import main as audit_mtp_workload_coverage
from audit_chunk_requirement_coverage import main as audit_chunk_requirement_coverage

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
CHUNK = ROOT.parent / 'Qchunksize'


def read(path):
    return json.loads(path.read_text())


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    for name in ('overnight_status.json', 'chunk_repeat_status.json', 'final_mtp_isolation_status.json'):
        assert read(ROOT / name).get('completed'), name
    valid = read(ROOT / '数据验收与时间戳核对.json')
    assert all(r['passed'] for r in valid['mtp_groups'] + valid['chunk_cells'])
    assert all(r['same_inputs'] and r['same_service_except_chunk'] for r in valid['comparison_consistency'])
    chunk_evidence = read(CHUNK / 'ChunkSize负载与阶段分析.json')
    groups = chunk_evidence['groups']
    assert len(groups) == 5 and all(len(g['candidates']) == 4 for g in groups)
    base = ROOT / 'results/mtp-v19-dp2-no-plan-lp-safe'
    isolated = ROOT / 'results/mtp-v19-dp2-no-multistream'
    cfg_a = read(ROOT / 'logs/cyt-auto-mtp-v19-dp2-no-plan-lp-safe/config.json')
    cfg_b = read(ROOT / 'logs/cyt-auto-mtp-v19-dp2-no-multistream/config.json')
    differences = {k: [cfg_a.get(k), cfg_b.get(k)] for k in set(cfg_a) | set(cfg_b)
                   if k not in ('name', 'started_at', 'env') and cfg_a.get(k) != cfg_b.get(k)}
    env_diff = {k: [cfg_a['env'].get(k), cfg_b['env'].get(k)] for k in set(cfg_a['env']) | set(cfg_b['env'])
                if cfg_a['env'].get(k) != cfg_b['env'].get(k)}
    assert not differences, differences
    assert env_diff == {'SGLANG_NPU_USE_MULTI_STREAM': ['1', '0']}, env_diff

    cycle = read(base / 'native_longinput_1k1024/result.json')['requests'][0]
    control = read(isolated / 'native_longinput_1k1024/result.json')['requests'][0]
    framing = {kind: read(isolated / f'framing_pair_{kind}_eos256/result.json')['requests']
               for kind in ('raw', 'chat')}
    assert len(framing['raw']) == len(framing['chat']) == 1
    raw_payload = read(Path(framing['raw'][0]['payload_file']))
    chat_payload = read(Path(framing['chat'][0]['payload_file']))
    assert chat_payload['input_ids'][3:-3] == raw_payload['input_ids'], 'paired random bodies differ'
    assert raw_payload['sampling_params'] == chat_payload['sampling_params'], 'paired sampling policies differ'
    replay = read(isolated / 'fixed_baseline_prefix_replay/summary.json')
    assert replay['completed'] and len(replay['probes']) == 16
    by_prefix = defaultdict(dict)
    for row in replay['probes']:
        assert row['actual_dp_rank'] == row['routed_dp_rank']
        original_payload = read(Path(row['source_payload']))
        replay_payload = read(Path(row['response_source']).with_name(
            Path(row['response_source']).name.replace('.response.json', '.payload.json')))
        assert original_payload == replay_payload, 'fixed-prefix replay payload drifted'
        key = (row['group'], row['index'], row['position'])
        by_prefix[key].setdefault(row['routed_dp_rank'], []).append(row)
    prefix_comparisons = []
    for key, routes in sorted(by_prefix.items()):
        assert set(routes) == {0, 1} and all(len(rows) == 2 for rows in routes.values())
        before = {route: [r['baseline_token'] for r in rows] for route, rows in routes.items()}
        after = {route: [r['token'] for r in rows] for route, rows in routes.items()}
        stable_before = all(len(set(tokens)) == 1 for tokens in before.values())
        stable_after = all(len(set(tokens)) == 1 for tokens in after.values())
        prefix_comparisons.append({'group': key[0], 'index': key[1], 'position': key[2],
            'baseline_tokens_by_dp': before, 'isolated_tokens_by_dp': after,
            'stable_baseline': stable_before, 'stable_isolated': stable_after,
            'baseline_dp_difference': before[0][0] != before[1][0] if stable_before else None,
            'isolated_dp_difference': after[0][0] != after[1][0] if stable_after else None})
    retained_dp_differences = sum(r['baseline_dp_difference'] is True and
                                  r['isolated_dp_difference'] is True for r in prefix_comparisons)
    resolved_dp_differences = sum(r['baseline_dp_difference'] is True and
                                  r['isolated_dp_difference'] is False for r in prefix_comparisons)
    cycle_same_input = read(Path(cycle['payload_file'])) == read(Path(control['payload_file']))
    assert cycle_same_input, 'multi-stream cycle probe inputs/policy drifted'
    cycle_same_output = cycle['output_ids'] == control['output_ids']
    cycle_same_counts = all(cycle['meta'].get(k) == control['meta'].get(k) for k in (
        'spec_verify_ct', 'spec_num_correct_drafts', 'spec_num_proposed_drafts',
        'spec_correct_drafts_histogram'))
    audit = read(isolated / 'target_prefill_audit/summary.json')
    checks = audit['prefill_checks']
    assert audit['valid_comparison']
    weight_pairs = read(ROOT / 'MTP自主探索实测.json')['weight_comparisons']
    combined = read(ROOT.parent / 'Q1MTPExp/MTP全部实验汇总_含自主探索.json')
    release_path = ROOT / '节点释放核对.json'
    release = read(release_path) if release_path.exists() else None
    review_path = ROOT / 'review_20261013/MTP一般情况检索与补充诊断.json'
    review = read(review_path) if review_path.exists() else None
    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    lines = ['# MTP与ChunkSize自主实验最终总结', '', f'更新时间：{stamp}。仅使用23节点；容器和原始数据保留。', '',
        '阅读说明：测试场景按“输入多少 → 输出多少”命名，长度单位均为token。服务配置用中文描述；原代号、四类场景及缩写见[负载与配置名称说明](负载与配置名称说明.md)。', '',
        '## 可以先看这几句话', '',
        '1. 权重确实有坏块，已用公开量化源校验并通过只读外挂恢复。恢复后随机输入的接近满接受现象仍然存在，坏权重不是它的唯一原因。',
        '2. 最典型的高接受请求是在反复续写“制表符 + #”。它没有遇到EOS；主模型本身也支持这个循环，草稿因此很容易连续猜中。AL接近4有真实计数支持，不能当作正常业务普遍收益。',
        '3. 正常完整聊天也有AL超过3的实例。文档中的AL约2和两组三位置率不是任意输入都必须满足的规律；缺少同一checkpoint、输入、采样及统计定义，不能强行凑数。',
        '4. ChunkSize完成四档、五个固定场景/并发的20单元及基线/候选复测。下面按整套服务吞吐比较，不除以NPU；配置与榜单默认的差异已明示。', '',
        '## 权重：修了什么，能证明什么', '',
        '公共模型未修改。在workspace保存329个差异块的恢复内容，并保留4个经公开HTTPS Range确认正确的公共原块。182个权重文件对应公开源整文件SHA256；180个结合历史整文件SHA及块级校验，两个疑点文件本次完整虚拟重建后重算整SHA。运行时每个读取块再次验SHA，失败不回退到坏字节。',
        'config、tokenizer、量化描述与公开源一致；三个已知注意力坏块的30900个NaN/Inf恢复后消失。这证明checkpoint字节来源与加载一致，不能替代全模型精度或KVV≤3%的验收。', '',
        '| 同输入正式短组 | 恢复前AL | 恢复后AL | 输入SHA一致请求 |', '|---|---|---|---|']
    for row in weight_pairs:
        pairs = row['paired_outputs']
        if not all(p['same_input_sha256'] is True for p in pairs):
            continue
        lines.append(f"| {row['group']} | {fmt(row['before']['AL'])} | {fmt(row['after']['AL'])} | {len(pairs)}/{len(pairs)} |")
    lines += ['', '恢复前后并不是所有AL都下降。旧native_exact短探针未留逐请求输入SHA/payload，只作为参考数字；1K/32K旧长输出与本次1K/1024短诊断也不混作同长度因果对照。', '',
        '## MTP接近满接受：一条请求逐项核算', '',
        '| 项目 | 值 |', '|---|---|',
        f"| 输出token / 验证轮数 | {cycle['completion_tokens']} / {cycle['meta']['spec_verify_ct']} |",
        f"| 接受0/1/2/3草稿的轮数 | {cycle['meta']['spec_correct_drafts_histogram']} |",
        f"| AL | {fmt(cycle['meta']['spec_accept_length'],6)} |",
        f"| P1/P2/P3 | {' / '.join(fmt(p,6) for p in cycle['pos_rates'])} |", '',
        '固定NEXTN steps=3、topk=1、draft_tokens=4，一轮最多3个草稿加1个主模型token。三个位置率是以全部验证轮为分母的无条件率，因此AL≈1+P1+P2+P3；短请求和首尾截断有计数边界差异。AL不是百分比，也不是只数草稿。',
        '原生random-ids是随机起点的连续词表ID序列；默认文本客户端还可能decode后重分词。它不是有语义的业务对话，也不同于独立uniform IDs。保存的典型输入引出两token循环，占输出99.90%；没有EOS，所以不能将该条归因于ignore_eos。',
        '独立主模型prefill评分和固定位置续写支持该循环；循环稳定段的128-token续写AL=4。终态直方图只能核算总数，不能定位少数拒绝发生在哪一轮。', '',
        '历史完整1K/32K的AL=3.998536也有直接证据：32768输出中32767个token组成两token周期，V=8195，h=[3,1,1,8190]；没有EOS。该输入与恢复后短探针逐ID相同，循环内容不同。见[历史长输出证据](MTP历史长输出的直接证据.md)。输出长度不同，不据两次总AL推断权重修复贡献。', '',
        '## 完整真实对话：高于3也不一定是循环', '',
        '下面使用预先选定的首两条完整ShareGPT对话、完整聊天模板、允许EOS、上限2048。全部实际自然结束、没有检出精确周期；这些条件与强制长度random-ids不同，不作同负载速度比较。', '',
        '| 服务配置 | 固定对话序号 | 实际输出token | AL | P1/P2/P3 |', '|---|---|---|---|---|']
    for tag in ('mtp-v16-auto-cap64-mem87-hccl650', 'mtp-v16-auto-cap64-mem90-hccl650',
                'mtp-v19-dp2-no-plan-lp-safe'):
        business = read(ROOT / 'results' / (tag + '_business/sharegpt_chat_complete/result.json'))
        for row in business['requests']:
            assert row['meta']['finish_reason']['type'] == 'stop'
            assert row['first_eos_position'] == row['completion_tokens'] - 1
            assert row['output_shape']['periodic_suffix'] is None
            lines.append(f"| {tag} | {row['index']} | {row['completion_tokens']} | {fmt(row['meta']['spec_accept_length'])} | {' / '.join(fmt(v,4) for v in row['pos_rates'])} |")
    lines += ['', '## NPU多流单变量与完全相同前缀重放', '',
        '在“关闭调度重叠、使用常规评分”方案上，只把NPU多流从开启改为关闭（SGLANG_NPU_USE_MULTI_STREAM=1→0）；启动参数及其余环境已逐项核对。保留所有原断言、plan_stream=0和标准Torch评分，不钳制token、不修改验证算法。', '',
        '| 循环输入 | AL | P1/P2/P3 | 主要周期尾段 |', '|---|---|---|---|']
    for name, row in [('多流开', cycle), ('多流关', control)]:
        lines.append(f"| {name} | {fmt(row['meta'].get('spec_accept_length'))} | {' / '.join(fmt(v,4) for v in row.get('pos_rates') or [])} | {row['output_shape']['periodic_suffix']} |")
    lines += ['', f'同一输入和采样策略已核对；两次完整输出IDs一致：{cycle_same_output}，验证轮数/直方图/草稿计数一致：{cycle_same_counts}。'
        + ('关闭多流没有消除该循环，也没有降低这条请求的接受率。' if cycle_same_output and cycle_same_counts else
           '输出或计数存在变化；该诊断结论以两次原始结果为准，不泛化到其他请求。'), '',
        f"关闭多流后的bulk-prefill核对：{len(checks)}条轨迹，"
        f"{sum(len(c.get('comparisons',[])) for c in checks)}个输出位置，"
        f"{sum(c.get('mismatches',0) for c in checks)}处与保存输出的argmax不同。有限top2及字节校验都不能证明全logits/全模型精度正确。", '',
        '| 原基线固定前缀 | DP0原基线 → 多流关 | DP1原基线 → 多流关 |', '|---|---|---|']
    for key, routes in sorted(by_prefix.items()):
        values = []
        for route in (0, 1):
            rows = routes[route]
            values.append(f"{[r['baseline_token'] for r in rows]} → {[r['token'] for r in rows]}")
        lines.append(f'| {key[0]} #{key[1]} pos{key[2]} | {values[0]} | {values[1]} |')
    lines += ['', '这16次复测使用原基线完全相同的input_ids和DP路由，每个前缀/路由两次并清缓存；实际dp_rank与全部payload已检查。位置来自基线差异诊断，不是随机精度样本。',
        f"16次中{sum(r['same_as_baseline'] for r in replay['probes'])}次预测与原基线相同。4个前缀中，原基线DP差异仍保留{retained_dp_differences}个、变为两个稳定DP一致{resolved_dp_differences}个；关闭多流后重复不稳定{sum(not r['stable_isolated'] for r in prefix_comparisons)}个。"
        + ('关闭多流未解决已复现的DP间预测差异；不能将该差异直接归因于多流。' if retained_dp_differences else
           '这些有限诊断位置不能据此证明全部DP输出一致或指定单一kernel。'), '',
        '## 相同随机正文，只改变聊天边界', '',
        '两组严格保持max_new_tokens=256、temperature=0、ignore_eos=False及raw body一致，仅一组添加已保存GLM模板的角色边界；模板后的输入长度必然不同。这是任务格式诊断，不替换标准workload。', '',
        '| 格式 | 输出token | 结束原因 | AL | P1/P2/P3 |', '|---|---|---|---|---|']
    for kind in ('raw', 'chat'):
        for row in framing[kind]:
            lines.append(f"| {kind}（同一正文） | {row['completion_tokens']} | {row['meta'].get('finish_reason')} | {fmt(row['meta'].get('spec_accept_length'))} | {' / '.join(fmt(v,4) for v in row.get('pos_rates') or []) or '—'} |")
    framed_raw, framed_chat = framing['raw'][0], framing['chat'][0]
    framing_explanation = ''
    if framed_raw['output_shape']['periodic_suffix'] and not framed_chat['output_shape']['periodic_suffix']:
        framing_explanation = ('本条raw仍为两token循环；添加聊天边界后未检出该精确循环。'
            '聊天边界后的输出正文及EOS保存在原始结果，可检查其实际任务内容；不能把随机正文的输出当作正常业务质量验收。')
    lines += ['', f"严格配对结果：raw输出{framed_raw['completion_tokens']}token，chat输出{framed_chat['completion_tokens']}token；"
        f"AL为{fmt(framed_raw['meta'].get('spec_accept_length'))} → {fmt(framed_chat['meta'].get('spec_accept_length'))}。"
        + framing_explanation +
        '若在prefill即遇EOS而没有verify，接受率留空，不能把它算成0或称为达到AL≈2；只改边界改变了模型面临的任务，不能替换原生固定长度压测。', '',
        '## 标准对齐的实际限制', '',
        '已实际尝试AboutModels默认镜像、DP2、DP attention、DeepEP auto和Graph。capacity128/HCCL650需要825MB而失败；只关Graph仍失败；HCCL1000及1536又分别触发通信/动态内存不足。capacity64、HCCL650、小Graph、MRR8可启动，mem0.87/0.90每DP KV分别7296/26496，后者仍不足32K输入。',
        'v19/DP2 normal、无Graph可得到每DP KV40448，正常问答和短组通过。不能把缩小配置的成功写成精确默认四负载均可跑。镜像v0.5.16标签实际server-info是0.5.17.dev386+gc5bd3d7dc，表中按实际版本保留。',
        '尚未有可执行定义的KVV≤3%精度基准，也没有与参考AL2.0完全同条件的数据。用户此前已取消新的正式MTP OFF性能网格，本轮没有重做，故不报恢复后MTP开关的净速度收益。', '',
        'MTP.md的100K总输入/90%前缀场景未测：本轮成功配置的KV上限不足100K。运行成功不意味着满足KVV、AL≈2.0及指定延迟门槛。', '',
        '## ChunkSize：完整矩阵与复测', '',
        '统一配置：23节点单机，SGLang 0.5.19，张量/专家并行各16路、数据并行1路；关闭MTP，DeepEP自动通信，计算图实际生效。静态显存比例90%，同时运行请求上限16，上下文上限131072个token，一次输入处理的总token预算上限32768，KV缓存容量77824个token。各组只改变输入分块大小。默认配置的数据并行2路、运行请求上限64、上下文上限180000和较大计算图与本次不同；详情见[实际配置](../Qchunksize/RUN_CONFIG_20261013.md)。',
        '先按测试形状预热再清缓存，带缓存的场景另预热共享前缀。W1和W4的正式请求总数为并发数的2倍，W2为并发数的1倍；不人工控制发请求时机。下面每格为均值〔最小～最大；轮数〕，单位是整套服务每秒输出token数。', '',
        '| 场景 / 并发 | Chunk1024 | Chunk4096 | Chunk8192 | Chunk16384 |', '|---|---|---|---|---|']
    recommendations = []
    for group in groups:
        candidates = {r['chunk']: r for r in group['candidates']}
        values = []
        for chunk in (1024, 4096, 8192, 16384):
            c = candidates[chunk]
            values.append(f"{fmt(c['throughput_mean'])}〔{fmt(c['throughput_min'])}～{fmt(c['throughput_max'])}；{c['repeats']}轮〕")
        lines.append(f"| {group['scene']} / C{group['concurrency']} | " + ' | '.join(values) + ' |')
        best = group['candidates'][0]
        near = [c['chunk'] for c in group['candidates'] if c['throughput_mean'] >= .95 * best['throughput_mean']]
        recommendations.append({'scene': group['scene'], 'concurrency': group['concurrency'],
            'highest_observed_mean_chunk': best['chunk'], 'near_5pct_chunks': near,
            'best_repeats': best['repeats'],
            'gain_over_16384_pct': (best['throughput_mean']/candidates[16384]['throughput_mean']-1)*100})
    lines += ['', '| 场景 / 并发 | 实测均值最高chunk | 相对16384均值 | 均值差5%内的候选 |', '|---|---|---|---|']
    for row in recommendations:
        lines.append(f"| {row['scene']} / C{row['concurrency']} | {row['highest_observed_mean_chunk']} | {fmt(row['gain_over_16384_pct'],2)}% | {row['near_5pct_chunks']} |")
    lines += ['', '下面比较16384与本场景实测均值最高候选的阶段均值（秒；各重复轮等权）。', '',
              '| 场景 / 并发 | 候选chunk | 接收至首次forward：基线 → 候选 | Prefill墙钟：基线 → 候选 | Decode至结束：基线 → 候选 | 客户端E2E：基线 → 候选 |',
              '|---|---|---|---|---|---|']
    for row in recommendations:
        pool = [c for c in chunk_evidence['cells'] if c['scene'] == row['scene'] and c['concurrency'] == row['concurrency']]
        pairs = []
        for key in ('server_before_forward_s', 'server_prefill_span_s', 'server_decode_and_finish_s', 'e2e_s_mean'):
            values = []
            for chunk in (16384, row['highest_observed_mean_chunk']):
                selected = [c for c in pool if c['chunk_requested'] == chunk]
                samples = [c['aggregate'][key] if key == 'e2e_s_mean' else c['aggregate']['timestamp_breakdown_mean_s'][key]
                           for c in selected]
                values.append(statistics.mean(samples))
            pairs.append(f'{fmt(values[0])} → {fmt(values[1])}')
        lines.append(f"| {row['scene']} / C{row['concurrency']} | {row['highest_observed_mean_chunk']} | " + ' | '.join(pairs) + ' |')
    lines += ['', '这不是统计显著性检验。样本很少、重复区间重叠或均值相差很小时，只推荐候选范围；没有复测的档位只作初测，不能宣称生产最优。不同场景和并发不能混算倍率。',
        '16K基线的8K/1K、C2确实发生波动：初测25.07、复测29.58 tok/s。相同并发不保证scheduler组批相同；复测第一批拆成两个8192-token prefill，而初测是一个16384-token批次。没有人工控制发送时机，不能只取较慢基线放大收益。见[基线波动证据](Chunk基线波动的批次说明.md)。',
        '小chunk把prefill拆为更多前向，可能降低单次阻塞却增加调度/算子开销；大chunk有相反取舍。本轮混合chunk关闭，必须结合实际prefill墙钟和decode跨度解释结果，不能仅凭总吞吐编造干扰或纯kernel收益。各组阶段均值、queue_time、回退计数和10秒负载采样在[阶段分析](../Qchunksize/ChunkSize负载与阶段分析.md)；采样不能代替逐步调度trace。',
        '1K/32K以decode为主，本轮没有重复小时级Chunk长输出扫描，因此覆盖三类负载、五个并发组合，并非默认全部四负载。p95是这些少量请求的描述统计，不是生产尾延迟保证。', '',
        '## 已排查的实现错误与剩余边界', '',
        '- 原诊断断言在plan stream过早读取compute产出的int64 ID。只移动检查到已有汇合点，或关闭plan并保留原断言后，均能完成；没有clamp或造token。',
        '- 融合input-logprob评分触发Bisheng编译SIGSEGV。统一使用镜像已有SGLANG_ENABLE_FAST_INPUT_LOGPROBS=0的Torch路径完成评分；未套未合并PR冒充修复。',
        '- 多DP、prefill/verify路径有输出差异证据；router BF16、dispatch dtype和多流相关公开PR只作线索。没有通过随意叠加补丁把某个AL调成2。',
        '- 本轮未开启SGLANG_SIMULATE_ACC_LEN；接受计数逐请求由直方图独立复算，并有输出轨迹与主模型核对。公开旧GPU榜单模拟值不证明老师参考或本轮使用模拟。', '',
        '## 文件入口与保留的数据', '',
        f"合并MTP表现有{len(combined['rows'])}条历史/有效新增组，含原始41条历史记录。新增sanity、预热、单token核对、失败组与Chunk OFF矩阵不混入该表。", '',
        '- [全部MTP表](../Q1MTPExp/MTP全部实验汇总_含自主探索.md)，同名JSON/CSV保存配置及原始结果索引。',
        '- [ChunkSize全部实测](../Qchunksize/ChunkSize自主实验实测.md)，同名JSON/CSV。',
        '- [权重公开源校验](权重恢复与公开源校验.md)、[网络依据](网络检索依据与适用范围.md)、[数据验收](数据验收与时间戳核对.json)。',
        '- `results/`：MTP payload、输出IDs、全文、压缩SSE、接受直方图、logprobs、固定前缀核对；`logs/`：启动脚本、配置、服务日志、运行时权重审计。',
        '- `../Qchunksize/results/autonomous_20261012/`：全部基础与repeat2记录；`datasets/native_20261012/`：固定输入、种子、源码及SHA。',
        '- `figures/`：Chunk性能、E2E分解及MTP轨迹图（PNG/PDF）。',
        '- 历史完整原生四负载在`/data_lib/cyt/q1mtp/native_full_20261012_190926`；它属于恢复前历史，不当作当前恢复后正式四负载结果。', '',
        ('节点释放已核对：' + release['checked_at'] + '，本轮全部自有容器已停止并保留，NPU-SMI显示8个设备均无运行进程。'
         if release and release['owned_containers_all_stopped'] and release['npu_reports_no_running_processes_on_all_8_devices']
         else '节点释放情况见实际采集的节点释放核对.json；不根据脚本已结束推定NPU空闲。'),
        '所有自有容器只stop；其他人的脚本未执行、目录未修改。', '']
    if review:
        direct_checks = review['target_only']['comparisons'][:5]
        direct_same = all(row['same_output_ids'] for row in direct_checks)
        cycle_same = review['conclusions']['target_only_original_cycle_same_ids']
        supplement = [
            '## 2026-10-13补充检索与OFF诊断', '',
            '公开GLM-5.2真实校准表报告3草稿AL2.99（FP8/vLLM、coding、thinking开启、T=1）。它与本轮正常完整聊天AL2.4～3.3量级相容，但不是Ascend/W8A8/greedy同条件基准，也不是全模型精度验收。', '',
            f"补测5项原输入的MTP ON/OFF逐ID对照，全部一致：{direct_same}。原始随机输入的1024-token输出完全一致：{cycle_same}，OFF仍检出两token循环；额外固定循环前缀的128-token续写也已完成。该循环不以启用MTP为必要条件。DP预测差异仍未解决。", '',
            '条件位置率与无条件位置率的公式不同，不能直接逐项比数；本轮AL包含主模型token。2.8大约对应每轮接受1.8个草稿，而不是93%的草稿接受率。', '',
            '来源、统计定义、有限样本边界、6次诊断原始数据及重新实查的节点状态，见[补充检索报告](review_20261013/MTP一般情况检索与补充诊断.md)。OFF诊断不混入原86条MTP ON合表；Chunk数据未改变。', '']
        lines[lines.index('## 权重：修了什么，能证明什么'):lines.index('## 权重：修了什么，能证明什么')] = supplement
    (ROOT / '自主实验最终总结.md').write_text(humanize_report('\n'.join(lines)))
    (ROOT / '自主实验最终总结.json').write_text(json.dumps({
        'updated': stamp, 'recommendations': recommendations, 'isolation_environment_difference': env_diff,
        'fixed_prefix_replay': replay, 'chunk_cells_validated': len(valid['chunk_cells']),
        'fixed_prefix_comparisons': prefix_comparisons,
        'cycle_same_output_ids': cycle_same_output, 'cycle_same_acceptance_counts': cycle_same_counts,
        'retained_dp_differences': retained_dp_differences,
        'resolved_dp_differences': resolved_dp_differences,
        'node_release_verification': release,
        'mtp_groups_validated': len(valid['mtp_groups']), 'mtp_combined_records': len(combined['rows']),
        'supplemental_mtp_review': None if review is None else {
            'report': str(review_path.with_suffix('.md')), 'updated': review['updated'],
            'conclusions': review['conclusions'],
            'target_only_diagnostics': len(review['target_only']['comparisons'])},
        'scope': 'Observed data only; unresolved full-model accuracy and reference equivalence remain explicit.'
    }, ensure_ascii=False, indent=2))
    audit_mtp_workload_coverage()
    audit_chunk_requirement_coverage()
    print('published stage report', len(valid['chunk_cells']), 'Chunk cells;', len(combined['rows']), 'MTP rows')


if __name__ == '__main__':
    main()
