#!/usr/bin/env python3
"""Combine pinned public evidence with completed target-only diagnostics."""
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/chengyetai/workspace/current_exp/autonomous_20261012')
OUT = ROOT / 'review_20261013'


def read(path):
    return json.loads(path.read_text())


def main():
    state = read(OUT / 'target_only_status.json')
    assert state.get('completed') and state.get('stopped'), 'finish and stop the diagnostic before publication'
    target = read(OUT / 'target_only/summary.json')
    assert target['completed'] and len(target['comparisons']) == 6
    existing = read(OUT / 'existing_data_recheck.json')
    weight_audit = read(OUT / 'target_only/runtime_weight_audit.json')
    assert weight_audit['pids'] == 16 and weight_audit['unique_files'] == 182
    assert not weight_audit['error_rows'] and not weight_audit['read_errors']
    assert weight_audit['fallback_blocks'] == 0 and weight_audit['original_drift_blocks'] == 0
    tree = read(OUT / 'network/inferencex_tree.json')['sha']
    stree = read(OUT / 'network/sglang_tree.json')['sha']
    sources = {
        'golden': f'https://github.com/SemiAnalysisAI/InferenceX/blob/{tree}/inferencex-e2e/infx/golden_al_distribution/glm5.2_mtp.yaml',
        'golden_readme': f'https://github.com/SemiAnalysisAI/InferenceX/blob/{tree}/inferencex-e2e/infx/golden_al_distribution/README.md',
        'golden_collector': f'https://github.com/SemiAnalysisAI/InferenceX/blob/{tree}/inferencex-e2e/benchmarks/single_node/speedbench/glm52_fp4_b300_vllm.sh',
        'golden_source_run': 'https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28058352479',
        'speedbench': 'https://arxiv.org/html/2604.09557v2',
        'deepseek_v3': 'https://arxiv.org/html/2412.19437v2#S5.SS4.SSS3',
        'sglang_cookbook': f'https://github.com/sgl-project/sglang/blob/{stree}/docs/cookbook/autoregressive/GLM/GLM-5.2.mdx',
        'adaptive': 'https://github.com/sgl-project/sglang/pull/21599',
        'exact_sampling': 'https://arxiv.org/abs/2211.17192',
    }
    for name in ('inferencex_golden_pinned', 'inferencex_golden_readme_pinned', 'sglang_cookbook_pinned',
                 'inferencex_collector_pinned', 'inferencex_workflow_pinned', 'inferencex_golden_run'):
        meta = read(OUT / 'network' / (name + '.fetch.json'))
        assert hashlib.sha256((OUT / 'network' / (name + '.txt')).read_bytes()).hexdigest() == meta['sha256']
    golden_text = (OUT / 'network/inferencex_golden_pinned.txt').read_text()
    assert '3: 2.99' in golden_text and 'temperature: 1.0' in golden_text
    by_name = {r['name']: r for r in target['comparisons']}
    original_cycle = by_name['native_longinput_1k1024_0']
    prefix_cycle = by_name['cycle_fixed_prefix32_window128']
    releases = read(OUT / '节点释放核对.json')
    assert releases['owned_containers_all_stopped']
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines = ['# MTP结果是否符合一般情况：检索与补充诊断', '', f'更新时间：{stamp}。', '',
        '## 重新判断', '',
        '**正常完整聊天的AL约2.4～3.3，与公开MTP实测的量级相容；没有证据要求GLM-5.2在任意输入上都保持AL≈2。原生随机输入接近AL4，已有循环输出的直接证据，应视作这类负载的行为，不能当作正常业务普遍收益。**', '',
        '这句话只判断是否明显反常，不是全模型准确率验收。我们只有两条独立的完整ShareGPT问题，在三种配置下重复测量；不是六条独立问题，更不是生产分布。已有DP和不同前向路径的预测差异仍必须保留，不能用公开高AL来掩盖它。', '',
        '## 公开资料实际说了什么', '',
        '| 来源 | 公开结果或结论 | 能怎样与本轮比较 |', '|---|---|---|',
        f"| [InferenceX的GLM-5.2真实校准表]({sources['golden']}) | 1/2/3/4/5个草稿对应AL1.84/2.50/2.99/3.33/3.61 | 同一模型系列，3草稿AL2.99直接说明2不是通用上限；FP8、vLLM、coding、thinking开启、T=1/top_p=.95、输出上限4096，与我们的W8A8/Ascend/greedy/ShareGPT不同 |",
        f"| [校准方法说明]({sources['golden_readme']}) | 在SPEED-Bench Qualitative的coding类别实测，记录accepted/verify计数，再将golden曲线用于性能测试的合成接受 | 真实采集曲线与后续模拟吞吐测试必须区分，后者不是又一次自然接受率测量 |",
        f"| [SPEED-Bench论文]({sources['speedbench']}) | 随机token可能引发容易预测的简单回复，使接受率虚高；也可能让模型抓住噪声中的关键词另起话题，接受率反而偏低 | 直接支持随机输入不代表真实接受率，不能概括为所有random都会偏高。论文的逐位置AR是条件率，不能与我们的无条件P逐项直接比较 |",
        f"| [DeepSeek-V3论文§5.4.3]({sources['deepseek_v3']}) | 一个额外预测token接受率85%～90% | 是不同模型、一个额外token；说明80%～90%并非天生可疑，不等于我们的P2/P3应有相同数值 |",
        f"| [SGLang GLM-5.2 cookbook]({sources['sglang_cookbook']}) | 文档提醒MTP接受长度在部分负载很高，5草稿时可接近5～6；同页速度榜单说明固定了模拟AL | 高AL是已知现象，但文档描述及模拟榜单不能代替Ascend同条件实测。我们的3草稿上限接近4，不能照搬5～6 |",
        f"| [已合并的自适应推测解码PR]({sources['adaptive']}) | 将逐行重复列为高接受负载，将创作/长列表列为较低接受负载 | 与本次循环解释一致，但不证明每个随机输入都循环，也不证明本机所有算子正确 |", '',
        '公开资料保留在本目录network/，关键文件固定到检索时的Git commit，并保存URL、下载SHA和原文。未找到同时匹配本机Ascend/W8A8/DP2/3草稿及相同输入的公开接受率基线，因此这里只判断量级和机制是否相容，不作严格复现或等价声明。未将别的硬件/模型的未合并PR当成本机修复。', '',
        f"另外核对了[采集脚本]({sources['golden_collector']})：启动真实MTP，读取accepted/drafts累计计数差，按1+accepted/drafts计算AL；没有在该脚本的speculative-config中设置synthetic接受。[来源Actions运行]({sources['golden_source_run']})的公开元数据为completed/success。本轮未下载该运行的完整日志和artifact，也未在本机复测该公开曲线；2.99是公开仓库报告值。当前固定commit的脚本可能晚于来源运行，不能把它说成当次运行日志。", '',
        'SPEED-Bench §6还明确区分两种随机输入失真：Trivial Response（很容易猜的简单回复，接受率偏高）与Topic Latching（抓住随机关键词另起话题，接受率可能偏低）。本轮的精确循环属于容易预测的退化输出证据，不把它说成论文已确认的同一个实现bug。论文§8.4报告的平均23%吞吐高估来自GPT-OSS/EAGLE3/TensorRT-LLM实验，不能拿这个百分比来修正我们的GLM/Ascend数字。',
        '论文附录G还发现随机输入会改变MoE专家路由和覆盖；它提示我们的random-ids性能/Chunk结论只适用于所测负载。不能据外部GPT-OSS的路由比例推断本机GLM专家热力，也不影响此前真实数据集热力本身的测量记录。', '',
        '## 把我们的完整聊天重新算一遍', '',
        '预先选定的两个问题分别是营销方案总结、Java字符串模板替换。两者都使用模型原生聊天模板、enable_thinking=False、允许EOS、上限2048，实际自然结束。两条问题在三个服务配置下重复，不跨配置相加冒充更大的样本。', '',
        '| 配置 | 两请求输出/verify的加权AL | 按1+接受草稿/verify计算的AL | 无条件P1/P2/P3 | 条件AR1/AR2/AR3 |',
        '|---|---|---|---|---|']
    for row in existing['complete_chat_groups']:
        values = [' / '.join(f'{v:.4f}' for v in row[key]) for key in ('unconditional_positions', 'conditional_positions')]
        lines.append(f"| {row['run']} | {row['AL_output_over_verify']:.4f} | {row['AL_1_plus_accepted_over_verify']:.4f} | {values[0]} | {values[1]} |")
    windows = existing['eight_prompt_windows']
    lines += ['', '另有同一批8条独立ShareGPT问题的256-token窗口（包含上述完整回复的两条问题），可提供更广一点的短窗口观察，但它们不是完整回复：', '',
              '| 模式 | 请求数 | 加权AL | P1/P2/P3 | 达长度上限 | 检出精确周期 |', '|---|---|---|---|---|---|']
    for row in windows:
        positions = ' / '.join(f'{v:.4f}' for v in row['P1_P2_P3'])
        lines.append(f"| {row['group']} | {row['requests']} | {row['AL']:.4f} | {positions} | {row['length_finished']} | {row['cycle_detected']} |")
    lines += ['', '8条窗口两组都达256长度上限；没有自然结束，不能据两组的细小AL差值量化EOS效应，动态组批及数值差异也可能影响轨迹。',
        '营销问题单请求AL约2.40～2.50，Java问题约3.03～3.25，没有检出精确循环。代码的局部结构更容易预测是合理解释，但仅这两题不能证明领域因果或业务均值。',
        '两个AL算式的微小差异来自首token/EOS/末轮裁剪的计数边界，不能据此称指标失真。golden2.99来自另一种采样与任务，既不是精度阈值，也不是要强行调到的目标。', '',
        '用大白话说，3草稿配置的AL=2.8，大约表示每轮“1个主模型token+1.8个接受草稿”；对应草稿整体接受约60%，不是2.8/3≈93%。AL接近4才表示三个草稿几乎每轮都被接受。', '',
        '## 位置率的一个关键澄清', '',
        '本轮P2=“这一轮至少接受2个草稿的轮数/全部verify轮数”；SPEED-Bench的AR2=“第1个已接受的前提下，第2个接受的概率”。因此AR2=P2/P1，AR3=P3/P2。',
        '同一条链上：AL≈1+P1+P2+P3，也可写成1+AR1+AR1×AR2+AR1×AR2×AR3。不是把条件AR直接相加。', '',
        '| MTP.md提供的位置数字 | 若为本轮无条件P，对应AL | 若为条件AR，对应AL |', '|---|---|---|']
    for row in existing['two_possible_reference_conventions']:
        lines.append(f"| {row['given_positions']} | {row['AL_if_unconditional']:.6f} | {row['AL_if_conditional']:.6f} |")
    lines += ['', '文档同时写到“线上高点2.11”，这与第一组按无条件率求和相吻合，是支持这种解释的线索；仍应取得参考端的原始统计定义，不能仅凭“Per-position acceptance rate”这个名字断定。AL≈2也不能据这组三位置数字反推出同模型、同任务、同采样。', '',
        '另一个区别是贪心与随机采样：greedy通常比较草稿与主模型的argmax是否一致；标准随机推测采样比较两套概率分布并按拒绝采样规则接受。理论上单步接受取决于分布重叠，而非一个全模型固定常数。我们没有做同条件T=0/T=1实验，不能量化温度解释了多少差距。', '',
        '## 为什么补一次MTP关闭的连续生成', '',
        f"[标准推测采样]({sources['exact_sampling']})的目标是保持主模型分布。因此AL高低与输出质量是两个问题；如果主模型本来续写一个确定循环，正确的草稿也可以几乎全中。此前我们有teacher-forcing评分及MTP续写窗口，这次再做完全不启用草稿的连续生成，增强对循环来源的判断。", '',
        '仅23单机、公开源校验权重。相对最后no-multistream配置，只将MTP ON改为OFF；TP16/EP16/DP2、DP attention/LM head、normal、无Graph、mem.90、MRR8、chunk16384、CTX180000、plan0、多流0全部保持。OFF不加载草稿，实际KV和执行形状自然会变化，不能称所有内部状态相同。',
        '各请求使用保存的原始input_ids和采样参数，并固定到原请求实际dp_rank；每次清缓存，所有实际路由核对通过。两个sanity、两个原生256短输出、原循环输入1024输出、循环固定前缀32后的128-token窗口，共6次短诊断。没有重新启动正式MTP OFF性能网格，不比较速度收益。', '',
        '前5项是原输入的ON/OFF逐ID对照；第6项从同一循环轨迹截取前缀再续写，属于相关的额外检查，不是第6条独立提示，也没有重新测量该前缀的MTP接受率。', '',
        f"这次target-only的运行时权重审计有{weight_audit['records']}条，覆盖16个进程、182个分片，无校验错误、公共文件漂移或fallback；它证明字节读取来源，不证明算子精度。", '',
        '| 诊断 | OFF输出token | 与保存的MTP轨迹一致 | 首个差异位置 | OFF检出周期尾段 |', '|---|---|---|---|---|']
    for row in target['comparisons']:
        shape = row['output_shape']['periodic_suffix']
        shape_text = '无' if shape is None else f"周期{shape['period']}，占{shape['fraction']:.2%}"
        lines.append(f"| {row['name']} | {row['completion_tokens']} | {row['same_output_ids']} | {row['first_difference']} | {shape_text} |")
    lines += ['']
    if original_cycle['output_shape']['periodic_suffix']:
        lines.append('**MTP关闭后，主模型从原始随机输入开始仍生成精确周期。该循环不以启用MTP为必要条件。** ' +
            ('本条1024个输出ID还与MTP轨迹完全一致。' if original_cycle['same_output_ids'] else
             '本条输出并非逐ID完全一致，不能宣称两个执行路径等价；具体周期与首差异见表。'))
    else:
        lines.append('**本次主模型从原始随机输入独立生成没有复现原精确周期。** 此前teacher-forcing与续写支持只说明已进入的循环可自维持，不能据此声称MTP OFF也会从原始输入进入同一循环；需保留实现/执行路径对循环起点的影响。')
    lines.append('这里解释的是高接受率的直接机制，不是主模型陷入循环的全部根因。随机输入格式、量化和主模型执行实现仍可能影响循环；本次不能断言官方checkpoint在正确实现下必定产生同样循环，也不能排除主模型路径的问题。')
    lines.append('固定循环前缀后的128-token窗口：' +
        ('OFF逐ID完全维持原循环，证明该已进入状态在target-only续写下也能自维持。' if prefix_cycle['same_output_ids'] else
         'OFF没有逐ID保持原循环；不得将teacher-forcing评分直接当作连续生成等价。'))
    differences = [r for r in target['comparisons'] if not r['same_output_ids']]
    comparison_conclusion = (
        '这6条诊断中未检出MTP导致的输出偏离，增强了循环并非必须由MTP制造的证据；但有限样本一致不能证明所有请求或算子正确。'
        if not differences else
        '贪心不同值得进一步排查，但本轮同时移除了草稿加载和verify批量路径；量化、归约、路由及attention的数值差异可能参与，不能只凭首差异指定验证器错误。')
    lines += ['', f"6次中有{len(differences)}次与保存MTP输出不同。" + comparison_conclusion +
        '既有同前缀DP差异没有因这次检索而被解决。', '',
        '保存循环的bulk-prefill第32～127位置，主模型对实际输出token报告的概率中位数约99.7%，该段全部为argmax。这个值是主模型对已给定轨迹的评分，不是草稿置信度，也不是额外的接受率测量；主模型很确定本身不能保证草稿猜中，还需真实接受计数支持。', '',
        '## 现在值得做什么、不值得盲调什么', '',
        '1. 保留AboutModels四类固定长度random-ids作为系统负载口径；将其AL同时标注输出周期/EOS，尤其不能拿1K/32K循环当真实业务收益。',
        '2. 真实接受率参考应另列完整聊天或SPEED-Bench类别，固定checkpoint、模板、thinking、采样、草稿数及统计定义。下一轮若要报告业务均值，应运行更多独立提示；本轮两题只能说明存在这样的正常回复。',
        '3. 当前镜像NPU验证存在argmax分支，不能只改temperature就称已复现golden的随机采样；先验证采样实现。也不启用模拟、改接受阈值或挑种子来凑AL2。',
        '4. 下一项实现排查应针对已保存的固定前缀/实际DP预测差异，比较相同因果前缀上的target logits、路由和执行形状；网络上的router FP32/dispatch dtype修复只作线索。本次不叠加这些补丁，避免接受率变化后失去归因。',
        '5. 全模型精度、KVV≤3%及延迟门槛仍未验收；公开的合理接受率不能代替精度。高AL也不直接等于相同倍数的加速，还需草稿和验证成本的实际ON/OFF性能对照。', '',
        '## 数据入口', '',
        '- `network/`：搜索结果、论文、固定commit的golden曲线与说明、SGLang文档及PR原文。',
        '- [既有数据复算](existing_data_recheck.json)：完整聊天的条件/无条件率、两种AL算式、参考口径两种解释、循环评分。',
        '- [补测原始结果索引](target_only/summary.json)：6条请求的payload、逐ID输出、全文、压缩SSE、实际server-info、配置对照与summary。',
        '- `../logs/cyt-auto-review-target-only-20261013/`：启动命令、服务日志和只读权重加载审计。',
        f"- [节点释放核对](节点释放核对.json)：{releases['checked_at']}重新实查，自有容器全部停止、保留；8个NPU设备均无运行进程。",
        '- 原[全部MTP表](../../Q1MTPExp/MTP全部实验汇总_含自主探索.md)仍86条；此次OFF正确性诊断不混入MTP ON接受率表。', '']
    text = '\n'.join(lines)
    (OUT / 'MTP一般情况检索与补充诊断.md').write_text(text)
    (OUT / 'MTP一般情况检索与补充诊断.json').write_text(json.dumps({
        'updated': stamp, 'sources': sources, 'existing_data': existing, 'target_only': target,
        'node_release': releases, 'target_only_weight_audit': weight_audit,
        'conclusions': {'normal_chat_values_plausible_against_public_evidence': True,
            'target_only_original_cycle_detected': original_cycle['output_shape']['periodic_suffix'] is not None,
            'target_only_original_cycle_same_ids': original_cycle['same_output_ids'],
            'target_only_fixed_cycle_prefix_same_ids': prefix_cycle['same_output_ids'],
            'target_only_differing_cases': len(differences), 'full_accuracy_verified': False}
    }, ensure_ascii=False, indent=2))
    print('published review', len(target['comparisons']), 'target-only diagnostics')


if __name__ == '__main__':
    main()
