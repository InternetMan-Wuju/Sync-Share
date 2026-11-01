"""Readable report labels; raw experiment identifiers and data stay separate."""
import re


WORKLOADS = {
    'w1_8k_1k': 'W1：输入8192 → 输出1024',
    'w2_32k_1k': 'W2：输入32768 → 输出1024',
    'w3_1k_32k': 'W3：输入1024 → 目标输出32768',
    'w4_prefix32k_300': 'W4：总输入32768 → 输出300，前缀缓存60%',
    'native_exact_1k256': '原生随机ID直发：输入1024 → 输出256（短诊断）',
    'native_exact_eos_1k256': '原生随机ID直发：输入1024 → 最多输出256，允许提前结束',
    'native_longinput_1k1024': '长输出场景的短诊断：输入1024 → 输出1024',
    'native_default_text_roundtrip_256': '随机ID转文本再分词 → 输出256（实际输入长度会变）',
    'native_random_body_chatframed_256': '随机正文加聊天模板 → 最多输出256，允许提前结束',
    'framing_pair_raw_eos256': '同一随机正文直接续写 → 最多输出256，允许提前结束',
    'framing_pair_chat_eos256': '同一随机正文加聊天边界 → 最多输出256，允许提前结束',
    'sharegpt_chat_eos': '真实聊天短窗口：最多输出256，允许提前结束',
    'sharegpt_chat_forced': '真实聊天短窗口：强制输出256，忽略结束标记',
    'sharegpt_chat_complete': '真实聊天完整回复：最多输出2048，允许提前结束',
    'sanity_chat': '问候与算术功能检查',
    'cycle_fixed_prefix32_window128': '固定循环前32个token，再续写128个token',
    'native_exact_ids': '原生随机ID直发：输入1024 → 输出256（历史短诊断）',
    'uniform_legacy': '自写独立随机ID：输入1024 → 输出256（历史短诊断）',
    'native_default_text': '原生随机ID转文本再分词：输入长度会变 → 输出256',
    'random_1024': '自写独立随机ID：输入1024 → 输出1024',
    'random_16k256': '自写独立随机ID：输入16384 → 输出256',
    'random_32k256': '自写独立随机ID：计划输入32768 → 输出256（无有效测量）',
    'real_1024': '截取真实聊天正文，无完整聊天模板：输入1024 → 输出1024',
    'real_16k256': '截取真实聊天正文，无完整聊天模板：输入16384 → 输出256',
}

SERVICES = {
    'publisher_v19': '0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图',
    'mtp-publisher-v19': '0.5.19版，公开源恢复权重，单路分流、常规通信、关闭计算图',
    'mtp-publisher-v19-slim': '0.5.19版，公开源恢复权重，草稿分片视图加载',
    'mtp-restored-v19': '0.5.19版，初次恢复权重方案',
    'mtp-v16-auto-cap64-mem87-hccl650': '旧镜像，双路分流、自动通信，静态显存87%方案',
    'mtp-v16-auto-cap64-mem90-hccl650': '旧镜像，双路分流、自动通信，静态显存90%方案',
    'mtp-v16-auto-cap64-mem84': '旧镜像，通信容量64，静态显存84%、通信缓冲1536MB',
    'mtp-v16-auto-cap64': '旧镜像，通信容量64，静态显存90%、通信缓冲1536MB',
    'mtp-default-v16': '旧镜像默认方案：双路分流、自动通信、开启计算图',
    'mtp-default-v16-graph-off': '旧镜像默认方案：关闭计算图',
    'mtp-default-v16-hccl1000': '旧镜像默认方案：通信缓冲增至1000MB',
    'mtp-v19-dp2-normal': '0.5.19版，双路分流、常规通信，原始断言诊断',
    'mtp-v19-dp2-ordered-probe': '0.5.19版，双路分流，调整断言检查时机',
    'mtp-v19-dp2-no-plan': '0.5.19版，双路分流，关闭调度重叠（评分回退前）',
    'mtp-v19-dp2-no-plan-lp-safe': '0.5.19版，双路分流，关闭调度重叠、常规评分',
    'mtp-v19-dp2-no-multistream': '0.5.19版，双路分流，关闭调度重叠及NPU多流、常规评分',
    'chunk_auto_sanity': '分块实验服务的功能检查',
    'chunk-auto-16384': '每块16384，自动通信，初始计算图批次设置',
    'chunk-normal-16384': '每块16384，常规通信，首次备选启动',
    'chunk-auto-aligned-16384': '每块16384，自动通信，修正计算图批次、通信缓冲1536MB',
    'chunk-normal-aligned-16384': '每块16384，常规通信，第二次备选启动',
    'chunk-auto-hccl650-16384': '每块16384，自动通信，修正计算图批次、通信缓冲650MB',
    'chunk-auto-1024': '每块1024，自动通信、通信缓冲650MB',
    'chunk-auto-4096': '每块4096，自动通信、通信缓冲650MB',
    'chunk-auto-8192': '每块8192，自动通信、通信缓冲650MB',
    'chunk-repeat-16384': '每块16384，基线复测',
    'chunk-repeat-4096': '每块4096，候选复测',
    'chunk-repeat-8192': '每块8192，候选复测',
    'no-plan-lp-safe': '关闭调度重叠、使用常规评分',
    'no-multistream': '关闭NPU多流',
    'native_exact': '原生随机ID直发',
}

LABELS = {**WORKLOADS, **SERVICES}
for _name, _label in list(SERVICES.items()):
    LABELS[_name + '_business'] = _label + '（真实聊天测试）'
for _name in ('sanity_chat', 'native_exact_1k256', 'native_longinput_1k1024'):
    for _index in (0, 1):
        LABELS[f'{_name}_{_index}'] = WORKLOADS[_name] + f'，第{_index + 1}条请求'
for _name in ('w1_8k_1k', 'w2_32k_1k', 'w4_prefix32k_300'):
    for _chunk in (1024, 4096, 8192, 16384):
        for _concurrency in (1, 2, 4):
            _key = f'{_name}_chunk{_chunk}_c{_concurrency}'
            _label = f'{WORKLOADS[_name]}；每块{_chunk}；并发{_concurrency}'
            LABELS[_key] = _label + '；第1轮'
            LABELS[_key + '_repeat2'] = _label + '；第2轮复测'

_LABEL_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_/-])(?:' + '|'.join(
        re.escape(k) for k in sorted(LABELS, key=len, reverse=True)
    ) + r')(?![A-Za-z0-9_-])'
)
_SHAPES = re.compile(
    r'(?<![A-Za-z0-9_/])(?P<input>32K|16K|8K|1K)\s*/\s*'
    r'(?P<output>32K|1K|2048|1024|300|256|32)(?![A-Za-z0-9_/])'
)


def workload_label(name):
    return WORKLOADS.get(name, name)


def service_label(name):
    return LABELS.get(name, name)


def combined_row_label(row):
    """Use saved lengths for historical variants instead of trusting their name."""
    name = row['label']
    if name == 'random':
        return '自写独立随机ID：输入1024 → 输出256'
    if name == 'real':
        return '截取真实聊天正文，无完整聊天模板：输入1024 → 输出256'
    if name == 'random：变长，上限8K/256':
        return '原生随机文本：输入/输出变长，上限8192/256'
    if name == 'random-ids：变长，上限8K/256':
        return '原生随机ID：输入/输出变长，上限8192/256'
    if name in ('C', 'D', 'E', 'F', 'G'):
        config = {
            'C': '旧镜像、静态显存92%、声明关图',
            'D': '旧镜像、静态显存88%、声明关图',
            'E': '旧镜像、静态显存92%、声明开图但未生效',
            'F': '0.5.19版、静态显存92%、声明开图但未生效',
            'G': '0.5.20版、静态显存92%、声明开图但未生效',
        }[name]
        return f'真实短问询：输入34 → 输出32（配置{name}：{config}）'
    if name == 'w3_1k_32k' and row.get('output_target') != 32768:
        return f"W3的缩短版：输入1024 → 实际目标输出{row.get('output_target')}（非完整32768）"
    return WORKLOADS.get(name, name)


def combined_lengths(row):
    lengths = row.get('input_lengths')
    inputs = '、'.join(str(v) for v in lengths) if lengths else '未记录'
    target = row.get('output_target')
    return f"输入{inputs}；目标输出{target if target is not None else '未记录'}"


def combined_counts(row):
    concurrency = row.get('concurrency')
    requests = row.get('requests_measured')
    return (f"并发{concurrency if concurrency is not None else '未记录'}；"
            f"共{requests if requests is not None else '未记录'}请求")


def _readable_text(text):
    text = _LABEL_PATTERN.sub(lambda m: LABELS[m[0]], text)
    def shape(match):
        def tokens(value):
            return str(int(value[:-1]) * 1024) if value.endswith('K') else value
        return f"输入{tokens(match['input'])} → 输出{tokens(match['output'])}"
    text = _SHAPES.sub(shape, text)
    text = re.sub(r'(?<![A-Za-z0-9_])C(1|2|4|32)(?![A-Za-z0-9_])', r'并发\1', text)
    text = re.sub(r'\bMTP\s+ON\b', 'MTP开启', text)
    text = re.sub(r'\bMTP\s+OFF\b', 'MTP关闭', text)
    text = re.sub(r'\bChunk(1024|4096|8192|16384)\b', r'每块\1个token', text)
    return text


def humanize_report(markdown):
    """Translate visible labels while preserving links, paths, code and markers."""
    protected = []
    def keep(match):
        protected.append(match[0])
        return f'\x00{len(protected) - 1}\x00'
    # Protect executable examples and automation markers before prose rewriting.
    markdown = re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*(?:\n|$)', keep,
                      markdown, flags=re.MULTILINE | re.DOTALL)
    markdown = re.sub(r'<!--.*?-->', keep, markdown, flags=re.DOTALL)
    # Target text is opaque: changing an experiment identifier would break links.
    markdown = re.sub(r'(?<=\]\()[^\n)]*(?=\))', keep, markdown)
    def inline(match):
        name = match[0][1:-1]
        return LABELS[name] if name in LABELS else keep(match)
    markdown = re.sub(r'`[^`\n]+`', inline, markdown)
    markdown = _readable_text(markdown)
    return re.sub(r'\x00(\d+)\x00', lambda m: protected[int(m[1])], markdown)
