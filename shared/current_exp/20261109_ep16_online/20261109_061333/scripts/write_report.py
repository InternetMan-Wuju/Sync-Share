"""Generate report numbers from SUMMARY and VALIDATION, not hand-entered values."""
import json,hashlib,statistics,gzip
from pathlib import Path
P=Path(__file__).resolve().parents[1]
s=json.loads((P/'SUMMARY.json').read_text());v=json.loads((P/'VALIDATION.json').read_text())
rows=[json.loads(x) for x in gzip.decompress((P/'artifacts/rank_records.jsonl.gz').read_bytes()).decode().splitlines()]
def med(mode,key):return s['profile_rank_max'][mode][key]['median']
def table(headers, lines):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,line))+' |\n' for line in lines)
plan_rows=[]; replay_rows=[]; ttft_rows=[]; paired_rows=[]
for c in range(3):
    p=v['hardware_cases'][c]['plan'];plan_rows.append([c,f"e{p['expert']}: rank{p['src']} → rank{p['dst']}",p['q'],max(p['old_loads']),max(p['new_loads']),f"{100*p['saving_tasks']/max(p['old_loads']):.2f}%"])
    d={m:s['layer_replay'][str(c)][m]['max_rank_block_mean_ms']['median'] for m in ('off','prepared','warm','cold')}
    replay_rows.append([c,*[f'{d[m]:.3f}' for m in ('off','prepared','warm','cold')],f"+{d['off']-d['warm']:.3f} ({100*(d['off']-d['warm'])/d['off']:.2f}%)",f"{d['off']-d['cold']:.3f} ({100*(d['off']-d['cold'])/d['off']:.2f}%)"])
    ttft_rows.append([c,*[f"{s['e2e'][str(c)][m]['client_ttft_ms']['median']:.3f}" for m in ('off','shadow','active_cold','active_warm')]])
    paired_rows.append([c,*[f"{-s['paired_ttft_saving'][str(c)][m]['saving_ms']['median']:+.3f}" for m in ('active_cold','active_warm')]])
profile=[]
for label,key in [('dispatch事件窗口','dispatch_device_ms'),('本地量化专家MLP事件窗口','compute_device_ms'),('combine事件窗口（含等其他rank）','combine_device_ms'),('计数计算 + HCCL交换 + D2H','count_exchange_d2h_ms'),('CPU planner（host计数已就绪）','planner_ms'),('配额映射提交','remap_ms'),('完整routed子链诊断wall time','target_experts_wall_ms')]:
    profile.append([label,*[f"{med(m,key):.3f}" if key in s['profile_rank_max'][m] else '—' for m in ('off','active_warm','active_cold')]])
for label,number in [('源端NZ→ND及打包',med('active_cold','source_prepare_ms')),('发送端send + 完成同步',s['copy_role_times']['src']['transfer_ms']['median']),('接收端recv窗口（包含等源端打包）',s['copy_role_times']['dst']['transfer_ms']['median']),('目的bank格式安装',med('active_cold','destination_install_ms')),('其他rank最长ready等待',med('active_cold','ready_wait_ms')),('状态准备到全组ready完整跨度',med('active_cold','state_prepare_through_ready_ms'))]:
    profile.append([label,'—','0',f'{number:.3f}'])
blocks=[r for r in rows if r['kind']=='timing_block']
first=[r for r in rows if r.get('request')=='validate_cold_0']
first_p2p=max(r['timing']['transfer_ms'] for r in first)
peak=max(r.get('hbm_peak_bytes',0) for r in rows)
text=f'''# 23节点16设备：8K Prefill + 32 token整机实验

Run：`20261109_061333`。本轮全部性能数字为**本机16个独立NPU设备并行实测**，不引用旧单卡max代理；无跨节点实验。

## 结论

**功能链已在16设备跑通，计算重分布有收益；当前同步冷复制没有回本，也没有证明整机TTFT加速。**

- 三个输入的已准备布局，使routed-expert完整子链耗时降低约24%。
- 副本已经就绪时，包含在线统计、CPU规划和配额映射，子链净快约3.9%–7.8%。这不包含此前准备该副本的成本。
- 每次真正冷复制时，完整子链慢约27.3%–30.8%。不能把暖缓存的正收益写成首次在线复制的正收益。
- 60个完整8192→32请求中，off首token约2.55秒；冷复制15个配对全部更慢，暖副本也没有稳定的TTFT改善。
- 同hidden/top-k的16rank独立输出快照比较：三输入及半配额验证均`max_abs=0, NRMSE=0`。**整段生成一致性未通过**：off自身重复请求也不确定，不能宣称生产数值/服务验收完成。

## 实际输入、布局与动作

输入为AgentX公开trace的真实长度/块身份驱动的**合成token物化**，并非原始AgentX文本。本地Arrow只有长度及hash块，没有用户原文；固定种子按128个块×64token生成8192个ID。输入身份、数据集revision与SHA256见[INPUTS.json](INPUTS.json)。路由由真实完整GLM模型计算，未人工指定热点直方图。

模型为`GlmMoeDsaForCausalLM`，ModelSlim W8A8，TP16/EP16/DP1，DeepEP auto，单节点设备0–15。npu-smi显示8个双芯模块、16个设备；双芯内SIO，其他设备HCCS_SW。仅layer_id=49启用实验策略；prefill/decode graph、overlap scheduler及prefix缓存关闭。每个完整请求temperature=0、ignore_eos、恰好生成32token，cached_tokens=0。

256逻辑专家，272物理槽，每rank16个主槽+1个备用槽；主映射`17*(e//16)+e%16`。每rank收到512个唯一输入token，top8，共65536个有效分支。独立验证输入行数及分支数，没有用总数直接除TP16；off的所有备用槽任务数为零。

{table(['输入','本轮计划','迁出任务','原最大rank任务','新最大rank任务','最大任务下降'],plan_rows)}
本轮启发式选择将e43的本次任务全部迁往备用槽；源rank继续计算其余专家。例如case0的rank2由20003降至11841，rank0由1281升至9443。每个rank的完整17槽向量、全局计数和多来源配额保存在[INPUTS_PLANS.json](INPUTS_PLANS.json)。

另做一次**不纳入性能统计**的半配额验证：e43的8162个任务，4081留在源rank、4081迁往目的rank。16来源配额为`[505,511,512,510,510,511,511,509,2,0,0,0,0,0,0,0]`，跨来源边界精确守恒，输出误差仍为零。

## 16设备同时执行的子链结果

下表单位ms，各格为5个正式块的中位数，每块先取16个实际并行rank的最大wall time再除调用数。收益为表中中位数之差；逐块配对差与范围见[SUMMARY.json](SUMMARY.json)。没有把16rank顺序模拟成并行。

{table(['输入','off原布局','prepared新布局','warm在线已就绪','cold每次复制','warm净节省','cold净节省'],replay_rows)}
`prepared`只测新布局的原生dispatch→完整量化MLP→combine，不含在线统计/规划/复制；`warm`每次重做当前计数、规划和整数配额映射，但已有匹配的副本；`cold`每次还执行真实独立buffer的状态传输、格式安装与全组ready。

三输入的15个配对块中，warm均比off快，cold均比off慢。每模式先20次预热，校准后每块实际11/14/15/18次调用；960条rank级正式记录对应60个并行组块，实际块长{min(r['block_ms'] for r in blocks):.3f}–{max(r['block_ms'] for r in blocks):.3f} ms。**块内调用不是独立重复，五块范围不是生产p95。**

计时边界是`DeepseekV2MoE.experts.forward`，包含routed专家完整通信和计算；不包含gate、shared expert、attention、层间通信及最后的shared/routed相加。重复相同真实层输入也会改变缓存和外围竞争条件，因此不能把子链收益乘75，或据此推导整机收益。

## Breakdown：时间花在哪里

case0，下面是**另外5个诊断请求**的分项中位数，单位ms。事件打点和同步会影响单次调用；主性能结论以上述无分项打点的正式块为准。分项并未各自做≥100ms块标定，只用于定位成本。

{table(['分项','off','warm','cold'],profile)}
**不能把这张表逐项相加。** 事件窗口的最大值可能来自不同rank；冷rank先进入combine等待忙rank，所以combine窗口包含其他rank的计算等待。recv窗口包含等待源端打包，其他rank的ready等待又覆盖整个复制过程。状态准备总跨度是先在同一rank相加顺序阶段、再取16rank最大值，避免重复计算等待。

由此可见：本地专家MLP最大事件窗口约4.484→2.913 ms（约35%下降）；但在线计数交换/D2H约1.2–1.3 ms、规划约0.08 ms、映射约0.26 ms已吃掉大部分节省。冷模式再付约3.79 ms状态准备及ready，超过约2.3 ms的prepared子链节省空间。

复制的是实际runner使用的完整6个运行态张量：W13、W2、各自BF16 scale与FP32 offset，共**37,810,176 B**。源权重是NZ格式29，传输用ND格式2，实际安装到目的slot16后核对全部字节。源端格式转换/打包约2.27 ms，比稳态发送端send及完成同步的约0.69 ms更突出。

第一次使用该P2P peer时，验证请求中的send/recv窗口最长{first_p2p:.2f} ms，含首次通信资源初始化；它与已预热通信但每次仍复制权重的cold分开保存。正确性请求中的SHA/输出比较耗时不计入任何正式性能数据。

## 完整模型结果

3输入×4模式×5交错重复=60个正式完整请求。下表为客户端流式TTFT中位数，单位ms；每次都执行8192token prefill及32token decode。

{table(['输入','off','shadow','cold','warm'],ttft_rows)}
按同输入、同重复块配对，`模式TTFT−off TTFT`的中位数如下，正数表示更慢。这与上表“两列中位数直接相减”是不同统计量。

{table(['输入','cold相对off / ms','warm相对off / ms'],paired_rows)}
冷复制15对全部更慢；暖副本的配对差有正有负，三个输入的配对中位数均更慢，**未观察到整机首token加速**。这不是已部署生产服务的吞吐/SLO评估：只改一层、eager、batch1、5次重复，局部不足1ms的收益不能外推。

原模式完整请求中位数约12.37–12.67秒，客户端平均decode TPOT的中位数约317–326ms。decode不执行新策略；完整32token输出在off的5次重复中分别出现5/5/4种序列，问题发生在基线自身，原因本轮未定位。保存全部输出ID和计时，**不把不同decode序列带来的E2E/TPOT波动归因于负载均衡**。测试期间有3次只读server-info查询，可能带来控制面扰动，未据此筛除样本。

## 正确性、显存与实现边界

- CPU逻辑6项测试通过：未受影响rank下界、独立整数枚举、多来源配额、无效ID/备用槽排除与逐分支身份保持。
- 三输入×16rank独立clone参考、真实全状态复制、正确slot安装、整数配额和原权重combine均通过；另有16rank半配额验证。详见[VALIDATION.json](VALIDATION.json)。这验证对原生链的等价性，不代替独立FP参考对原量化模型的精度认证。
- 原SGLang安装和模型文件未修改；仅外挂wrapper，默认off。active是显式`test_only`的强制标定入口，当前计数planner**尚未实现经过校准的生产净收益门控**，未知成本不能据此自动投产。
- 75个MoE层均分配每rank一个备用槽，虽然只操作layer49。按运行态字节推算额外专家状态约2.836 GB/rank（2.641 GiB），不是实测相对无槽服务的HBM差值；每rank另有37.81 MB打包buffer。不能声称解决了跨层备用状态的内存占用问题。
- KV预算固定12288token，日志分配约1.27 GB/rank；模型日志约50.67 GB/rank，PyTorch实际峰值最大{peak/2**30:.3f} GiB。未对比无备用槽服务的最大并发容量。各对照均有同样的权重驻留、备用槽和固定KV预算。
- 启动曾因PYTHONPATH遗漏CANN的tbe路径失败，已修复并加真实NPU预检。未隐去失败，见[开发记录](artifacts/DEVELOPMENT_NOTES.md)及[失败证据](artifacts/startup_failure_tbe.txt)。
- 正式性能全部执行revision2，源码快照[measured_experiment_v2.py](artifacts/measured_experiment_v2.py)的SHA与16worker日志一致；revision3仅增加显式半配额验证开关，未用于性能采样。完整原日志仍在本地`local/`，权重、hidden和大型dump没有上传。

## 可以交给下一轮的具体问题

当前已证明“16设备下重新分布可以减少计算”，尚未证明“每次prefill同步冷复制能赚钱”。优先处理有实测来源的两个成本：约1.2ms的统计交换/D2H，以及约2.27ms的源端NZ→ND/打包；保留全状态正确性、实际slot安装和ready保证。随后才用完整子链的实测总时间标定copy-aware门控。不能用单项中位数相减代替完整调用，更不能用本轮暖副本结果授权每次冷复制。

本轮不再启动额外采样，不做跨机实验。自己的模型容器已停止，16设备释放证据见[npu_released.txt](artifacts/npu_released.txt)。

## 审阅与复现入口

- [MANIFEST.json](MANIFEST.json)、[ENVIRONMENT.json](ENVIRONMENT.json)：环境、版本、输入、运行状态与源码摘要。
- [TIMINGS.csv](TIMINGS.csv)：全部正式块、诊断窗口与完整请求原始计时；[SUMMARY.json](SUMMARY.json)为机械汇总。
- [压缩rank原始记录](artifacts/rank_records.jsonl.gz)、[INPUTS_PLANS.json](INPUTS_PLANS.json)：完整槽向量、每来源计数、配额和校验结果。
- [IMPLEMENTATION.md](IMPLEMENTATION.md)、[启动脚本](scripts/launch.sh)、[客户端](scripts/client.py)、[固定A/B序列](scripts/suite.py)、[汇总脚本](scripts/analyze.py)。
- 严格复现本轮性能代码可先将保存的`artifacts/measured_experiment_v2.py`复制为这个run副本下的`plugin/experiment.py`再启动隔离容器；不改系统安装。输入生成依赖同revision的本地Arrow，见[scripts/prepare_inputs.py](scripts/prepare_inputs.py)。
'''
(P/'RESULTS.md').write_text(text)
print('RESULTS.md generated')
