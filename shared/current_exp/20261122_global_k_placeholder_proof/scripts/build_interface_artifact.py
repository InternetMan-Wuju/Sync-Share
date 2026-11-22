#!/usr/bin/env python3
import json
from pathlib import Path
art=Path(__file__).resolve().parents[1]/'artifacts'
def load(n): return json.loads((art/n).read_text())
i256=[load(f'interface_run_256_rank{r}.json') for r in (0,1)]
i257=[load(f'interface_run_257_rank{r}.json') for r in (0,1)]
i258=[load(f'interface_run_258_rank{r}.json') for r in (0,1)]
zero=[load(f'zero_rank_258_rank{r}.json') for r in (0,1)]
out={
 'backend':'SGLang DeepEPDispatcher NORMAL with DEEP_USE_MODE=alltoall',
 'default_normal_failure':{'evidence':'artifacts/default_normal_256_failure.log','error':'aclnnDispatchLayout not present in libopapi.so'},
 'backend_note':'Native torch_npu npu_moe_init_routing_v2/npu_moe_finalize_routing plus torch.distributed all_to_all_single.',
 'profile_256':i256,'profile_257':i257,'profile_258':i258,'zero_rank_with_empty_padding':zero,
 'checks':{
  '256_native_positive':all(x.get('constructor')=='PASS' and x.get('correct') for x in i256),
  '257_actual_failure':all(x.get('constructor')=='PASS' and x.get('failure_type') and 'shape' in x.get('failure_message','') for x in i257),
  '258_native_protocol':all(x.get('constructor')=='PASS' and x.get('correct') for x in i258),
  'zero_rank_participant':all(x.get('correct') for x in zero) and zero[1].get('legal_tokens_sent')==0 and zero[1].get('legal_recv_tasks_rank')==0 and zero[0].get('empty_padding_rows_received')==8,
 }
}
(art/'INTERFACE_RUNS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['checks'],indent=2))
if not all(out['checks'].values()): raise SystemExit(1)
