#!/usr/bin/env python3
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; art=root/'artifacts'
audit=json.loads((art/'INTERFACE_AUDIT.json').read_text())
life=json.loads((art/'lifecycle_summary.json').read_text())
device=json.loads((art/'DEVICE_TIMELINE_SUMMARY.json').read_text())
summary={
 'one_line_answers':{
  'native_k1_can_attach':'No. The installed uniform per-rank expert contract cannot represent 256 logical experts plus one global extra physical instance (257 total).',
  'resident_stably_break_even':'No. At 8192/2048 resident medians were +0.0111 ms in one round and -0.0098 ms in the other, with 9/12 then 4/12 positive pairs; 4096/1024 was negative in both rounds.',
  'largest_certain_blocker':'FusedMoE requires num_global_routed % storage_ep_size == 0 and gives every rank the same local expert count; the native runner also reads layer-owned weights rather than an external K=1 slot handle.'
 },
 'interface_audit':audit,
 'lifecycle_summary':life,
 'device_timeline':{
  'path':'artifacts/DEVICE_TIMELINE_SUMMARY.json',
  'scope':device['scope'],
  'clock_alignment':device['clock_alignment'],
  'ranks':device['ranks'],
  'raw_trace_root':'artifacts/profiler_raw/',
  'raw_trace_size_bytes_approx':4700000,
  'note':'Profiler overhead inflates absolute kernel and wait durations. Trace evidence is used only to identify operator classes and extra communication/wait structure, not for formal timing.'
 },
 'changes_verified_this_round':[
  'Old lifecycle and continuously resident lifecycle are separated.',
  'Resident slot stays ready across a block; each split call enters/exits a use reference and evicts only at block end.',
  'Complete formal path including HCCL send/recv, refill and ack is warmed outside samples.',
  'Native interface audit records the exact uniform-sharding blocker.'
 ],
 'not_measured':[
  'full MoE layer correctness', 'shared expert', 'all-expert combine', 'native 257-instance dispatch/combine',
  'K=2', 'Decode', 'cross-node', 'multi-request concurrency', 'full-model TTFT'
 ]
}
(art/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['one_line_answers'],indent=2))
