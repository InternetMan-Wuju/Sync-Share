#!/usr/bin/env python3
"""Call the target image's DeepSeek EPLB algorithm on captured layer49 counts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from vendor_deepseek_eplb import rebalance_experts

ap=argparse.ArgumentParser()
ap.add_argument('--routes',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--summary',type=Path,required=True)
ap.add_argument('--ep',type=int,required=True)
ap.add_argument('--redundant',type=int,required=True)
ap.add_argument('--layer',type=int,default=49)
ap.add_argument('--num-layers',type=int,default=78)
a=ap.parse_args()
rows=[json.loads(x) for x in a.routes.read_text().splitlines() if x.strip()]
assert rows, 'no observed routes'
counts=torch.tensor([r['logical_counts'] for r in rows],dtype=torch.int64).sum(0)
assert counts.shape==(256,)
weight=torch.zeros((a.num_layers,256),dtype=torch.float32)
weight[a.layer]=counts.float()
num_physical=256+a.redundant
phy2log,log2phy,logcnt=rebalance_experts(weight=weight,num_replicas=num_physical,num_groups=1,num_nodes=1,num_gpus=a.ep,enable_hierarchical=False)
assert phy2log.shape==(a.num_layers,num_physical)
assert num_physical%a.ep==0
local=num_physical//a.ep
layer_map=phy2log[a.layer].tolist()
for rank in range(a.ep):
    part=layer_map[rank*local:(rank+1)*local]
    assert len(part)==local
rank_rows=[]
for rank in range(a.ep):
    part=layer_map[rank*local:(rank+1)*local]
    rank_rows.append({'rank':rank,'local_slots':local,'physical_to_logical':part,'replica_slots':[{'slot':i,'logical':e} for i,e in enumerate(part) if int(logcnt[a.layer,e].item())>1]})
summary={'ep':a.ep,'redundant':a.redundant,'num_physical':num_physical,'local_slots':local,'layer':a.layer,'observed_sequences':len(rows),'observed_task_count':int(counts.sum()),'logical_counts':counts.tolist(),'top_hot':[{'logical':int(i),'tasks':int(counts[i])} for i in torch.topk(counts,16).indices],'rank_layouts':rank_rows,'replica_count_per_logical':logcnt[a.layer].tolist(),'algorithm':'target-image sglang.srt.eplb.eplb_algorithms.deepseek.rebalance_experts','algorithm_sha256':hashlib.sha256(Path('vendor_deepseek_eplb.py').read_bytes()).hexdigest()}
a.out.parent.mkdir(parents=True,exist_ok=True);a.summary.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'physical_to_logical_map':phy2log.tolist()})+'\n')
a.summary.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('logical_counts','replica_count_per_logical','rank_layouts')},indent=2))
