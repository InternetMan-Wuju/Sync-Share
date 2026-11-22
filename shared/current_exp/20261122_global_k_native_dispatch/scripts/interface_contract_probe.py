#!/usr/bin/env python3
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
audit=json.loads((root/'artifacts/INTERFACE_AUDIT.json').read_text())
assert 256 + 1 == 257
assert 257 % 2 == 1
assert audit['five_contract_checks']['nonuniform_extra_capacity']['verdict']=='CURRENT_NOT_SUPPORTED'
assert audit['five_contract_checks']['all_rank_protocol']['verdict']=='CURRENT_NOT_SUPPORTED_FOR_UNEVEN_LAYOUT'
print(json.dumps({
 'logical_experts':256,
 'global_extra_slots':1,
 'physical_instances':257,
 'ep_size':2,
 'divisible_by_ep':False,
 'native_k1_constructible':False,
 'decisive_blocker':audit['decisive_blocker']['short'],
},indent=2))
