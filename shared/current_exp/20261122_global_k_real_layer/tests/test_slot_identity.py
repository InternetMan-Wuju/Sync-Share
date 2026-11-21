#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from global_slot_pool import GlobalSlotPool

def alloc(c,b): return [bytearray(b) for _ in range(c)]

calls=[]
a=GlobalSlotPool([1],8,alloc,name='a',synchronize=lambda:calls.append('sync'))
b=GlobalSlotPool([1],8,alloc,name='b',synchronize=lambda:calls.append('other'))
lease=a.acquire(request_id='r',layer_id=1,expert_id=1)
try:
    b.mark_ready(lease)
    raise AssertionError('foreign lease accepted')
except Exception:
    pass
a.abort(lease)
assert a.stats()['active_bytes']==0
lease=a.acquire(request_id='r2',layer_id=1,expert_id=1)
a.mark_ready(lease)
a.release(lease)
assert calls==['sync','sync'],calls
print('SLOT_IDENTITY_PASS',calls)
