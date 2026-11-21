#!/usr/bin/env python3
import unittest, threading
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from global_slot_pool import GlobalSlotPool, SlotPoolExhausted
from toy_global_search import Slot, Action, apply_action, search

class GlobalKTests(unittest.TestCase):
    def test_k_not_multiple_and_no_dummy_alloc(self):
        calls=[]
        def alloc(c,b): calls.append((c,b)); return [bytearray(b) for _ in range(c)]
        p=GlobalSlotPool([0,2,5,9,13], 64, alloc, name='k5')
        s=p.stats()
        self.assertEqual(s['K'],5)
        self.assertEqual(s['committed_bytes'],320)
        self.assertEqual(calls, [(1,64)]*5)
        self.assertEqual(p.per_rank_capacity(), {0:1,2:1,5:1,9:1,13:1})
        leases=[p.acquire(request_id=f'r{i}',layer_id=i,expert_id=i%3) for i in range(5)]
        self.assertEqual({x.owner_rank for x in leases}, {0,2,5,9,13})
        with self.assertRaises(SlotPoolExhausted):
            p.acquire(request_id='extra', layer_id=0, expert_id=0)
        for x in leases:
            p.mark_ready(x)
            with p.use(x):
                pass
            p.release(x)
        self.assertEqual(p.stats()['cached_free_bytes'],320)
        self.assertEqual(p.stats()['active_bytes'],0)

    def test_cross_layer_reuse(self):
        p=GlobalSlotPool([7,7], 16, lambda c,b:[bytearray(b) for _ in range(c)])
        for layer in range(20):
            x=p.acquire(request_id=f'l{layer}',layer_id=layer,expert_id=layer)
            p.mark_ready(x)
            with p.use(x) as b: b[0]=layer
            p.release(x)
        s=p.stats(); self.assertEqual(s['committed_bytes'],32); self.assertEqual(s['active_bytes'],0)
        self.assertGreater(sum(v.reuses for v in s['local'].values()),0)

    def test_counterexample_requires_multi_action(self):
        loads=(40,39,1,1)
        slots=(Slot(2),Slot(3))
        best=search(loads,slots,max_actions=2,exhaustive=True,copy_ms=2)
        self.assertEqual(best.max_load,21)
        self.assertEqual(len(best.actions),2)
        # Explicitly verify the requested counterexample transition.
        slots2=(Slot(2),Slot(3))
        x=apply_action((40,39,1,1),slots2,set(),Action(0,2,20,0,True))
        x=apply_action(x,slots2,{0},Action(1,3,19,1,True))
        self.assertEqual(x,(20,20,21,20))
        # A greedy one-action score is worse than the start in this toy.
        one=search(loads,slots,max_actions=1,exhaustive=True,copy_ms=2)
        self.assertGreaterEqual(one.max_load+2*one.copies,40)

    def test_beam_finds_pair(self):
        loads=(40,39,1,1); slots=(Slot(2),Slot(3))
        best=search(loads,slots,max_actions=2,beam_width=32,exhaustive=False,copy_ms=2)
        self.assertEqual(best.max_load,21)
        self.assertEqual(len(best.actions),2)
        # Explicitly verify the requested counterexample transition.
        slots2=(Slot(2),Slot(3))
        x=apply_action((40,39,1,1),slots2,set(),Action(0,2,20,0,True))
        x=apply_action(x,slots2,{0},Action(1,3,19,1,True))
        self.assertEqual(x,(20,20,21,20))

    def test_existing_replica_no_copy(self):
        loads=(40,1,1); slots=(Slot(2,expert=5,ready=True),)
        best=search(loads,slots,max_actions=1,exhaustive=True,copy_ms=2)
        self.assertEqual(best.copies,0)
        self.assertEqual(best.actions[0].copy,False)

if __name__=='__main__': unittest.main(verbosity=2)
