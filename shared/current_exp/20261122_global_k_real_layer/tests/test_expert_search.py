import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from expert_search import Slot,State,search,candidates,apply_action

def test_identity_and_reuse():
    counts=((40,0,0,0),(39,0,0,0),(1,0,0,0),(1,0,0,0))
    s0=State(counts,(Slot(2,expert=7,layer=0,ready=True),),(0,))
    a=next(x for x in candidates(s0) if x.copy)
    assert apply_action(s0,a) is not None
    s=search(counts,[Slot(2,expert=0,layer=0,ready=True)],max_actions=1,beam_width=16)
    assert s.actions and s.actions[0].copy is False

def test_busy_slot():
    counts=((8,0),(9,0),(1,0))
    s=search(counts,[Slot(2,expert=0,layer=0,ready=True,busy=True)],max_actions=1)
    assert not s.actions

def test_multi_action_counterexample():
    counts=((40,0,0,0),(39,0,0,0),(1,0,0,0),(1,0,0,0))
    s=search(counts,[Slot(2),Slot(3)],max_actions=2,beam_width=64)
    assert len(s.actions)==2
    print('EXPERT_SEARCH_PASS',s.rank_loads,s.actions)

if __name__=='__main__': test_identity_and_reuse(); test_busy_slot(); test_multi_action_counterexample()
