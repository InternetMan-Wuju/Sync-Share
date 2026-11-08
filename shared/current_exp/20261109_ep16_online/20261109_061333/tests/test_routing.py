import unittest,sys,random
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plugin'))
from experiment import local_counts,remap
Top=namedtuple('Top','topk_ids topk_weights')
class Routing(unittest.TestCase):
    def test_invalid_and_reserved_not_expert_zero(self):
        logical,valid,c=local_counts(torch.tensor([[-1,272,16,33,0,17,271]]))
        self.assertEqual(c.sum().item(),2)
        self.assertEqual(c[0].item(),1);self.assertEqual(c[16].item(),1)
        self.assertEqual(valid.tolist(),[[False,False,False,False,True,True,False]])
    def test_multisource_branch_identity_and_integer_quota(self):
        rng=random.Random(917)
        for trial in range(40):
            e=rng.randrange(256);src=e//16;dst=(src+1+trial%15)%16
            inputs=[];counts=[]
            for rank in range(16):
                values=[rng.randrange(256) for _ in range(64)]
                for i in range(rank%8):values[i]=e
                logical=torch.tensor(values).reshape(8,8)
                physical=logical+logical//16
                weights=torch.rand((8,8))
                top=Top(physical,weights);inputs.append((logical,top))
                counts.append(torch.bincount(logical.reshape(-1),minlength=256).tolist())
            available=sum(c[e] for c in counts);quota=rng.randrange(available+1)
            p={'expert':e,'src':src,'dst':dst,'q':quota}
            moved=0;seen=0
            for r,(logical,top) in enumerate(inputs):
                got,q=remap(SimpleNamespace(rank=r),logical,top,p,counts)
                for old,new in zip(logical.reshape(-1).tolist(),got.topk_ids.reshape(-1).tolist()):
                    expected=(old+old//16)
                    if old==e:
                        if seen<quota:expected=dst*17+16;moved+=1
                        seen+=1
                    self.assertEqual(new,expected)
                self.assertIs(got.topk_weights,top.topk_weights)
            self.assertEqual(moved,quota)
if __name__=='__main__':unittest.main(verbosity=2)
