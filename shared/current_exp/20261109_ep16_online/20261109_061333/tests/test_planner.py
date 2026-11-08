import sys,random,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plugin'))
from planner import plan,source_quotas
class Tests(unittest.TestCase):
    def test_expert_specific_prefix(self):
        self.assertEqual(source_quotas([[99,6],[88,4]],1,7),[6,1])
        self.assertEqual(source_quotas([[99,0],[88,4]],1,3),[0,3])
        self.assertEqual(source_quotas([[0,6],[1,4]],1,0),[0,0])
        self.assertEqual(source_quotas([[0,6],[1,4]],1,10),[6,4])
    def test_empty_uniform(self):
        self.assertIsNone(plan([0]*256));self.assertIsNone(plan([32]*256))
    def test_hot(self):
        counts=[0]*256;counts[43]=960;counts[40]=64
        p=plan(counts);self.assertIsNotNone(p);self.assertEqual(sum(p['old_loads']),sum(p['new_loads']))
        self.assertLess(max(p['new_loads']),max(p['old_loads']))
    def test_random_conservation_and_independent_bound(self):
        rng=random.Random(42)
        for _ in range(200):
            c=[rng.randrange(32) for _ in range(256)];c[rng.randrange(256)]+=rng.randrange(1000)
            p=plan(c)
            if p is None:continue
            src,dst,e,q=(p[k] for k in ('src','dst','expert','q'))
            self.assertLessEqual(q,c[e]);self.assertEqual(src,e//16)
            # Enumerate every integer q for the chosen pair independently.
            lo=p['old_loads'];best=min(max(v-(q if r==src else 0)+(q if r==dst else 0) for r,v in enumerate(lo)) for q in range(c[e]+1))
            self.assertEqual(max(p['new_loads']),best)
            self.assertEqual(sum(lo),sum(p['new_loads']))
            for r in range(16):
                if r not in (src,dst):self.assertEqual(lo[r],p['new_loads'][r])
if __name__=='__main__':unittest.main(verbosity=2)
