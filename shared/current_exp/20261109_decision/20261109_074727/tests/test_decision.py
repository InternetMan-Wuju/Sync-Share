import sys,unittest,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plugin'))
from decision import decide,apply,rank_ms,slots,admit
from planner import source_quotas

def model(**kw):
    m=dict(tile=4,coef=[0.3,0.2,1.0],copy_ms=0,control_ms=0,remap_ms=0,planner_budget_ms=0,extra_comm_ms=0,margin_ms=0,costs_known=True,max_global_tasks=10000)
    m.update(kw);return m

class DecisionTests(unittest.TestCase):
    def test_fail_closed(self):
        for m in [None,{},model(costs_known=False),model(max_global_tasks=1),model(copy_ms=-1),model(coef=[float('nan'),1,1])]:
            self.assertIsNone(decide([20,0,0,0],m,ep=2)[0])
    def test_noop_cost_and_uniform(self):
        self.assertIsNone(decide([20,0,0,0],model(copy_ms=100),ep=2)[0])
        self.assertIsNone(decide([8]*4,model(),ep=2)[0])
        self.assertIsNone(decide([0]*4,model(),ep=2)[0])
    def test_untouched_bottleneck(self):
        self.assertIsNone(decide([40,0,40,0,40,0,0,0],model(),ep=4)[0])
    def test_independent_integer_exhaustion(self):
        rng=random.Random(3109)
        for _ in range(150):
            c=[rng.randrange(30) for _ in range(8)];m=model();p,d=decide(c,m,ep=4)
            base=slots(c,4);t=[rank_ms(v,m) for v in base];s=max(range(4),key=lambda r:(t[r],-r))
            exhaustive=[]
            for e in range(s*2,s*2+2):
                for dst in range(4):
                    if dst==s:continue
                    for q in range(1,c[e]+1):
                        out=apply(c,dict(expert=e,src=s,dst=dst,q=q),ep=4)
                        exhaustive.append(max(t)-max(rank_ms(v,m) for v in out))
            best=max(exhaustive,default=0)
            if p:
                self.assertAlmostEqual(p['predicted_net_ms'],best)
                out=apply(c,p,4)
                for e,n in enumerate(c):
                    self.assertEqual(out[e//2][e%2]+(out[p['dst']][-1] if e==p['expert'] else 0),n)
                self.assertEqual(sum(source_quotas([[n//2 for n in c],[n-n//2 for n in c]],p['expert'],p['q'])),p['q'])
            else:self.assertLessEqual(best,1e-10)
    def test_unknown_copy_pair(self):
        self.assertIsNone(decide([20,0,0,0],model(known_copy_pairs=[]),ep=2)[0])
    def test_precheck_and_compute_bound(self):
        self.assertFalse(admit(0,model()))
        self.assertFalse(admit(8,model(copy_ms=100)))
        self.assertTrue(admit(100,model()))
        rng=random.Random(23);m=model()
        for _ in range(100):
            c=[rng.randrange(1,25) for _ in range(4)];rows=max(c)
            upper=m['coef'][1]+m['coef'][2]*((rows+m['tile']-1)//m['tile']+1)
            base=max(rank_ms(v,m) for v in slots(c,2))
            for e,n in enumerate(c):
                for q in range(1,n+1):
                    out=apply(c,dict(src=e//2,dst=1-e//2,expert=e,q=q),2)
                    self.assertLessEqual(base-max(rank_ms(v,m) for v in out),upper+1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
