"""One replica per destination, one move per call; count-based test planner."""
def plan(counts, ep=16, slots=16):
    assert len(counts)==ep*slots and all(n>=0 for n in counts)
    loads=[sum(counts[r*slots:(r+1)*slots]) for r in range(ep)]
    src=max(range(ep), key=lambda r: loads[r])
    candidates=sorted(range(src*slots,(src+1)*slots), key=lambda e: -counts[e])[:2]
    receivers=sorted((r for r in range(ep) if r!=src),key=lambda r:loads[r])[:3]
    best=None
    for e in candidates:
        if counts[e]<=2*sum(counts)/len(counts): continue
        for dst in receivers:
            q0=min(counts[e],max(0,(loads[src]-loads[dst])//2))
            for q in sorted({q0//2,q0}):
                if not q:continue
                new=loads.copy();new[src]-=q;new[dst]+=q
                saving=max(loads)-max(new)
                if saving<=0:continue
                candidate={'expert':e,'src':src,'dst':dst,'q':q,'old_loads':loads,'new_loads':new,'saving_tasks':saving}
                if best is None or (saving,-q)>(best['saving_tasks'],-best['q']): best=candidate
    return best

def source_quotas(by_source, expert, q):
    result=[]
    assert 0<=q<=sum(row[expert] for row in by_source)
    for row in by_source:
        take=min(q,row[expert]);result.append(take);q-=take
    assert q==0
    return result
