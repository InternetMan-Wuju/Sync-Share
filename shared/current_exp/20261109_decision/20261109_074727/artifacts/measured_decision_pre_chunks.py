"""Current-batch, one-replica cost gate. No prediction of future requests.

Model is an experimentally fitted monotone bank-time proxy, not a claim about
the Ascend kernel tile size. Unknown/out-of-domain costs fail closed.
"""
import math


def valid_model(model):
    try:
        return (model.get('costs_known') is True and isinstance(model['tile'],int) and model['tile']>0
                and len(model['coef'])==3
                and all(math.isfinite(float(x)) and x>=0 for x in model['coef'])
                and all(math.isfinite(float(model[k])) and model[k]>=0 for k in
                        ('copy_ms','control_ms','remap_ms','planner_budget_ms','extra_comm_ms','margin_ms')))
    except (KeyError,TypeError,ValueError,AttributeError):
        return False


def features(vector, tile):
    return (1, sum(n > 0 for n in vector), sum((n + tile - 1) // tile for n in vector))


def rank_ms(vector, model):
    if not any(vector):
        return 0.0
    return sum(a * b for a, b in zip(features(vector, model['tile']), model['coef']))


def admit(global_rows, model):
    """Cheap model-based bound before paying count exchange.

    Assumes distinct top-k experts per row, verified by this harness. At most
    global_rows branches of one expert can move. This bounds COMPUTE saving
    under this monotone model; it is not a theorem about communication savings.
    """
    if not valid_model(model) or global_rows<=0:
        return False
    if global_rows*8 > model['max_global_tasks']:
        return False
    upper = model['coef'][1] + model['coef'][2]*((global_rows+model['tile']-1)//model['tile']+1)
    cost = sum(model[k] for k in ('copy_ms','control_ms','remap_ms','planner_budget_ms','extra_comm_ms'))
    return upper > cost + model['margin_ms']


def slots(counts, ep=16):
    width = len(counts) // ep
    return [counts[r*width:(r+1)*width] + [0] for r in range(ep)]


def apply(counts, p, ep=16):
    out = slots(counts, ep)
    if p:
        width = len(counts) // ep
        assert p['src'] == p['expert'] // width and p['src'] != p['dst']
        assert 0 < p['q'] <= counts[p['expert']]
        out[p['src']][p['expert'] % width] -= p['q']
        out[p['dst']][-1] += p['q']
    assert sum(map(sum, out)) == sum(counts)
    return out


def decide(counts, model, ep=16):
    assert len(counts) % ep == 0 and all(isinstance(n, int) and n >= 0 for n in counts)
    if not valid_model(model):
        return None, {'reason': 'unknown_cost'}
    if not all(math.isfinite(float(x)) and x >= 0 for x in model['coef']):
        return None, {'reason': 'invalid_model'}
    if model.get('max_global_tasks') is not None and sum(counts) > model['max_global_tasks']:
        return None, {'reason': 'outside_calibration_domain'}
    vec = slots(counts, ep); width = len(counts) // ep; tile = model['tile']
    times = [rank_ms(v, model) for v in vec]; before = max(times)
    src = max(range(ep), key=lambda r: (times[r], -r))
    experts = sorted(range(src*width, (src+1)*width), key=lambda e: (-counts[e], e))[:2]
    pairs=model.get('known_copy_pairs')
    receivers = sorted((r for r in range(ep) if r != src and (pairs is None or [src,r] in pairs)), key=lambda r: (times[r], r))[:3]
    loads = list(map(sum, vec)); best = None; candidates = 0
    overhead = sum(model[k] for k in ('copy_ms', 'control_ms', 'remap_ms', 'planner_budget_ms', 'extra_comm_ms'))
    for dst in receivers:
        unchanged = max((times[r] for r in range(ep) if r not in (src, dst)), default=0)
        sf = features(vec[src], tile); df = features(vec[dst], tile)
        for e in experts:
            n = counts[e]
            if not n:
                continue
            def pair(q):
                st = sf[2] - (n+tile-1)//tile + (n-q+tile-1)//tile
                dt = df[2] + (q+tile-1)//tile
                sa = sf[1] - int(q == n); da = df[1] + 1
                a,b,c = model['coef']
                ts = a+b*sa+c*st if sa else 0.0
                td = a+b*da+c*dt
                return ts, td
            # Source time is monotone decreasing, destination increasing.
            # Their crossing minimizes max(source,destination), including tiles.
            lo,hi=1,n
            while lo<hi:
                mid=(lo+hi)//2;ts,td=pair(mid)
                if ts<=td:hi=mid
                else:lo=mid+1
            qs={1,n,lo,max(1,lo-1)}
            target=min(max(unchanged,*pair(q)) for q in qs)
            # Pick the smallest quota on the optimum plateau: less token traffic.
            lo,hi=1,n
            while lo<hi:
                mid=(lo+hi)//2
                if pair(mid)[0]<=target:hi=mid
                else:lo=mid+1
            qs.add(lo)
            for q in sorted(qs):
                ts,td=pair(q)
                after = max(unchanged, ts, td); gain = before-after-overhead
                candidates += 1
                score = (gain, -q, -e, -dst)
                if best is None or score > best[0]:
                    best = (score, {'expert':e,'src':src,'dst':dst,'q':q,
                        'old_loads':loads,'new_loads':[v-q if r==src else v+q if r==dst else v for r,v in enumerate(loads)],
                        'predicted_before_compute_ms':before,'predicted_after_compute_ms':after,
                        'predicted_net_ms':gain,'priced_overhead_ms':overhead})
    detail = {'reason':'no_candidate','candidates':candidates,'before_compute_ms':before,'overhead_ms':overhead}
    if best:
        p = best[1]; detail.update(best_candidate=p)
        if p['predicted_net_ms'] > model['margin_ms']:
            detail['reason'] = 'predicted_profitable'
            return p, detail
        detail['reason'] = 'not_profitable_with_margin'
    return None, detail
