#!/usr/bin/env python3
"""最小玩具验证：4 rank / 8 专家 / 2 层，直接调用 stock 的 EPLB 算法。

目的：回答"EPLB 之后，每层的热点落在哪个 rank？层A和层B会不会相反？"
所有算法代码来自镜像 quay.io/ascend/sglang:v0.5.18-cann9.0.0-a3
（/sgl-workspace/sglang/python/sglang/srt/eplb/eplb_algorithms/），
已原样拷到 patches/stock_alg_ref/，未做任何修改。

真实配置：64 rank x 4 槽 = 256 专家，78 层。这里等比缩小成 4 rank x 2 槽 = 8 专家。
"""
import sys, pathlib
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patches" / "stock_alg_ref"))
import deepseek                      # 实际在跑的（auto 选中的）
import deepseek_vec                  # 带 snake 的向量版
import deepseek_vec_noflip           # 把 snake 那行注释掉的对照版

RANKS, EXPERTS, LAYERS = 4, 8, 2
LOCAL = EXPERTS // RANKS             # 每 rank 槽数 = 2

# 层A: 热点在专家0；层B: 同样的热度分布，但热点在专家7
W = torch.tensor([
    [100., 20., 10., 5., 1., 1., 1., 1.],   # 层A  hot=专家0
    [1., 1., 1., 1., 5., 10., 20., 100.],   # 层B  hot=专家7
])

def show(name, phy2log):
    """phy2log: [L, EXPERTS]，第 p 个物理槽放哪个逻辑专家；rank = p // LOCAL"""
    print(f"\n--- {name} ---")
    loads = {}
    for l in range(LAYERS):
        per = phy2log[l].reshape(RANKS, LOCAL)
        ld = W[l][per].sum(-1)
        loads[l] = ld
        tag = "层A" if l == 0 else "层B"
        hot_rank = int(ld.argmax())
        print(f"{tag}: " + "  ".join(
            f"r{r}=[{' '.join(f'e{int(e)}' for e in per[r])}]={float(ld[r]):.0f}"
            for r in range(RANKS)))
        print(f"     负载 = [{' '.join(f'{float(v):.0f}' for v in ld)}]   max/mean={float(ld.max()/ld.mean()):.2f}x  最热rank=r{hot_rank}")
    tot = loads[0] + loads[1]
    print(f"合计: [{' '.join(f'{float(v):.0f}' for v in tot)}]  max/mean={float(tot.max()/tot.mean()):.2f}x")

def asc_greedy(w, num_packs):
    """假想：如果排序方向反过来（从冷到热发牌），会得到什么"""
    x = w.sort(-1, descending=False).indices.tolist()
    packs = [[] for _ in range(num_packs)]
    load = [0.] * num_packs
    per = len(w) // num_packs
    for e in x:
        cand = [j for j in range(num_packs) if len(packs[j]) < per]
        p = min(cand, key=load.__getitem__)
        packs[p].append(e); load[p] += float(w[e])
    return packs, load

print("=" * 78)
print(f"玩具设定: {LAYERS} 层, {EXPERTS} 专家, {RANKS} rank, 每 rank {LOCAL} 槽")
print("=" * 78)

# 1) 真实运行的算法：deepseek（auto 在 num_groups=1 时选中的就是它）
phy2log, log2phy, logcnt = deepseek.rebalance_experts(
    weight=W, num_replicas=EXPERTS, num_groups=1, num_nodes=1,
    num_gpus=RANKS, enable_hierarchical=False)
show("① stock deepseek（我们这次实际跑的算法）", phy2log)

# 2) deepseek_vec（带 snake 的向量版）
tpe = W.reshape(1, LAYERS, EXPERTS)
p2l_vec, _, _ = deepseek_vec.rebalance_experts(
    tokens_per_expert=tpe, num_physical_experts=EXPERTS,
    num_local_physical_experts=LOCAL, num_groups=1, num_nodes=1,
    enable_hierarchical=False)
show("② stock deepseek_vec（带 snake：indices[:,:,1::2,:].flip(-1)）", p2l_vec)

# 3) 去掉 snake 的对照
p2l_nf, _, _ = deepseek_vec_noflip.rebalance_experts(
    tokens_per_expert=tpe, num_physical_experts=EXPERTS,
    num_local_physical_experts=LOCAL, num_groups=1, num_nodes=1,
    enable_hierarchical=False)
show("③ deepseek_vec 去掉 snake（对照）", p2l_nf)

# 4) 什么情况下才会出现 6-11-21-101
print("\n--- ④ 假设把排序方向反过来（从冷到热发牌，stock 里没有这个逻辑）---")
for tag, w in (("层A", W[0]), ("层B", W[1])):
    packs, load = asc_greedy(w, RANKS)
    print(f"{tag}: " + "  ".join(f"r{r}=[{','.join(f'e{e}' for e in packs[r])}]={load[r]:.0f}"
                                 for r in range(RANKS)))
