import argparse, os, pickle, random
from collections import defaultdict

random.seed(42)

def load_edges(edge_dir):
    # 期望有三个 csv：edges_drug_target_std.csv, edges_protein_pathway_std.csv, edges_pathway_pathway_std.csv
    # 列形如：src_id,dst_id （字符串ID，与你 nodes_* 表一致）
    import pandas as pd
    DT = pd.read_csv(os.path.join(edge_dir, "edges_drug_target_std.csv"))
    TP = pd.read_csv(os.path.join(edge_dir, "edges_protein_pathway_std.csv"))
    PP = pd.read_csv(os.path.join(edge_dir, "edges_pathway_pathway_std.csv"))
    return DT, TP, PP

def build_adj(DT, TP, PP):
    d2t, t2d = defaultdict(set), defaultdict(set)
    t2p, p2t = defaultdict(set), defaultdict(set)
    p2p = defaultdict(set)
    for a,b in DT.itertuples(index=False):
        d2t[a].add(b); t2d[b].add(a)
    for a,b in TP.itertuples(index=False):
        t2p[a].add(b); p2t[b].add(a)
    for a,b in PP.itertuples(index=False):
        p2p[a].add(b); p2p[b].add(a)  # 无向
    return d2t,t2d,t2p,p2t,p2p

def enumerate_paths_for_pair(a,b, d2t,t2d,t2p,p2t,p2p, enable_pp=False, K=16):
    paths = []
    # M1: D-T-D
    Ta = d2t.get(a,()); Tb = d2t.get(b,())
    interT = Ta & Tb
    for t in list(interT)[:K]:
        paths.append(("M1",[a,("T",t),b]))
        if len(paths)>=K: return paths
    # M2: D-T-P-T-D
    if len(paths)<K:
        for t1 in list(Ta):
            for p in list(t2p.get(t1,())):
                for t2 in list(p2t.get(p,())):
                    if t2 in Tb and a!=b:
                        paths.append(("M2",[a,("T",t1),("P",p),("T",t2),b]))
                        if len(paths)>=K: return paths
    # M3: D-T-P-P-T-D (可选)
    if enable_pp and len(paths)<K:
        for t1 in list(Ta):
            for p1 in list(t2p.get(t1,())):
                for p2 in list(p2p.get(p1,())):
                    for t2 in list(p2t.get(p2,())):
                        if t2 in Tb and a!=b:
                            paths.append(("M3",[a,("T",t1),("P",p1),("P",p2),("T",t2),b]))
                            if len(paths)>=K: return paths
    return paths

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--edge_dir", default="Datasets/normalized")
    ap.add_argument("--pairs_csv", required=True, help="")
    ap.add_argument("--out", default="GNN/GNN_datasets/cpa_cache.pkl")
    ap.add_argument("--k_paths", type=int, default=16)
    ap.add_argument("--enable_pp", action="store_true")
    args=ap.parse_args()

    import pandas as pd
    DT,TP,PP = load_edges(args.edge_dir)
    d2t,t2d,t2p,p2t,p2p = build_adj(DT,TP,PP)

    pairs = pd.read_csv(args.pairs_csv)  # 需要有 drug_a, drug_b 列（已规范 a<=b）
    cache = {}
    for i,(a,b) in enumerate(pairs[['drug_a','drug_b']].itertuples(index=False)):
        ps = enumerate_paths_for_pair(a,b, d2t,t2d,t2p,p2t,p2p, args.enable_pp, args.k_paths)
        cache[(a,b)] = ps
        if (i+1)%5000==0: print(f"[{i+1}] cached pairs")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"wb") as f:
        pickle.dump(cache,f)
    print(f"[OK] CPA cache -> {args.out}  pairs: {len(cache)}")

if __name__=="__main__":
    main()