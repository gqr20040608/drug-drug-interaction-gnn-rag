import pandas as pd, argparse, collections, os

def load_nodes(p):
    df = pd.read_csv(p)
    assert {"reactome_id","name","level"}.issubset(df.columns)
    return df

def load_edges(p):
    # 无表头：src,dst；或已标准化列
    try:
        df = pd.read_csv(p)
    except:
        df = pd.read_csv(p, header=None, names=["src_reactome_id","dst_reactome_id"])
    cols = list(df.columns)
    if {"src_reactome_id","dst_reactome_id"}.issubset(cols):
        src, dst = "src_reactome_id","dst_reactome_id"
    else:
        src, dst = cols[0], cols[1]
    df = df[[src,dst]].dropna().drop_duplicates()
    df.columns = ["src","dst"]
    return df

def compute_levels(nodes, edges):
    # 构建 DAG（Reactome 关系文件基本是父->子）
    parents = collections.defaultdict(set)
    children = collections.defaultdict(set)
    all_ids = set(nodes["reactome_id"])
    for s,d in edges.itertuples(index=False):
        if s in all_ids and d in all_ids:
            children[s].add(d)
            parents[d].add(s)

    # roots：没有父的节点
    roots = [nid for nid in all_ids if len(parents[nid])==0]
    # BFS 计算最短层级
    level = {nid: None for nid in all_ids}
    q = collections.deque()
    for r in roots:
        level[r] = 0
        q.append(r)
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            cand = (level[u] + 1) if level[u] is not None else 0
            if level[v] is None or cand < level[v]:
                level[v] = cand
                q.append(v)
    # 孤立节点或环导致仍为 None → 置 0
    nodes["level"] = nodes["reactome_id"].map(lambda x: level.get(x, 0) if level.get(x, None) is not None else 0)
    return nodes

def main(a):
    nodes = load_nodes(a.nodes)
    edges = load_edges(a.edges_rel)
    nodes2 = compute_levels(nodes, edges)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    nodes2.to_csv(a.out, index=False)
    print("[OK] wrote:", a.out, "levels stats:", nodes2["level"].describe())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="GNN/GNN_datasets/nodes_pathway_std.csv")
    ap.add_argument("--edges_rel", default="GNN/GNN_datasets/edges_pathway_pathway_std.csv")
    ap.add_argument("--out", default="GNN/GNN_datasets/nodes_pathway_std.csv")
    main(ap.parse_args())