#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def latest_release_dir(root=Path("Datasets/release")) -> Path:
    vers = sorted([p for p in root.iterdir() if p.is_dir()])
    if not vers:
        raise SystemExit("No release folder found.")
    return vers[-1]

ver = latest_release_dir()

pp = ver / "edges_pathway_pathway.csv"
npw = ver / "nodes_pathway.csv"

# 读入
e = pd.read_csv(pp, dtype=str)
n = pd.read_csv(npw, dtype=str)

# 统一列名到小写、去空格
e.columns = [c.strip().lower() for c in e.columns]

# 自动定位列名
def pick(colnames, keywords):
    for k in keywords:
        hits = [c for c in colnames if k in c]
        if hits:
            return hits[0]
    return None

parent_col = pick(e.columns, ["parent_pathway", "parent"])
child_col  = pick(e.columns, ["child_pathway", "child"])
rel_col    = pick(e.columns, ["rel", "relation", "relationship"])
src_col    = pick(e.columns, ["source"])

need = [parent_col, child_col]
if any(c is None for c in need):
    raise SystemExit(f"Cannot find parent/child columns in: {list(e.columns)}")

# 规范列
e = e.rename(columns={parent_col: "parent", child_col: "child"})
if rel_col: e = e.rename(columns={rel_col: "rel"})
if src_col: e = e.rename(columns={src_col: "source"})

# 去首尾空格
for c in ["parent", "child"]:
    e[c] = e[c].astype(str).str.strip()

pathways = set(n.iloc[:, 0].astype(str).str.strip())

print(f"[STATS] edges_pathway_pathway: {len(e):,}  pathways: {len(pathways):,}")

# 规则检查
non_rhsa = ((~e["parent"].str.startswith("R-HSA")) | (~e["child"].str.startswith("R-HSA"))).sum()
self_loops = (e["parent"] == e["child"]).sum()
dups = e.duplicated(subset=["parent", "child"]).sum()
missing_parent = (~e["parent"].isin(pathways)).sum()
missing_child  = (~e["child"].isin(pathways)).sum()

print("[CHECK] non R-HSA ids:", non_rhsa)
print("[CHECK] self-loops:", self_loops)
print("[CHECK] duplicate edges:", dups)
print("[FK] missing parent pathways:", missing_parent)
print("[FK] missing child pathways :", missing_child)

# 可选：关系与来源的分布
if "rel" in e.columns:
    rel_counts = e["rel"].value_counts(dropna=False).to_dict()
    print("[REL] distribution:", rel_counts)
if "source" in e.columns:
    src_counts = e["source"].value_counts(dropna=False).to_dict()
    print("[SRC] distribution:", src_counts)