#!/usr/bin/env python3
import pandas as pd, json
from pathlib import Path

def latest_release_dir(root=Path("Datasets/release")) -> Path:
    vs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not vs: raise SystemExit("No release folder found.")
    return vs[-1]

ver = latest_release_dir()

# ---------- load nodes ----------
def load_ids(path):
    df = pd.read_csv(path, dtype=str)
    col = df.columns[0]
    ids = df[col].astype(str).str.strip().drop_duplicates().reset_index(drop=True)
    return ids

nodes = {
    "drug":    ver/"nodes_drug.csv",
    "protein": ver/"nodes_protein.csv",
    "pathway": ver/"nodes_pathway.csv",
}
id_maps, meta = {}, {"nodes":{}, "edges":{}}
out_maps = (ver/"id_maps"); out_maps.mkdir(exist_ok=True)
out_idx  = (ver/"edges_idx"); out_idx.mkdir(exist_ok=True)

for kind, p in nodes.items():
    ids = load_ids(p)
    mp = {v:i for i,v in enumerate(ids)}
    id_maps[kind] = mp
    pd.DataFrame({"external_id": ids, f"{kind}_id": range(len(ids))}).to_csv(out_maps/f"{kind}_id_map.csv", index=False)
    meta["nodes"][kind] = len(ids)

# ---------- load edges helpers ----------
def read_csv(path):
    return pd.read_csv(path, dtype=str)

def pick(colnames, keywords):
    cols = [c.strip().lower() for c in colnames]
    for k in keywords:
        hit = [c for c in cols if k in c]
        if hit: return hit[0]
    return None

def map_series(s, mp, name):
    s = s.astype(str).str.strip()
    x = s.map(mp)
    missing = x.isna().sum()
    if missing:
        # 只警告，不中断；后续 dropna
        print(f"[WARN] {name}: {missing} ids not found in id_map; will drop.")
    return x

# ---------- edges: drug_target ----------
edt = ver/"edges_drug_target.csv"
if edt.exists():
    e = read_csv(edt)
    e.columns = [c.strip().lower() for c in e.columns]
    drug_col = pick(e.columns, ["drug"])
    prot_col = pick(e.columns, ["uniprot", "protein"])
    if not drug_col or not prot_col:
        raise SystemExit(f"Cannot find drug/protein columns in {edt.name}: {list(e.columns)}")
    s = map_series(e[drug_col], id_maps["drug"], "drug")
    t = map_series(e[prot_col], id_maps["protein"], "protein")
    idx = pd.DataFrame({"src": s, "dst": t}).dropna().astype(int)
    idx.to_csv(out_idx/"drug_target.csv", index=False)
    meta["edges"]["drug_target"] = {"count": len(idx), "etype": ("drug","targets","protein")}
    print("[OK] edges_idx/drug_target.csv", len(idx))

# ---------- edges: protein_pathway ----------
epp = ver/"edges_protein_pathway.csv"
if epp.exists():
    e = read_csv(epp)
    e.columns = [c.strip().lower() for c in e.columns]
    prot_col = pick(e.columns, ["uniprot", "protein"])
    pwy_col  = pick(e.columns, ["pathway"])
    if not prot_col or not pwy_col:
        raise SystemExit(f"Cannot find protein/pathway columns in {epp.name}: {list(e.columns)}")
    s = map_series(e[prot_col], id_maps["protein"], "protein")
    t = map_series(e[pwy_col],  id_maps["pathway"], "pathway")
    idx = pd.DataFrame({"src": s, "dst": t}).dropna().astype(int)
    idx.to_csv(out_idx/"protein_pathway.csv", index=False)
    meta["edges"]["protein_pathway"] = {"count": len(idx), "etype": ("protein","in_pathway","pathway")}
    print("[OK] edges_idx/protein_pathway.csv", len(idx))

# ---------- edges: pathway_pathway ----------
epp2 = ver/"edges_pathway_pathway.csv"
if epp2.exists():
    e = read_csv(epp2)
    e.columns = [c.strip().lower() for c in e.columns]
    parent_col = pick(e.columns, ["parent_pathway", "parent"])
    child_col  = pick(e.columns, ["child_pathway", "child"])
    if not parent_col or not child_col:
        raise SystemExit(f"Cannot find parent/child columns in {epp2.name}: {list(e.columns)}")
    s = map_series(e[parent_col], id_maps["pathway"], "parent_pathway")
    t = map_series(e[child_col],  id_maps["pathway"], "child_pathway")
    idx = pd.DataFrame({"src": s, "dst": t}).dropna().astype(int)
    idx.to_csv(out_idx/"pathway_pathway.csv", index=False)
    meta["edges"]["pathway_pathway"] = {"count": len(idx), "etype": ("pathway","is_parent_of","pathway")}
    print("[OK] edges_idx/pathway_pathway.csv", len(idx))

with open(ver/"graph_meta.json","w") as f:
    json.dump(meta, f, indent=2)
print("[DONE] id maps + indexed edges written. meta:", json.dumps(meta, indent=2))