import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

def latest_release_dir(root=Path("Datasets/release")) -> Path:
    vs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not vs: raise SystemExit("No release folder found.")
    return vs[-1]

ver = latest_release_dir()
out_dir = ver / "features"
out_dir.mkdir(exist_ok=True)

# ---------- Load base data ----------
dt = pd.read_csv(ver / "edges_drug_target.csv", dtype=str)
pp = pd.read_csv(ver / "edges_protein_pathway.csv", dtype=str)
drug_col, prot_col = dt.columns[:2]
prot_col_pp, pwy_col = pp.columns[:2]

# ---------- Load STRING (Protein–Protein) ----------
string_file = Path("Datasets/STRING/9606.protein.links.v12.0.txt")
ppi = pd.read_csv(string_file, sep=' ')
ppi = ppi[ppi['combined_score'] >= 700]
# remove species prefix "9606."
ppi['protein1'] = ppi['protein1'].str.replace('9606.', '', regex=False)
ppi['protein2'] = ppi['protein2'].str.replace('9606.', '', regex=False)

prot_neighbors = {}
for a, b in zip(ppi['protein1'], ppi['protein2']):
    prot_neighbors.setdefault(a, set()).add(b)
    prot_neighbors.setdefault(b, set()).add(a)

print(f"[INFO] Loaded {len(ppi):,} STRING edges")

# ---------- Build mappings ----------
drug2p = dt.groupby(drug_col)[prot_col].apply(lambda s: set(s.dropna())).to_dict()
prot2pwy = pp.groupby(prot_col_pp)[pwy_col].apply(lambda s: set(s.dropna())).to_dict()

def expand_to_ppi(prot_set):
    expanded = set(prot_set)
    for p in list(prot_set):
        expanded |= prot_neighbors.get(p, set())
    return expanded

def drug_to_pathways(d):
    ps = drug2p.get(d, set())
    pwys = set()
    for p in ps:
        pwys |= prot2pwy.get(p, set())
    return pwys

def expand_pathways(pwy_set, pathway_edges):
    expanded = set(pwy_set)
    for p in list(pwy_set):
        expanded |= pathway_edges.get(p, set())
    return expanded

# load pathway hierarchy if exists
pathway_edges_file = ver / "edges_pathway_pathway.csv"
pathway_edges = {}
if pathway_edges_file.exists():
    df_pwy = pd.read_csv(pathway_edges_file, dtype=str)
    parent_col, child_col = df_pwy.columns[:2]
    for p, c in zip(df_pwy[parent_col], df_pwy[child_col]):
        pathway_edges.setdefault(p, set()).add(c)

# ---------- Build features ----------
drug_list = list(drug2p.keys())
rows = []

print(f"[INFO] Computing pair features for {len(drug_list)} drugs...")
for a, b in tqdm(combinations(drug_list, 2), total=len(drug_list)*(len(drug_list)-1)//2, ncols=100):
    P1, P2 = drug2p[a], drug2p[b]
    if not P1 and not P2:
        continue

    # 1-hop
    inter1 = len(P1 & P2)
    union1 = len(P1 | P2)
    jacc1 = inter1 / union1 if union1 else 0

    # 2-hop (via STRING)
    P1x, P2x = expand_to_ppi(P1), expand_to_ppi(P2)
    inter2 = len(P1x & P2x)
    union2 = len(P1x | P2x)
    jacc2 = inter2 / union2 if union2 else 0

    # pathways
    Pw1, Pw2 = drug_to_pathways(a), drug_to_pathways(b)
    Pw1x, Pw2x = expand_pathways(Pw1, pathway_edges), expand_pathways(Pw2, pathway_edges)

    inter_pwy = len(Pw1 & Pw2)
    jacc_pwy = inter_pwy / len(Pw1 | Pw2) if (Pw1 | Pw2) else 0
    inter_pwy2 = len(Pw1x & Pw2x)
    jacc_pwy2 = inter_pwy2 / len(Pw1x | Pw2x) if (Pw1x | Pw2x) else 0

    rows.append((a, b, inter1, jacc1, inter2, jacc2, inter_pwy, jacc_pwy, inter_pwy2, jacc_pwy2))

feat = pd.DataFrame(rows, columns=[
    "drug_a", "drug_b",
    "shared_targets_1hop", "jaccard_targets_1hop",
    "shared_targets_2hop", "jaccard_targets_2hop",
    "shared_pathways_1hop", "jaccard_pathways_1hop",
    "shared_pathways_2hop", "jaccard_pathways_2hop"
])

out_file = out_dir / "mechanistic_pair_features_v2.csv"
feat.to_csv(out_file, index=False)
print(f"[DONE] Enhanced features written to {out_file}")