import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

def latest_release_dir(root=Path("Datasets/release")) -> Path:
    vs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not vs:
        raise SystemExit("No release folder found.")
    return vs[-1]

ver = latest_release_dir()
out_dir = ver / "features"
out_dir.mkdir(exist_ok=True)

# -------------------- load data --------------------
dt = pd.read_csv(ver / "edges_drug_target.csv", dtype=str)
pp = pd.read_csv(ver / "edges_protein_pathway.csv", dtype=str)

drug_col = dt.columns[0]
prot_col = dt.columns[1]
prot_col_pp = pp.columns[0]
pwy_col = pp.columns[1]

# -------------------- build mappings --------------------
drug2p = (
    dt.groupby(drug_col)[prot_col]
    .apply(lambda s: set(s.dropna().astype(str)))
    .to_dict()
)
prot2pwy = (
    pp.groupby(prot_col_pp)[pwy_col]
    .apply(lambda s: set(s.dropna().astype(str)))
    .to_dict()
)

def drug_to_pathways(drug):
    proteins = drug2p.get(drug, set())
    pwys = set()
    for p in proteins:
        pwys |= prot2pwy.get(p, set())
    return pwys

drug_list = list(drug2p.keys())
N = len(drug_list)
print(f"[INFO] {N:,} drugs loaded")

# Optional quick test
# drug_list = drug_list[:2000]

pairs_total = N * (N - 1) // 2
print(f"[INFO] Generating ~{pairs_total:,} drug pairs")

rows = []

# tqdm progress bar
for a, b in tqdm(combinations(drug_list, 2), total=pairs_total, ncols=100, desc="Computing pairs"):
    Pa, Pb = drug2p[a], drug2p[b]
    if not Pa and not Pb:
        continue

    inter_p = len(Pa & Pb)
    union_p = len(Pa | Pb)
    jacc_p = inter_p / union_p if union_p else 0.0

    A_pwy, B_pwy = drug_to_pathways(a), drug_to_pathways(b)
    inter_w = len(A_pwy & B_pwy)
    union_w = len(A_pwy | B_pwy)
    jacc_w = inter_w / union_w if union_w else 0.0

    rows.append((a, b, inter_p, jacc_p, inter_w, jacc_w))

# -------------------- save output --------------------
feat = pd.DataFrame(
    rows,
    columns=[
        "drug_a",
        "drug_b",
        "shared_targets",
        "jaccard_targets",
        "shared_pathways",
        "jaccard_pathways",
    ],
)

out_file = out_dir / "mechanistic_pair_features.csv"
feat.to_csv(out_file, index=False)
print(f"[DONE] {len(feat):,} pairs written to {out_file}")