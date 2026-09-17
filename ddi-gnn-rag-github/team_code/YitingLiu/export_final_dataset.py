from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# -------------------- paths --------------------
ROOT   = Path(__file__).resolve().parent
NORM   = ROOT / "Datasets" / "normalized"
MERGED = NORM / "merged"
CLEAN  = NORM / "cleaned"

# 选用“filled”版本（如果存在），否则退回 clean 版本
DRUGS_FILE = (CLEAN / "drugs_master_filled.csv") if (CLEAN / "drugs_master_filled.csv").exists() \
             else (CLEAN / "drugs_master_clean.csv")

# 创建 release 目录（带时间戳）
stamp = datetime.now().strftime("%Y%m%d_%H%M")
RELEASE = ROOT / "Datasets" / "release" / f"v_{stamp}"
(NODES := RELEASE / "nodes").mkdir(parents=True, exist_ok=True)
(EDGES := RELEASE / "edges").mkdir(parents=True, exist_ok=True)
(FEATS := RELEASE / "features").mkdir(parents=True, exist_ok=True)
(META  := RELEASE / "meta").mkdir(parents=True, exist_ok=True)

print(f"[INFO] Using DRUGS from: {DRUGS_FILE}")
print(f"[INFO] Release folder:   {RELEASE}")

# -------------------- helpers --------------------
def read_csv(p, **kw):
    return pd.read_csv(p, dtype=str, keep_default_na=True, na_values=["", "\\N"], **kw)

def safe_len(df): return 0 if df is None else len(df)

# -------------------- load sources --------------------
drugs = read_csv(DRUGS_FILE)
# 补充类型信息（小分子 / 生物制剂）
try:
    db2 = read_csv(MERGED / "drugs_patched2.csv")[["drug_id","type","approval_status"]]
    drugs = drugs.merge(db2, on="drug_id", how="left")
except Exception:
    drugs["type"] = pd.NA
    drugs["approval_status"] = pd.NA

# 结构可用性标记
drugs["has_structure"] = drugs["smiles"].notna() & (drugs["smiles"].astype(str)!="")

# 小分子标记：若 type 含 small 或有结构，则视作小分子（保守做法）
drugs["is_small_molecule"] = drugs["type"].str.contains("small", case=False, na=False) | drugs["has_structure"]

# 边与特征
tgt_raw = read_csv(CLEAN / "edges_drug_target_clean.csv") if (CLEAN / "edges_drug_target_clean.csv").exists() else read_csv(MERGED / "edges_drug_target.csv")
pth_raw = read_csv(CLEAN / "edges_drug_pathway_clean.csv") if (CLEAN / "edges_drug_pathway_clean.csv").exists() else read_csv(MERGED / "edges_drug_pathway.csv")
atc     = read_csv(CLEAN / "feat_atc_clean.csv") if (CLEAN / "feat_atc_clean.csv").exists() else read_csv(MERGED / "feat_atc.csv")
syn     = read_csv(CLEAN / "feat_synonym_clean.csv") if (CLEAN / "feat_synonym_clean.csv").exists() else read_csv(MERGED / "feat_synonym.csv")
props   = read_csv(CLEAN / "feat_properties_clean.csv") if (CLEAN / "feat_properties_clean.csv").exists() else read_csv(MERGED / "feat_properties.csv")

# -------------------- nodes --------------------
# drugs 节点
drug_nodes = drugs[["drug_id","name","smiles","inchikey","has_structure","is_small_molecule","type","approval_status"]].drop_duplicates()
drug_nodes.to_csv(NODES / "drugs.csv", index=False)

# proteins 节点（从边抽取）
prot_nodes = (tgt_raw[["uniprot_id"]].dropna()
              .assign(uniprot_id=lambda d: d["uniprot_id"].str.upper().str.strip())
              .drop_duplicates())
prot_nodes.to_csv(NODES / "proteins.csv", index=False)

# pathways 节点（从边抽取）
path_nodes = pth_raw[["pathway_id"]].dropna().drop_duplicates()
path_nodes.to_csv(NODES / "pathways.csv", index=False)

# -------------------- edges（只保留两端存在于节点集的边） --------------------
drug_set = set(drug_nodes["drug_id"])
prot_set = set(prot_nodes["uniprot_id"])
pth_set  = set(path_nodes["pathway_id"])

edges_tgt = (tgt_raw.dropna(subset=["drug_id","uniprot_id"])
             .assign(uniprot_id=lambda d: d["uniprot_id"].str.upper().str.strip()))
edges_tgt = edges_tgt[edges_tgt["drug_id"].isin(drug_set) & edges_tgt["uniprot_id"].isin(prot_set)]
edges_tgt = edges_tgt.drop_duplicates()
edges_tgt.to_csv(EDGES / "drug_target.csv", index=False)

edges_pth = pth_raw.dropna(subset=["drug_id","pathway_id"])
edges_pth = edges_pth[edges_pth["drug_id"].isin(drug_set) & edges_pth["pathway_id"].isin(pth_set)]
edges_pth = edges_pth.drop_duplicates()
edges_pth.to_csv(EDGES / "drug_pathway.csv", index=False)

# -------------------- features --------------------
atc = atc[atc["drug_id"].isin(drug_set)].drop_duplicates()
atc.to_csv(FEATS / "atc.csv", index=False)

syn = syn[syn["drug_id"].isin(drug_set)].drop_duplicates()
syn.to_csv(FEATS / "synonyms.csv", index=False)

props = props[props["drug_id"].isin(drug_set)].drop_duplicates()
props.to_csv(FEATS / "properties.csv", index=False)

# 小分子/生物制剂划分
drug_nodes.query("is_small_molecule == True").to_csv(FEATS / "drugs_small_molecule.csv", index=False)
drug_nodes.query("is_small_molecule == False").to_csv(FEATS / "drugs_biologic.csv", index=False)

# 可选：保存 DrugCentral 变体及汇总（若存在）
variants_path = CLEAN / "drugcentral_variants.csv"
summary_path  = CLEAN / "drugcentral_variant_summary.csv"
if variants_path.exists():
    pd.read_csv(variants_path, dtype=str).to_csv(META / "drugcentral_variants.csv", index=False)
if summary_path.exists():
    pd.read_csv(summary_path, dtype=str).to_csv(META / "drugcentral_variant_summary.csv", index=False)

# -------------------- manifest & README --------------------
manifest = {
    "version": stamp,
    "paths": {
        "nodes": {
            "drugs": str((NODES / "drugs.csv").relative_to(RELEASE)),
            "proteins": str((NODES / "proteins.csv").relative_to(RELEASE)),
            "pathways": str((NODES / "pathways.csv").relative_to(RELEASE)),
        },
        "edges": {
            "drug_target": str((EDGES / "drug_target.csv").relative_to(RELEASE)),
            "drug_pathway": str((EDGES / "drug_pathway.csv").relative_to(RELEASE)),
        },
        "features": {
            "atc": str((FEATS / "atc.csv").relative_to(RELEASE)),
            "synonyms": str((FEATS / "synonyms.csv").relative_to(RELEASE)),
            "properties": str((FEATS / "properties.csv").relative_to(RELEASE)),
            "drugs_small_molecule": str((FEATS / "drugs_small_molecule.csv").relative_to(RELEASE)),
            "drugs_biologic": str((FEATS / "drugs_biologic.csv").relative_to(RELEASE)),
        }
    },
    "counts": {
        "nodes": {
            "drugs": safe_len(drug_nodes),
            "proteins": safe_len(prot_nodes),
            "pathways": safe_len(path_nodes),
        },
        "edges": {
            "drug_target": safe_len(edges_tgt),
            "drug_pathway": safe_len(edges_pth),
        },
        "coverage": {
            "drugs_has_smiles": int((drug_nodes["smiles"].notna() & (drug_nodes["smiles"]!="")).sum()),
            "drugs_has_inchikey": int((drug_nodes["inchikey"].notna() & (drug_nodes["inchikey"]!="")).sum()),
            "small_molecule": int(drug_nodes["is_small_molecule"].sum()),
        }
    }
}
with open(META / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

readme = f"""# Unified Drug–Protein–Pathway Dataset (Release {stamp})

## Layout
- `nodes/drugs.csv` — columns: drug_id, name, smiles, inchikey, has_structure, is_small_molecule, type, approval_status
- `nodes/proteins.csv` — columns: uniprot_id
- `nodes/pathways.csv` — columns: pathway_id
- `edges/drug_target.csv` — columns: drug_id, uniprot_id, [*, source if present]
- `edges/drug_pathway.csv` — columns: drug_id, pathway_id, [*, source if present]
- `features/atc.csv` — columns: drug_id, atc_code
- `features/synonyms.csv` — columns: drug_id, synonym
- `features/properties.csv` — columns: drug_id, kind, value
- splits: `features/drugs_small_molecule.csv`, `features/drugs_biologic.csv`
- meta: manifest, optional DrugCentral variant audit

## Notes
- Non-structural biologics are *kept* as nodes; structure features should use `has_structure` as a mask.
- Edges are filtered to ensure both endpoints exist in node sets.
- Reactome edges are expected to be human-specific (Homo sapiens).

## Quick sanity checks
- drugs with SMILES: {manifest["counts"]["coverage"]["drugs_has_smiles"]} / {manifest["counts"]["nodes"]["drugs"]}
- drug–target edges: {manifest["counts"]["edges"]["drug_target"]}
- drug–pathway edges: {manifest["counts"]["edges"]["drug_pathway"]}
"""
(RELEASE / "meta" / "README.md").write_text(readme, encoding="utf-8")

print("\n✅ Export complete.")
print(f"Release folder: {RELEASE}")
print("Manifest:", META / "manifest.json")