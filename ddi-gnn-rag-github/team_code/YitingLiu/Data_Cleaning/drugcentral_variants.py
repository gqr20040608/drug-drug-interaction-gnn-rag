from pathlib import Path
import pandas as pd
from rdkit import Chem

# -------------------- paths --------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Datasets" / "normalized" / "merged" / "drugs_master.csv"
OUT = ROOT / "Datasets" / "normalized" / "cleaned"
OUT.mkdir(parents=True, exist_ok=True)  # auto-create output directory

print(f"Reading from: {SRC}")
df = pd.read_csv(SRC, dtype=str)
df = df.fillna("")  # Replace NaN with empty string
df = df[df["drug_id"].notna()]

# -------------------- helper: check SMILES validity --------------------
def is_valid_smiles(s):
    """Return True if a SMILES string is chemically valid."""
    if not s or s.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(s)
        return mol is not None
    except Exception:
        return False

df["valid_smiles"] = df["smiles"].apply(is_valid_smiles)

# -------------------- group by drug_id --------------------
clean_rows = []
variants = []

for drug, sub in df.groupby("drug_id"):
    # 优先保留第一个合法 SMILES 的记录
    valid = sub[sub["valid_smiles"]]
    if len(valid) > 0:
        rep = valid.iloc[0]
    else:
        rep = sub.iloc[0]

    # 保存代表行
    clean_rows.append(rep[["drug_id", "name", "smiles", "inchikey", "drugcentral_id"]])

    # 记录同一 drug_id 下的其他 DrugCentral 变体
    others = sub[~sub.index.isin([rep.name])]
    if len(others) > 0:
        others = others.assign(main_drug_id=drug)
        variants.append(others[["main_drug_id", "drugcentral_id", "smiles", "inchikey"]])

# -------------------- save results --------------------
df_clean = pd.DataFrame(clean_rows)
df_clean.to_csv(OUT / "drugs_master_clean.csv", index=False)

if len(variants) > 0:
    df_variants = pd.concat(variants, ignore_index=True)
    df_variants.to_csv(OUT / "drugcentral_variants.csv", index=False)
else:
    print("No variants found.")

# -------------------- summary statistics --------------------
summary = df.groupby("drug_id")["drugcentral_id"].nunique().reset_index()
summary = summary.rename(columns={"drugcentral_id": "num_drugcentral_variants"})
summary.to_csv(OUT / "drugcentral_variant_summary.csv", index=False)

print("✅ Cleaning complete.")
print(f"  → {len(df_clean)} unique drug entries saved to: {OUT / 'drugs_master_clean.csv'}")
if len(variants) > 0:
    print(f"  → {len(df_variants)} variant records saved to: {OUT / 'drugcentral_variants.csv'}")
print(f"  → Variant count summary saved to: {OUT / 'drugcentral_variant_summary.csv'}")