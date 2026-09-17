import pandas as pd
from pathlib import Path

root = Path("Datasets/Reactome")
src  = root / "Reactome_Uniprot_to_Pathways.csv"
dst  = root / "Reactome_Uniprot_to_Pathways_human.csv"
pwyh = root / "ReactomePathways_human.csv"

# 1) read Uniprot→Pathways (tab or comma)
try:
    df = pd.read_csv(src, sep="\t", dtype=str, on_bad_lines="skip", engine="python")
except Exception:
    df = pd.read_csv(src, sep=",", dtype=str, on_bad_lines="skip", engine="python")

# clean whitespace
df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

# 2) keep only rows whose LAST column == "Homo sapiens"
species_col = df.columns[-1]
df = df[df[species_col] == "Homo sapiens"]

# 3) (optional) also restrict to pathway IDs present in ReactomePathways_human.csv (first col)
if pwyh.exists():
    pwy = pd.read_csv(pwyh, sep=",", dtype=str, on_bad_lines="skip", engine="python")
    pwy = pwy.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    human_ids = set(pwy.iloc[:, 0].dropna().unique())
    # assume the 2nd column in Uniprot file is Pathway_ID
    df = df[df.iloc[:, 1].isin(human_ids)]

# 4) save as comma-separated
df.to_csv(dst, sep=",", index=False)
print(f"✅ Saved {len(df):,} Homo sapiens rows to {dst.name}")