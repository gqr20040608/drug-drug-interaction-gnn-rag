from pathlib import Path
import pandas as pd
import sys

# ---------- 路径设置（关键修复点） ----------
ROOT = Path(__file__).resolve().parent            # 脚本所在目录：DS_340W_PROJECT/
SRC  = ROOT / "Datasets" / "DrugCentral" / "extracted"
OUT  = ROOT / "Datasets" / "normalized"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[info] SRC = {SRC}")
print(f"[info] OUT = {OUT}")

# 先做存在性检查，避免一头雾水
need = ["structures.tsv", "structures.smiles.tsv", "synonyms.tsv", "drug.target.interaction.tsv"]
missing = [p for p in need if not (SRC / p).exists()]
if missing:
    print("[error] 以下文件未找到：")
    for m in missing:
        print("   -", SRC / m)
    sys.exit(1)

# 读取小助手（兼容 \N、制表符、奇怪的转义）
def read_tsv(path):
    return pd.read_csv(
        path, sep="\t", engine="python",
        na_values=["\\N", ""], keep_default_na=True,
        dtype=str  # 先全部按字符串读，避免类型/小数导致的问题
    )

# ---------- 1) structures + smiles 组主视图 ----------
struct  = read_tsv(SRC / "structures.tsv")
smiles  = read_tsv(SRC / "structures.smiles.tsv")

# 标准列名
# structures.tsv: id, name, cas_reg_no, …（其他列保留不影响）
struct = struct.rename(columns={"id": "drugcentral_id", "name": "preferred_name"})

# structures.smiles.tsv: ID, SMILES, InChI, InChIKey
smiles = smiles.rename(columns={
    "ID": "drugcentral_id", "SMILES": "smiles",
    "InChI": "inchi", "InChIKey": "inchikey"
})

drugs = struct.merge(smiles, on="drugcentral_id", how="left")

# 只保留我们要用到的核心列
keep_cols = ["drugcentral_id", "preferred_name", "smiles", "inchikey", "cas_reg_no"]
for c in keep_cols:
    if c not in drugs.columns:
        drugs[c] = pd.NA
drugs = drugs[keep_cols]
drugs["source"] = "drugcentral"
drugs.to_csv(OUT / "drugcentral_drugs.csv", index=False)
print("[ok] wrote:", OUT / "drugcentral_drugs.csv", f"rows={len(drugs)}")

# ---------- 2) synonyms ----------
syn = read_tsv(SRC / "synonyms.tsv")
# synonyms.tsv: id, name(同义词), preferred_name, …
syn = syn.rename(columns={"id": "drugcentral_id", "name": "synonym"})
need_cols = ["drugcentral_id", "synonym", "preferred_name"]
for c in need_cols:
    if c not in syn.columns:
        syn[c] = pd.NA
syn = syn[need_cols]
syn["source"] = "drugcentral"
syn.to_csv(OUT / "drugcentral_synonym.csv", index=False)
print("[ok] wrote:", OUT / "drugcentral_synonym.csv", f"rows={len(syn)}")

# ---------- 3) targets ----------
tgt = read_tsv(SRC / "drug.target.interaction.tsv")
# drug.target.interaction.tsv: STRUCT_ID, ACCESSION, TARGET_NAME, TARGET_CLASS, ACT_TYPE, ACT_VALUE, …
tgt = tgt.rename(columns={
    "STRUCT_ID": "drugcentral_id",
    "ACCESSION": "uniprot",
    "TARGET_NAME": "target_name",
    "TARGET_CLASS": "target_class",
    "ACT_TYPE": "act_type",
    "ACT_VALUE": "act_value"
})
need_cols = ["drugcentral_id", "uniprot", "target_name", "target_class", "act_type", "act_value"]
for c in need_cols:
    if c not in tgt.columns:
        tgt[c] = pd.NA
tgt = tgt[need_cols]
tgt["source"] = "drugcentral"
tgt.to_csv(OUT / "drugcentral_targets.csv", index=False)
print("[ok] wrote:", OUT / "drugcentral_targets.csv", f"rows={len(tgt)}")

print("✅ DrugCentral extraction complete.")