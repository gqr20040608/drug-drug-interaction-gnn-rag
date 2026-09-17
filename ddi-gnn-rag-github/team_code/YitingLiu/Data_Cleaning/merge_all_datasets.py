from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
NORM = ROOT / "Datasets" / "normalized"
OUT  = NORM / "merged"
RX   = ROOT / "Datasets" / "Reactome"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path, **kw):
    return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["\\N", ""], **kw)

# 1) ------- 主药物清单：以 DrugBank 为主 -----------
db_drugs = read_csv(NORM / "drugs_patched2.csv")  # 你已有的更完整版本
db_drugs = db_drugs.rename(columns={
    "drug_id":"drug_id",
    "preferred_name":"name_db",
    "inchi_key":"inchikey_db",
    "smiles":"smiles_db"
})
db_drugs = db_drugs[["drug_id","name_db","inchikey_db","smiles_db"]].drop_duplicates()

# 2) ------- DrugCentral 补丁 ------------------------
dc = read_csv(NORM / "drugcentral_drugs.csv") \
        .rename(columns={"preferred_name":"name_dc"})[
            ["drugcentral_id","name_dc","smiles","inchikey"]
        ]
dc.columns = ["drugcentral_id","name_dc","smiles_dc","inchikey_dc"]

# 对齐策略：先用 InChIKey，再用名称（宽松）
m = db_drugs.merge(dc, left_on="inchikey_db", right_on="inchikey_dc", how="left")
# 名称兜底（小写、去空格）
if m["drugcentral_id"].isna().any():
    db2 = db_drugs.assign(_name_l=db_drugs["name_db"].str.lower().str.strip())
    dc2 = dc.assign(_name_l=dc["name_dc"].str.lower().str.strip())
    m2 = db2.merge(dc2[["drugcentral_id","_name_l","smiles_dc","inchikey_dc"]], on="_name_l", how="left")
    m = m.combine_first(m2)

# 选择最终 name/smiles/inchikey（只补不改）
def fill(a, b): return a.where(a.notna() & (a.astype(str)!=""), b)
m["name"]     = fill(m["name_db"], m["name_dc"])
m["smiles"]   = fill(m["smiles_db"], m["smiles_dc"])
m["inchikey"] = fill(m["inchikey_db"], m["inchikey_dc"])

drugs_master = m[["drug_id","name","smiles","inchikey","drugcentral_id"]].drop_duplicates()
drugs_master.to_csv(OUT / "drugs_master.csv", index=False)

# 3) ------- ID 映射表（DrugBank + DrugCentral + ChEMBL from xref） -------
maps = []
# DrugBank 自身
maps.append(drugs_master[["drug_id"]].assign(source="drugbank", source_id=lambda d:d["drug_id"]))
# DrugCentral
maps.append(drugs_master.dropna(subset=["drugcentral_id"])[["drug_id","drugcentral_id"]]
            .rename(columns={"drugcentral_id":"source_id"})
            .assign(source="drugcentral"))
# ChEMBL from xref
xref = read_csv(NORM / "drug_xref.csv")
chembl = xref[xref["xref_db"].str.contains("ChEMBL", case=False, na=False)]
chembl = chembl.rename(columns={"xref_id":"source_id"})[["drug_id","source_id"]].drop_duplicates().assign(source="chembl")
maps.append(chembl)
drug_id_map = pd.concat(maps, ignore_index=True).drop_duplicates()
drug_id_map.to_csv(OUT / "drug_id_map.csv", index=False)

# 4) ------- 边：drug-target（合并 DrugBank + DrugCentral） -------
db_tgt = read_csv(NORM / "drug_targets.csv").rename(columns={"uniprot_id":"uniprot_id"})
db_tgt = db_tgt[["drug_id","uniprot_id"]].assign(source="drugbank").dropna()

dc_tgt = read_csv(NORM / "drugcentral_targets.csv").rename(columns={"uniprot":"uniprot_id"})
# 将 dc 的 drugcentral_id 映射回 drug_id
dc_map = drugs_master.dropna(subset=["drugcentral_id"])[["drugcentral_id","drug_id"]]
dc_tgt2 = dc_tgt.merge(dc_map, on="drugcentral_id", how="inner")[["drug_id","uniprot_id"]].assign(source="drugcentral")

edges_drug_target = pd.concat([db_tgt, dc_tgt2], ignore_index=True).drop_duplicates()
edges_drug_target.to_csv(OUT / "edges_drug_target.csv", index=False)

# 5) ------- Reactome：drug → pathway（基于 target 的 uniprot 映射） -------
# 需要 UniProt→Pathway 映射文件
up2rx = pd.read_csv(RX / "UniProt2Reactome_All_Levels.txt", sep="\t", header=None, dtype=str,
                    names=["uniprot_id","pathway_id","pathway_name","evidence","species","url","_extra1","_extra2"],
                    usecols=["uniprot_id","pathway_id","pathway_name","species"])
# 只保留人类（可按需放宽）
up2rx = up2rx[up2rx["species"].str.contains("Homo sapiens", na=False)]

edges_drug_pathway = edges_drug_target[["drug_id","uniprot_id"]].merge(
    up2rx[["uniprot_id","pathway_id"]], on="uniprot_id", how="inner"
)[["drug_id","pathway_id"]].drop_duplicates().assign(source="reactome")

edges_drug_pathway.to_csv(OUT / "edges_drug_pathway.csv", index=False)

# 6) ------- 特征：ATC / 同义词 / 理化性质 -------------------------------
atc = read_csv(NORM / "drug_atc.csv")[["drug_id","atc_code"]].drop_duplicates()
atc.to_csv(OUT / "feat_atc.csv", index=False)

syn_db = read_csv(NORM / "drug_synonym.csv")[["drug_id","synonym"]].drop_duplicates()
# DrugCentral 同义词映射回 drug_id
dc_syn = read_csv(NORM / "drugcentral_synonym.csv")[["drugcentral_id","synonym"]]
dc_syn = dc_syn.merge(dc_map, on="drugcentral_id", how="inner")[["drug_id","synonym"]]
feat_synonym = pd.concat([syn_db, dc_syn], ignore_index=True).drop_duplicates()
feat_synonym.to_csv(OUT / "feat_synonym.csv", index=False)

props = read_csv(NORM / "drug_properties.csv")[["drug_id","kind","value"]]
props.to_csv(OUT / "feat_properties.csv", index=False)

print("✅ Done. Merged outputs written to:", OUT)