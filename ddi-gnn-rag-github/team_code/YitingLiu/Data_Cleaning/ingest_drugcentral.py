#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest DrugCentral into normalized schema with NAME/SYNONYM matching.

Inputs (Datasets/DrugCentral/):
  - structures.smiles.tsv
  - drug.target.interaction.tsv  (columns: DRUG_NAME, TARGET_NAME, ACCESSION, MOA, ...)

Outputs (Datasets/normalized/):
  - drugs.csv          (patch inchi_key/smiles/preferred_name when missing)
  - drug_targets.csv   (append with source_db='DrugCentral')

Matching priority for interactions:
  1) exact match on DRUG_NAME vs drugs.preferred_name
  2) exact match on DRUG_NAME vs drug_synonym.synonym
  (all casefolded & stripped; punctuation collapsed)

Idempotent and de-duplicated.
"""

from pathlib import Path
import pandas as pd
import re
import string

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
DC   = DATA / "DrugCentral"
OUT  = DATA / "normalized"

P_DRUGS   = OUT / "drugs.csv"
P_TARGETS = OUT / "drug_targets.csv"
P_SYNS    = OUT / "drug_synonym.csv"

def read_table(path: Path):
    if not path.exists(): 
        return pd.DataFrame()
    sep = "\t" if path.suffix.lower()==".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.strip('"').str.strip()
    return df

def write_csv(df: pd.DataFrame, path: Path, subset=None):
    if df is None or df.empty: return
    if path.exists():
        base = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = pd.concat([base, df], ignore_index=True)
    if subset: df = df.drop_duplicates(subset=subset)
    df.to_csv(path, index=False)

def ensure(path: Path, cols):
    if not path.exists():
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def log(m): print(f"[drugcentral] {m}")

# ---------- text normalization for name matching ----------
_PUNC = str.maketrans({c: " " for c in string.punctuation})
def norm_name(x: str) -> str:
    """lowercase; remove punctuation; collapse spaces."""
    if not isinstance(x, str): return ""
    x = x.casefold().translate(_PUNC)
    x = re.sub(r"\s+", " ", x).strip()
    return x

# ---------- ensure outputs ----------
ensure(P_DRUGS,   ["drug_id","preferred_name","type","approval_status","inchi_key","smiles"])
ensure(P_TARGETS, ["drug_id","uniprot_id","target_name","action","source_db"])
ensure(P_SYNS,    ["drug_id","synonym"])

# ---------- load existing drugs & synonyms ----------
drugs = read_table(P_DRUGS)
syns  = read_table(P_SYNS)

# map for inchikey -> drug_id (structure patch 用)
ik2id = {}
for _, r in drugs.iterrows():
    ik = (r.get("inchi_key","") or "").strip()
    did = (r.get("drug_id","") or "").strip()
    if ik and did and ik not in ik2id:
        ik2id[ik] = did

# build name -> drug_id map (preferred_name + synonyms)
name2id = {}
# preferred names
for _, r in drugs.iterrows():
    did = (r.get("drug_id","") or "").strip()
    nm  = norm_name(r.get("preferred_name",""))
    if did and nm and nm not in name2id:
        name2id[nm] = did
# synonyms
for _, r in syns.iterrows():
    did = (r.get("drug_id","") or "").strip()
    sn  = norm_name(r.get("synonym",""))
    if did and sn and sn not in name2id:
        name2id[sn] = did

# ---------- 1) structures.smiles.tsv -> patch drugs.csv ----------
sdf = read_table(DC / "structures.smiles.tsv")
if sdf.empty:
    log("structures.smiles.tsv missing/empty; skip structure patch.")
else:
    col_inchi  = next((c for c in sdf.columns if c.lower().startswith("inchikey")), None)
    col_smiles = next((c for c in sdf.columns if "smiles" in c.lower()), None)
    col_name   = next((c for c in sdf.columns if c.lower() in ("drug_name","name","preferred_name")), None)
    col_dbid   = next((c for c in sdf.columns if "drugbank" in c.lower() and "id" in c.lower()), None)

    if not col_smiles or not (col_inchi or col_dbid or col_name):
        log("structures.smiles.tsv: missing key cols; skip.")
    else:
        patch_rows = []
        for _, r in sdf.iterrows():
            ik  = (r.get(col_inchi,"") or "").strip() if col_inchi else ""
            smi = (r.get(col_smiles,"") or "").strip()
            nm  = (r.get(col_name,"") or "").strip() if col_name else ""
            dbid= (r.get(col_dbid,"") or "").strip() if col_dbid else ""

            if not smi: 
                continue
            drug_id = ""
            if dbid.startswith("DB"):
                drug_id = dbid
            elif ik and ik in ik2id:
                drug_id = ik2id[ik]
            elif nm:
                drug_id = name2id.get(norm_name(nm), "")

            if not drug_id:
                continue
            patch_rows.append((drug_id, ik, smi, nm))

        if patch_rows:
            patch = pd.DataFrame(patch_rows, columns=["drug_id","inchi_key__new","smiles__new","preferred_name__new"])
            merged = drugs.merge(patch, on="drug_id", how="left")
            need_ik  = (merged["inchi_key"].eq("") | merged["inchi_key"].isna()) & merged["inchi_key__new"].notna() & merged["inchi_key__new"].ne("")
            need_smi = (merged["smiles"].eq("") | merged["smiles"].isna()) & merged["smiles__new"].notna() & merged["smiles__new"].ne("")
            need_nm  = (merged["preferred_name"].eq("") | merged["preferred_name"].isna()) & merged["preferred_name__new"].notna() & merged["preferred_name__new"].ne("")
            merged.loc[need_ik,  "inchi_key"]      = merged.loc[need_ik,  "inchi_key__new"]
            merged.loc[need_smi, "smiles"]         = merged.loc[need_smi, "smiles__new"]
            merged.loc[need_nm,  "preferred_name"] = merged.loc[need_nm,  "preferred_name__new"]
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__new")])
            merged.to_csv(P_DRUGS, index=False)
            log(f"structures.smiles.tsv: patched drugs.csv ({int(need_ik.sum())} inchi_key, {int(need_smi.sum())} smiles, {int(need_nm.sum())} names)")
            drugs = merged  # refresh

# ---------- 2) drug.target.interaction.tsv -> append to drug_targets.csv ----------
idf = read_table(DC / "drug.target.interaction.tsv")
if idf.empty:
    log("drug.target.interaction.tsv missing/empty; skip targets.")
else:
    # real column names per your file
    c_drugname = "DRUG_NAME" if "DRUG_NAME" in idf.columns else None
    c_uniprot  = "ACCESSION" if "ACCESSION" in idf.columns else None
    c_tname    = "TARGET_NAME" if "TARGET_NAME" in idf.columns else None
    c_action   = "MOA" if "MOA" in idf.columns else ("RELATION" if "RELATION" in idf.columns else None)

    if not (c_drugname and c_uniprot):
        log("interaction: required columns not present; stop.")
    else:
        rows, miss = [], 0
        for _, r in idf.iterrows():
            nm = norm_name(r.get(c_drugname,""))
            if not nm: 
                continue
            drug_id = name2id.get(nm, "")
            if not drug_id:
                miss += 1
                continue
            unip = (r.get(c_uniprot,"") or "").strip()
            if not unip:
                continue
            tname = (r.get(c_tname,"") or "").strip() if c_tname else ""
            act   = (r.get(c_action,"") or "").strip() if c_action else ""
            rows.append((drug_id, unip, tname, act, "DrugCentral"))

        if rows:
            tdf = pd.DataFrame(rows, columns=["drug_id","uniprot_id","target_name","action","source_db"])
            write_csv(tdf, P_TARGETS, subset=["drug_id","uniprot_id","target_name","action","source_db"])
            log(f"interaction: +{len(tdf)} target rows (name-matched), missed {miss}")
        else:
            log(f"interaction: 0 rows matched; missed {miss} (likely names not aligning)")

log("Done. Outputs in Datasets/normalized/")