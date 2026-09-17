"""
Ingest DrugBank helper CSVs into the normalized schema.

Inputs (under Datasets/DrugBank/):
  - all.csv
  - pharmacologically_active.csv
  - drug links.csv
  - drugbank vocabulary.csv

Outputs (under Datasets/normalized/):
  - drugs.csv                 (may be updated with preferred_name / inchi_key)
  - drug_synonym.csv
  - drug_xref.csv
  - drug_targets.csv

Idempotent & de-duplicated. Safe to re-run.
"""

import os
import re
from pathlib import Path
import pandas as pd

# ------------------------- config -------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
DBK  = DATA / "DrugBank"
OUT  = DATA / "normalized"

OUT.mkdir(parents=True, exist_ok=True)

PATH_DRUGS       = OUT / "drugs.csv"
PATH_SYNONYM     = OUT / "drug_synonym.csv"
PATH_XREF        = OUT / "drug_xref.csv"
PATH_TARGETS     = OUT / "drug_targets.csv"

# ------------------------- helpers ------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # keep_default_na=False -> empty strings stay empty, not NaN
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def write_csv(df: pd.DataFrame, path: Path, subset=None):
    """Append with de-dup; create file with header if missing."""
    if df is None or df.empty:
        return
    if path.exists():
        base = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = pd.concat([base, df], ignore_index=True)
    if subset:
        df = df.drop_duplicates(subset=subset)
    df.to_csv(path, index=False)

def ensure_file(path: Path, columns):
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def split_multi(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    # split on ; | , but keep tokens non-empty
    return [t.strip() for t in re.split(r"[;|,]\s*", s) if t.strip()]

def pick_col(df: pd.DataFrame, *names):
    """Find the first existing column by loose case-insensitive match."""
    if df.empty:
        return None
    norm = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n is None:
            continue
        k = n.lower().strip()
        if k in norm:
            return norm[k]
        # relaxed contains match (e.g., "DrugBank ID" vs "Drugbank Id")
        for nk, orig in norm.items():
            if k == nk or k in nk:
                return orig
    return None

def log(msg):
    print(f"[ingest] {msg}")

# ---------------------- ensure headers --------------------
ensure_file(PATH_DRUGS,   ["drug_id","preferred_name","type","approval_status","inchi_key","smiles"])
ensure_file(PATH_SYNONYM, ["drug_id","synonym"])
ensure_file(PATH_XREF,    ["drug_id","xref_db","xref_id"])
ensure_file(PATH_TARGETS, ["drug_id","uniprot_id","target_name","action","source_db"])

# =========================================================
# 1) all.csv + pharmacologically_active.csv  -> drug_targets.csv
# =========================================================
targets_frames = []
for fname in ["all.csv", "pharmacologically_active.csv"]:
    fpath = DBK / fname
    df = read_csv(fpath)
    if df.empty:
        log(f"{fname} not found or empty; skip.")
        continue

    col_drugids = pick_col(df, "Drug IDs", "DrugIDs", "Drug Ids")
    col_uniprot = pick_col(df, "UniProt ID", "Uniprot ID")
    col_tname   = pick_col(df, "Name")

    if not (col_drugids and col_uniprot):
        log(f"{fname}: missing required columns; skip.")
        continue

    rows = []
    for _, r in df.iterrows():
        uniprot = (r.get(col_uniprot, "") or "").strip()
        if not uniprot:
            continue
        tname = (r.get(col_tname, "") or "").strip()
        for dbid in split_multi(r.get(col_drugids, "")):
            # accept only canonical DrugBank IDs
            if not dbid.startswith("DB"):
                continue
            rows.append((dbid, uniprot, tname, "", "DrugBank-CSV"))

    if rows:
        tdf = pd.DataFrame(rows, columns=["drug_id","uniprot_id","target_name","action","source_db"])
        targets_frames.append(tdf)
        log(f"{fname}: +{len(tdf)} target rows")

if targets_frames:
    targets_all = pd.concat(targets_frames, ignore_index=True)
    write_csv(targets_all, PATH_TARGETS, subset=["drug_id","uniprot_id","target_name","action","source_db"])

# =========================================================
# 2) drug links.csv -> drug_xref.csv
# =========================================================
links = read_csv(DBK / "drug links.csv")
if links.empty:
    log("drug links.csv not found or empty; skip xrefs.")
else:
    id_col = pick_col(links, "DrugBank ID", "Drugbank ID")
    if not id_col:
        log("drug links.csv: no DrugBank ID column; skip xrefs.")
    else:
        # map link columns to xref_db label
        mapping = {
            "KEGG Compound ID": "KEGG.Compound",
            "KEGG Drug ID": "KEGG.Drug",
            "PubChem Compound ID": "PubChem",
            "PubChem Substance ID": "PubChem.SID",
            "ChEBI ID": "ChEBI",
            "PharmGKB ID": "PharmGKB",
            "ChemSpider ID": "ChemSpider",
            "BindingDB ID": "BindingDB",
            "TTD ID": "TTD",
            "Wikipedia ID": "Wikipedia",
            "RxList Link": "RxList",
            "Drugs.com Link": "Drugs.com",
            "HET ID": "PDB.HET",
        }
        rows = []
        for _, r in links.iterrows():
            dbid = (r.get(id_col, "") or "").strip()
            if not dbid:
                continue
            for col_name, db_tag in mapping.items():
                src_col = pick_col(links, col_name)
                if not src_col:
                    continue
                val = r.get(src_col, "")
                for v in split_multi(val):
                    rows.append((dbid, db_tag, v))
        if rows:
            xdf = pd.DataFrame(rows, columns=["drug_id","xref_db","xref_id"])
            write_csv(xdf, PATH_XREF, subset=["drug_id","xref_db","xref_id"])
            log(f"drug links.csv: +{len(xdf)} xref rows")

# =========================================================
# 3) drugbank vocabulary.csv -> synonyms (+ patch drugs.csv)
# =========================================================
vocab = read_csv(DBK / "drugbank vocabulary.csv")
if vocab.empty:
    log("drugbank vocabulary.csv not found or empty; skip synonyms/patch.")
else:
    col_id     = pick_col(vocab, "DrugBank ID", "Drugbank ID")
    col_syn    = pick_col(vocab, "Synonyms", "Synonym")
    col_common = pick_col(vocab, "Common name", "Common Name")
    col_inchi  = pick_col(vocab, "Standard InChI Key", "InChIKey", "InChI Key")

    # 3a) synonyms
    if col_id and col_syn:
        syn_rows = []
        for _, r in vocab.iterrows():
            dbid = (r.get(col_id, "") or "").strip()
            if not dbid:
                continue
            for s in split_multi(r.get(col_syn, "")):
                syn_rows.append((dbid, s))
        if syn_rows:
            sdf = pd.DataFrame(syn_rows, columns=["drug_id","synonym"])
            write_csv(sdf, PATH_SYNONYM, subset=["drug_id","synonym"])
            log(f"vocabulary: +{len(sdf)} synonyms")

    # 3b) patch drugs.csv (preferred_name / inchi_key if missing)
    if PATH_DRUGS.exists() and col_id:
        drugs = pd.read_csv(PATH_DRUGS, dtype=str, keep_default_na=False)
        patch = pd.DataFrame({"drug_id": vocab[col_id].astype(str).str.strip()})
        if col_common:
            patch["preferred_name__new"] = vocab[col_common].astype(str).str.strip()
        else:
            patch["preferred_name__new"] = ""
        if col_inchi:
            patch["inchi_key__new"] = vocab[col_inchi].astype(str).str.strip()
        else:
            patch["inchi_key__new"] = ""

        merged = drugs.merge(patch, on="drug_id", how="left")
        # fill preferred_name only if currently empty and new is non-empty
        need_name = (merged["preferred_name"].eq("") | merged["preferred_name"].isna()) & merged["preferred_name__new"].ne("")
        merged.loc[need_name, "preferred_name"] = merged.loc[need_name, "preferred_name__new"]
        # fill inchi_key only if currently empty
        need_inchi = (merged["inchi_key"].eq("") | merged["inchi_key"].isna()) & merged["inchi_key__new"].ne("")
        merged.loc[need_inchi, "inchi_key"] = merged.loc[need_inchi, "inchi_key__new"]

        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__new")])
        merged.to_csv(PATH_DRUGS, index=False)
        log("vocabulary: drugs.csv patched (preferred_name/inchi_key where missing)")

log("Done. Outputs in Datasets/normalized/")