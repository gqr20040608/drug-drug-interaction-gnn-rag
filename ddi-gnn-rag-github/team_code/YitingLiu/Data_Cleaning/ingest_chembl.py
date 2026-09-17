from pathlib import Path
import sqlite3
import pandas as pd

# -------------------- paths --------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
OUT  = DATA / "normalized"

CH_EMBL_DB = DATA / "ChEMBL" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"

P_DRUGS   = OUT / "drugs.csv"
P_XREF    = OUT / "drug_xref.csv"
P_TARGETS = OUT / "drug_targets.csv"

# ------------------- helpers -------------------
def log(msg): print(f"[chembl] {msg}")

def ensure(path: Path, cols):
    if not path.exists():
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def write_csv(df: pd.DataFrame, path: Path, subset=None):
    if df is None or df.empty: return
    if path.exists():
        base = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = pd.concat([base, df], ignore_index=True)
    if subset:
        df = df.drop_duplicates(subset=subset)
    df.to_csv(path, index=False)

def batched(seq, n=800):
    L = list(seq)
    for i in range(0, len(L), n):
        yield L[i:i+n]

def first_nonempty(series):
    for v in series:
        if isinstance(v, str) and v.strip():
            return v
    return ""

# ---------------- bootstrap --------------------
OUT.mkdir(parents=True, exist_ok=True)
ensure(P_DRUGS,   ["drug_id","preferred_name","type","approval_status","inchi_key","smiles"])
ensure(P_XREF,    ["drug_id","xref_db","xref_id"])
ensure(P_TARGETS, ["drug_id","uniprot_id","target_name","action","source_db"])

if not CH_EMBL_DB.exists():
    raise FileNotFoundError(f"ChEMBL SQLite not found: {CH_EMBL_DB}")

# load normalized
drugs = read_csv(P_DRUGS)
xrefs = read_csv(P_XREF)

# 清理历史 "None"
if not drugs.empty:
    cnt = (drugs.eq("None")).sum().sum()
    if cnt:
        drugs = drugs.fillna("").replace("None", "")
        log(f"cleaned 'None' literals in drugs.csv: {cnt} cells")

# 当前全部 drug_id（只对这些尝试补齐）
all_drug_ids = drugs["drug_id"].dropna().unique().tolist()

# 取 xref 中与 ChEMBL 相关的行
mask_chembl = xrefs["xref_db"].str.lower().str.contains("chembl", na=False)
chembl_xref = xrefs[mask_chembl][["drug_id","xref_id"]].drop_duplicates()
chembl_xref.rename(columns={"xref_id":"chembl_mol_id"}, inplace=True)

# open DB
con = sqlite3.connect(str(CH_EMBL_DB))
con.row_factory = sqlite3.Row

# =========================================================
# A) 结构补齐：两条路合并（A: xref→chembl; B: chembl compound_records via DRUGBANK）
# =========================================================
patched_inchi = patched_smiles = patched_names = 0

# --- A: 通过 drug_xref 的 ChEMBL ID ---
frames_A = []
if not chembl_xref.empty:
    chembl_ids = chembl_xref["chembl_mol_id"].dropna().unique().tolist()
    for chunk in batched(chembl_ids, n=600):
        ph = ",".join(["?"]*len(chunk))
        q = f"""
        SELECT
            md.chembl_id          AS chembl_mol_id,
            md.pref_name          AS pref_name,
            cs.canonical_smiles   AS smiles,
            cs.standard_inchi_key AS inchi_key
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs
          ON md.molregno = cs.molregno
        WHERE md.chembl_id IN ({ph})
          AND (cs.canonical_smiles IS NOT NULL OR cs.standard_inchi_key IS NOT NULL)
        """
        df = pd.read_sql_query(q, con, params=chunk, dtype=str).fillna("")
        frames_A.append(df)

patch_A = (
    chembl_xref.merge(pd.concat(frames_A, ignore_index=True) if frames_A else pd.DataFrame(),
                      on="chembl_mol_id", how="inner")
    .drop(columns=["chembl_mol_id"], errors="ignore")
)

# --- B: 用 ChEMBL 的 compound_records / source 由 DrugBank ID 直连到结构 ---
frames_B = []
if all_drug_ids:
    for chunk in batched(all_drug_ids, n=600):
        ph = ",".join(["?"]*len(chunk))
        q = f"""
        SELECT
            cr.src_compound_id    AS drug_id,
            md.chembl_id          AS chembl_mol_id,
            md.pref_name          AS pref_name,
            cs.canonical_smiles   AS smiles,
            cs.standard_inchi_key AS inchi_key
        FROM compound_records cr
        JOIN source s              ON cr.src_id = s.src_id
        JOIN molecule_dictionary md ON cr.molregno = md.molregno
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE s.src_short_name = 'DRUGBANK'
          AND cr.src_compound_id IN ({ph})
          AND (cs.canonical_smiles IS NOT NULL OR cs.standard_inchi_key IS NOT NULL)
        """
        df = pd.read_sql_query(q, con, params=chunk, dtype=str).fillna("")
        frames_B.append(df)

patch_B = pd.concat(frames_B, ignore_index=True) if frames_B else pd.DataFrame()

# --- 合并两路补丁 → 每个 drug_id 取第一个非空 ---
patch_all = pd.concat([
    patch_A[["drug_id","pref_name","smiles","inchi_key"]].rename(columns={"pref_name":"preferred_name"}),
    patch_B[["drug_id","pref_name","smiles","inchi_key"]].rename(columns={"pref_name":"preferred_name"})
], ignore_index=True)

if not patch_all.empty:
    per_drug = (
        patch_all.groupby("drug_id", as_index=False)
                 .agg({"preferred_name": first_nonempty,
                       "smiles": first_nonempty,
                       "inchi_key": first_nonempty})
    )
else:
    per_drug = pd.DataFrame(columns=["drug_id","preferred_name","smiles","inchi_key"])

# --- merge 回 drugs，只在原值为空且新值非空时补 ---
merged = drugs.merge(per_drug, on="drug_id", how="left", suffixes=("", "__new")).fillna("")

need_name = (merged["preferred_name"].eq("")) & merged["preferred_name__new"].ne("")
need_smi  = (merged["smiles"].eq(""))         & merged["smiles__new"].ne("")
need_ik   = (merged["inchi_key"].eq(""))      & merged["inchi_key__new"].ne("")

patched_names  = int(need_name.sum())
patched_smiles = int(need_smi.sum())
patched_inchi  = int(need_ik.sum())

merged.loc[need_name, "preferred_name"] = merged.loc[need_name, "preferred_name__new"]
merged.loc[need_smi,  "smiles"]         = merged.loc[need_smi,  "smiles__new"]
merged.loc[need_ik,   "inchi_key"]      = merged.loc[need_ik,   "inchi_key__new"]

merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__new")], errors="ignore")
merged = merged.replace("None", "").fillna("")
merged.to_csv(P_DRUGS, index=False)

log(f"structures patched: +{patched_inchi} InChIKeys, +{patched_smiles} SMILES, +{patched_names} names")

# =========================================================
# B) drug→target（dm.molregno + tid → UniProt accession）
# =========================================================
# 通过 molecule_dictionary 将我们的 drug_id 映射到 molregno
mol_map_frames = []
if not per_drug.empty:
    # 我们有 drug_id 列表，不依赖 xref；用 compound_records 的映射覆盖更多
    for chunk in batched(all_drug_ids, n=600):
        ph = ",".join(["?"]*len(chunk))
        q = f"""
        SELECT
            cr.src_compound_id AS drug_id,
            md.molregno       AS molregno
        FROM compound_records cr
        JOIN source s              ON cr.src_id = s.src_id
        JOIN molecule_dictionary md ON cr.molregno = md.molregno
        WHERE s.src_short_name = 'DRUGBANK'
          AND cr.src_compound_id IN ({ph})
        """
        df = pd.read_sql_query(q, con, params=chunk, dtype=str).fillna("")
        mol_map_frames.append(df)

mol_join = pd.concat(mol_map_frames, ignore_index=True) if mol_map_frames else pd.DataFrame()
molregnos = mol_join["molregno"].dropna().unique().tolist()

edge_frames = []
if molregnos:
    for chunk in batched(molregnos, n=600):
        ph = ",".join(["?"]*len(chunk))
        q = f"""
        SELECT
            dm.molregno           AS molregno,
            td.pref_name          AS target_name,
            cs.accession          AS uniprot_id,
            COALESCE(dm.action_type, dm.mechanism_of_action) AS action
        FROM drug_mechanism dm
        LEFT JOIN target_dictionary td ON dm.tid = td.tid
        LEFT JOIN target_components tc ON td.tid = tc.tid
        LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
        WHERE dm.molregno IN ({ph})
        """
        df = pd.read_sql_query(q, con, params=chunk, dtype=str).fillna("")
        edge_frames.append(df)

edges_raw = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()

if edges_raw.empty:
    log("targets appended: +0 (no mechanisms)")
else:
    edges_raw = edges_raw[(edges_raw["uniprot_id"].ne(""))]
    edges = edges_raw.merge(mol_join, on="molregno", how="inner")[["drug_id","uniprot_id","target_name","action"]]
    edges["source_db"] = "ChEMBL"
    write_csv(edges, P_TARGETS, subset=["drug_id","uniprot_id","target_name","action","source_db"])
    log(f"targets appended: +{len(edges)} edges (ChEMBL)")

con.close()
log("Done. Outputs in Datasets/normalized/")