#!/usr/bin/env python3
import os, argparse, pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

def rdkit_inchikey_from_smiles(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return inchi.MolToInchiKey(m)
    except Exception:
        return None

def pick_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    return None

def norm_name(x: str):
    return "".join(ch.lower() for ch in (x or "").strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", default="Datasets/normalized/drugs.csv")
    ap.add_argument("--xref",  default="Datasets/normalized/drug_xref.csv")
    ap.add_argument("--dc_dir", default="Datasets/DrugCentral/extracted")
    ap.add_argument("--out",   default="Datasets/normalized/drugs_patched.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.drugs, dtype=str)
    for c in ["smiles","inchi_key"]:
        if c not in df.columns:
            df[c] = None

    before_smi = df["smiles"].isna().sum()
    before_ik  = df["inchi_key"].isna().sum()

    need_mask = df["smiles"].isna() & df["inchi_key"].isna()
    if not need_mask.any():
        print("[info] nothing to patch")
        df.to_csv(args.out, index=False)
        print(f"[done] wrote {args.out}")
        return

    # ---- DrugCentral structures ----
    dc_struct_path = None
    for g in ["structures.tsv","structures.smiles.tsv","drug_structures.tsv"]:
        p = os.path.join(args.dc_dir, g)
        if os.path.exists(p):
            dc_struct_path = p
            break
    if dc_struct_path is None:
        raise RuntimeError(f"Could not find structures TSV in {args.dc_dir}")

    dc_struct = pd.read_csv(dc_struct_path, sep="\t", dtype=str)
    id_col  = pick_col(dc_struct.columns, ["drug_id","id","struct_id","structure_id","compound_id"])
    smi_col = pick_col(dc_struct.columns, ["smiles","canonical_smiles","isomeric_smiles"])
    ik_col  = pick_col(dc_struct.columns, ["inchikey","inchi_key","standard_inchikey","standard_inchi_key"])
    if id_col is None or smi_col is None:
        raise RuntimeError(f"Cannot identify ID/SMILES columns in {dc_struct_path}. Columns: {list(dc_struct.columns)}")

    dc_struct = dc_struct.rename(columns={id_col:"dc_id", smi_col:"dc_smiles"})
    if ik_col:
        dc_struct = dc_struct.rename(columns={ik_col:"dc_inchikey"})
    else:
        dc_struct["dc_inchikey"] = None
    dc_struct = dc_struct.drop_duplicates("dc_id")

    # ---- Try xref mapping if any DrugCentral rows exist ----
    patched_from_xref = 0
    dc_map = pd.DataFrame(columns=["drug_id","dc_id"])
    if os.path.exists(args.xref):
        xref = pd.read_csv(args.xref, dtype=str)
        if set(["drug_id","xref_db","xref_id"]).issubset(xref.columns):
            mask_dc = xref["xref_db"].astype(str).str.lower().str.contains("drugcentral")
            if mask_dc.any():
                dc_map = xref.loc[mask_dc, ["drug_id","xref_id"]].rename(columns={"xref_id":"dc_id"})
                dc_map["drug_id"] = dc_map["drug_id"].astype(str)

    patched = df.copy()
    if len(dc_map) > 0:
        xref_join = dc_map.merge(dc_struct, on="dc_id", how="left")
        rows = patched.index[need_mask]
        tmp = patched.loc[rows, ["drug_id"]].merge(
            xref_join[["drug_id","dc_smiles","dc_inchikey"]],
            on="drug_id", how="left"
        )
        patched.loc[rows, "smiles"]    = tmp["dc_smiles"].values
        patched.loc[rows, "inchi_key"] = tmp["dc_inchikey"].values
        patched_from_xref = tmp["dc_smiles"].notna().sum()

    # ---- Synonym-based fallback (exact normalized match on preferred_name) ----
    patched_from_syn  = 0
    computed_ik       = 0

    need_mask = patched["smiles"].isna() & patched["inchi_key"].isna()
    syn_path = os.path.join(args.dc_dir, "synonyms.tsv")
    if need_mask.any() and os.path.exists(syn_path):
        syn_df = pd.read_csv(syn_path, sep="\t", dtype=str)
        syn_id_col   = pick_col(syn_df.columns, ["drug_id","id","compound_id"])
        syn_name_col = pick_col(syn_df.columns, ["synonym","name","label"])
        if syn_id_col and syn_name_col:
            syn_df = syn_df.rename(columns={syn_id_col:"dc_id", syn_name_col:"dc_synonym"})
            syn_df["dc_syn_norm"] = syn_df["dc_synonym"].map(norm_name)

            # unique (unambiguous) synonym -> dc_id
            counts = syn_df.groupby("dc_syn_norm")["dc_id"].nunique().reset_index(name="n")
            syn_unique = syn_df.merge(counts[counts["n"]==1], on="dc_syn_norm")
            syn_unique = syn_unique[["dc_syn_norm","dc_id"]].drop_duplicates()

            need_names = (patched.loc[need_mask, ["drug_id","preferred_name"]]
                          .dropna()
                          .assign(name_norm=lambda d: d["preferred_name"].map(norm_name)))

            m = need_names.merge(syn_unique, left_on="name_norm", right_on="dc_syn_norm", how="left")
            m = m.merge(dc_struct, on="dc_id", how="left")

            # reduce to one suggestion per drug_id
            m = m.sort_values(["drug_id"]).drop_duplicates("drug_id", keep="first")

            # compute IK where missing but we got SMILES
            to_compute = m["dc_inchikey"].isna() & m["dc_smiles"].notna()
            if to_compute.any():
                m.loc[to_compute, "dc_inchikey"] = m.loc[to_compute, "dc_smiles"].apply(rdkit_inchikey_from_smiles)
                computed_ik += int(to_compute.sum())

            rows = patched.index[need_mask]
            tmp = patched.loc[rows, ["drug_id"]].merge(
                m[["drug_id","dc_smiles","dc_inchikey"]], on="drug_id", how="left"
            )
            patched.loc[rows, "smiles"]    = tmp["dc_smiles"].values
            patched.loc[rows, "inchi_key"] = tmp["dc_inchikey"].values
            patched_from_syn = tmp["dc_smiles"].notna().sum()

    # ---- Final: compute IK for any rows that now have SMILES but no IK ----
    has_smi_no_ik = patched["smiles"].notna() & patched["inchi_key"].isna()
    if has_smi_no_ik.any():
        patched.loc[has_smi_no_ik, "inchi_key"] = patched.loc[has_smi_no_ik, "smiles"].apply(rdkit_inchikey_from_smiles)
        computed_ik += int(has_smi_no_ik.sum())

    after_smi = patched["smiles"].isna().sum()
    after_ik  = patched["inchi_key"].isna().sum()

    print(f"[report] SMILES missing:   {before_smi} -> {after_smi}")
    print(f"[report] InChIKey missing: {before_ik} -> {after_ik}")
    print(f"[report] patched from DrugCentral xref:      {patched_from_xref}")
    print(f"[report] patched from DrugCentral synonyms:  {patched_from_syn}")
    print(f"[report] InChIKeys computed via RDKit:       {computed_ik}")

    patched.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()