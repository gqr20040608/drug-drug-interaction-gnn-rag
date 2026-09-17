
import os, sys, argparse
import pandas as pd

OUTDIR = "GNN/GNN_datasets"

def info(x): print(f"[INFO] {x}")
def warn(x): print(f"[WARN] {x}", file=sys.stderr)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_csv_header_or_none(path, names_if_none=None, **kw):
    """
    Try reading with header row; if that doesn't yield expected columns, read as headerless with provided names.
    """
    # First attempt: header=0
    try:
        df = pd.read_csv(path, **kw)
        return df, "header"
    except Exception as e:
        warn(f"Reading with header failed for {path}: {e}")

    # Second attempt: header=None
    if names_if_none is None:
        raise
    df = pd.read_csv(path, header=None, names=names_if_none, **{k:v for k,v in kw.items() if k!="header"})
    return df, "noheader"

# ---------- Standardizers ----------
def std_drug_nodes(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {
        "drug_id":  ["drug_id","DrugBankID","db_id","drugbank_id","id"],
        "inchikey": ["inchikey","InChIKey","InChIKEY","inchi_key"],
        "name":     ["name","drug_name","Name","INN"],
        "smiles":   ["smiles","SMILES"],
    }
    out = {}
    for k, cands in cmap.items():
        for c in cands:
            if c in df.columns:
                out[k] = df[c]
                break
        if k not in out:
            out[k] = "" if k!="drug_id" else None
    out = pd.DataFrame(out).dropna(subset=["drug_id"]).drop_duplicates(subset=["drug_id"]).reset_index(drop=True)
    info(f"Drug nodes: {out.shape}")
    return out[["drug_id","inchikey","name","smiles"]]

def std_protein_nodes(df: pd.DataFrame) -> pd.DataFrame:
    org = next((c for c in ["organism","Organism","species","taxonomy","taxon"] if c in df.columns), None)
    if org:
        df = df[df[org].astype(str).str.contains("Homo sapiens", case=False, na=False)]
    acc = next((c for c in ["uniprot_id","accession","Entry","Accession","UniProtKB-AC","UniProtKB"] if c in df.columns), None)
    if not acc: raise ValueError("Protein nodes need a UniProt accession column")
    gene = next((c for c in ["gene","Gene","gene_symbol","Gene Names","symbol"] if c in df.columns), None)
    pname = next((c for c in ["protein_name","Protein names","name","Name","Recommended name"] if c in df.columns), None)
    out = pd.DataFrame({
        "uniprot_id": df[acc].astype(str),
        "gene": df[gene].astype(str) if gene else "",
        "name": df[pname].astype(str) if pname else "",
    }).dropna(subset=["uniprot_id"]).drop_duplicates(subset=["uniprot_id"]).reset_index(drop=True)
    info(f"Protein nodes (H. sapiens): {out.shape}")
    return out[["uniprot_id","gene","name"]]

def std_pathway_nodes(df: pd.DataFrame) -> pd.DataFrame:
    # Expect at least reactome_id + name; if species exists, filter Homo sapiens
    id_col = next((c for c in ["reactome_id","Reactome_ID","pathway_id","id","stId","Identifier","Stable identifier"] if c in df.columns), None)
    name_col = next((c for c in ["name","Name","pathway_name","displayName","label","Label","Pathway name"] if c in df.columns), None)
    # Headerless fallback common in your file: ["reactome_id","name","species"]
    if id_col is None and name_col is None and df.shape[1] >= 2:
        df = df.rename(columns={df.columns[0]:"reactome_id", df.columns[1]:"name"})
        if df.shape[1] >= 3: df = df.rename(columns={df.columns[2]:"species"})
        id_col, name_col = "reactome_id", "name"

    # Filter species if present
    if "species" in df.columns:
        df = df[df["species"].astype(str).str.contains("Homo sapiens", case=False, na=False)]

    if id_col is None or name_col is None:
        raise ValueError("Pathway nodes require columns for reactome_id and name (headerless accepted).")

    out = pd.DataFrame({
        "reactome_id": df[id_col].astype(str),
        "name": df[name_col].astype(str),
        "level": 0,  # depth not provided; set 0 as default
    }).dropna(subset=["reactome_id"]).drop_duplicates(subset=["reactome_id"]).reset_index(drop=True)
    info(f"Pathway nodes: {out.shape}")
    return out[["reactome_id","name","level"]]

def std_edges_dt(df: pd.DataFrame) -> pd.DataFrame:
    dcol = next((c for c in ["drug_id","DrugBankID","db_id"] if c in df.columns), None)
    pcol = next((c for c in ["uniprot_id","target_uniprot","uniprot","accession"] if c in df.columns), None)
    if not (dcol and pcol): raise ValueError("drug_targets need columns: drug_id & uniprot_id")
    out = pd.DataFrame({
        "src_drug_id": df[dcol].astype(str),
        "dst_uniprot_id": df[pcol].astype(str)
    }).dropna().drop_duplicates()
    out["relation"] = "targets"
    info(f"edges drug-target: {out.shape}")
    return out[["src_drug_id","dst_uniprot_id","relation"]]

def std_edges_ppw(df: pd.DataFrame) -> pd.DataFrame:
    # Accept named columns or first two columns (headerless)
    up = next((c for c in ["uniprot_id","uniprot","accession"] if c in df.columns), None)
    pw = next((c for c in ["reactome_id","pathway_id","Reactome_ID","rid","stId"] if c in df.columns), None)
    if up is None or pw is None:
        up, pw = df.columns[0], df.columns[1]
    out = pd.DataFrame({
        "src_uniprot_id": df[up].astype(str),
        "dst_reactome_id": df[pw].astype(str)
    }).dropna().drop_duplicates()
    out["relation"] = "in_pathway"
    info(f"edges protein-pathway: {out.shape}")
    return out[["src_uniprot_id","dst_reactome_id","relation"]]

def std_edges_ww(df: pd.DataFrame) -> pd.DataFrame:
    # Headerless two columns fallback; otherwise try common names
    src = next((c for c in ["src","source","from","reactome_id_a","parent","from_id"] if c in df.columns), None)
    dst = next((c for c in ["dst","target","to","reactome_id_b","child","to_id"] if c in df.columns), None)
    if src is None or dst is None:
        src, dst = df.columns[0], df.columns[1]
    out = pd.DataFrame({
        "src_reactome_id": df[src].astype(str),
        "dst_reactome_id": df[dst].astype(str)
    }).dropna().drop_duplicates()
    out["relation"] = "related_to"
    info(f"edges pathway-pathway: {out.shape}")
    return out[["src_reactome_id","dst_reactome_id","relation"]]

# ---------- Main ----------
def main(a):
    ensure_dir(OUTDIR)
    info(f"Output dir: {OUTDIR}")

    # 1) Drug nodes
    drugs_df, _ = read_csv_header_or_none(
        a.drugs,
    )
    nodes_drug = std_drug_nodes(drugs_df)
    nodes_drug.to_csv(f"{OUTDIR}/nodes_drug_std.csv", index=False)

    # 2) Protein nodes
    prot_df, _ = read_csv_header_or_none(
        a.uniprot_human,
    )
    nodes_protein = std_protein_nodes(prot_df)
    nodes_protein.to_csv(f"{OUTDIR}/nodes_protein_std.csv", index=False)

    # 3) Pathway nodes (headerless: id,name,species)
    pw_df, how = read_csv_header_or_none(
        a.reactome_pathways,
        names_if_none=["reactome_id","name","species"]
    )
    if how == "header":
        # If someone accidentally saved with a header but different order, standardizer will handle
        pass
    nodes_pathway = std_pathway_nodes(pw_df)
    nodes_pathway.to_csv(f"{OUTDIR}/nodes_pathway_std.csv", index=False)

    # 4) Drug->Target edges (has header)
    dt_df, _ = read_csv_header_or_none(
        a.targets
    )
    e_dt = std_edges_dt(dt_df)
    e_dt.to_csv(f"{OUTDIR}/edges_drug_target_std.csv", index=False)

    # 5) Protein->Pathway edges (headerless: uniprot_id,reactome_id,url,pathway_name,evidence,species) + filter H. sapiens
    ppw_df, _ = read_csv_header_or_none(
        a.reactome_protein_pathway,
        names_if_none=["uniprot_id","reactome_id","url","pathway_name","evidence","species"]
    )
    if "species" in ppw_df.columns:
        ppw_df = ppw_df[ppw_df["species"].astype(str).str.contains("Homo sapiens", case=False, na=False)]
    e_ppw = std_edges_ppw(ppw_df)
    e_ppw.to_csv(f"{OUTDIR}/edges_protein_pathway_std.csv", index=False)

    # 6) Pathway<->Pathway edges (headerless: src,dst)
    ww_df, _ = read_csv_header_or_none(
        a.reactome_relations,
        names_if_none=["src_reactome_id","dst_reactome_id"]
    )
    e_ww = std_edges_ww(ww_df)
    e_ww.to_csv(f"{OUTDIR}/edges_pathway_pathway_std.csv", index=False)

    # 7) Stats
    info("=== Summary ===")
    info(f"Drugs   : {len(nodes_drug):6d}")
    info(f"Proteins: {len(nodes_protein):6d}")
    info(f"Pathways: {len(nodes_pathway):6d}")
    info(f"E_DT    : {len(e_dt):6d}")
    info(f"E_PPW   : {len(e_ppw):6d}")
    info(f"E_WW    : {len(e_ww):6d}")
    info("Done. Next: run check_fk_integrity.py on GNN/GNN_datasets/*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", default="Datasets/normalized/cleaned/drugs_master_clean.csv")
    ap.add_argument("--targets", default="Datasets/normalized/drug_targets.csv")
    ap.add_argument("--uniprot_human", default="Datasets/normalized/protein_layer/proteins_master_human_filtered.csv")
    ap.add_argument("--reactome_pathways", default="Datasets/Reactome/ReactomePathways_human.csv")
    ap.add_argument("--reactome_protein_pathway", default="Datasets/Reactome/Reactome_Uniprot_to_Pathways_human.csv")
    ap.add_argument("--reactome_relations", default="Datasets/Reactome/ReactomePathwaysRelation_comma.csv")
    main(ap.parse_args())