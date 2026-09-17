import os, sys, argparse, json
import pandas as pd

def info(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err (msg): print(f"[ERROR] {msg}", file=sys.stderr)

def load_csv(path, required_cols=None, header="infer", names=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=header, names=names)
    if required_cols:
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{path} missing columns: {miss}; has: {list(df.columns)}")
    return df

def show_examples(df, cols, n=5):
    try:
        print(df[cols].head(n).to_string(index=False))
    except Exception:
        print(df.head(n).to_string(index=False))

def check_edge_fk(edge_path, src_col, dst_col, src_set, dst_set, edge_name, limit=10):
    df = load_csv(edge_path)
    # normalize types
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)
    bad_src = df[~df[src_col].isin(src_set)]
    bad_dst = df[~df[dst_col].isin(dst_set)]
    ok = True
    if len(bad_src):
        ok = False
        err(f"{edge_name}: {len(bad_src)} rows have src not in nodes ({src_col}) @ {edge_path}")
        show_examples(bad_src, [src_col, dst_col], n=min(limit, len(bad_src)))
    if len(bad_dst):
        ok = False
        err(f"{edge_name}: {len(bad_dst)} rows have dst not in nodes ({dst_col}) @ {edge_path}")
        show_examples(bad_dst, [src_col, dst_col], n=min(limit, len(bad_dst)))
    if ok:
        info(f"{edge_name}: ✅ FK PASSED ({len(df)} edges)")
    return ok

def check_pairs_fk(base, nodes_drug_set, id_maps_dir=None, limit=10):
    ok_all = True
    def _check(path_csv, uses_idx=False):
        nonlocal ok_all
        if not os.path.exists(path_csv): 
            return
        df = load_csv(path_csv)
        if uses_idx:
            # index form: require id_maps
            if not id_maps_dir or not os.path.exists(os.path.join(id_maps_dir,"drug2idx.json")):
                warn(f"Skip idx FK check (id_maps missing) for {path_csv}")
                return
            with open(os.path.join(id_maps_dir,"drug2idx.json"),"r") as f: d2i = {k:int(v) for k,v in json.load(f).items()}
            # inverse map
            i2d = {v:k for k,v in d2i.items()}
            for col in ["a_idx","b_idx"]:
                if col not in df.columns:
                    warn(f"{path_csv} missing {col}, skip idx check.")
                    return
            # Map back to ids and check
            a_ids = df["a_idx"].map(i2d)
            b_ids = df["b_idx"].map(i2d)
            bad_a = df[a_ids.isna()]
            bad_b = df[b_ids.isna()]
            ok = True
            if len(bad_a):
                ok = False
                err(f"{os.path.basename(path_csv)}: {len(bad_a)} rows have a_idx not in id_maps")
                show_examples(bad_a, ["a_idx","b_idx"], n=min(limit, len(bad_a)))
            if len(bad_b):
                ok = False
                err(f"{os.path.basename(path_csv)}: {len(bad_b)} rows have b_idx not in id_maps")
                show_examples(bad_b, ["a_idx","b_idx"], n=min(limit, len(bad_b)))
            if ok:
                info(f"{os.path.basename(path_csv)} (idx): ✅ FK PASSED ({len(df)} pairs)")
            ok_all = ok_all and ok
        else:
            # id form
            need = []
            for c in ["drug_a","drug_b"]:
                if c not in df.columns: 
                    warn(f"{path_csv} missing {c}, skip id check.")
                    return
                need.append(c)
            df["drug_a"] = df["drug_a"].astype(str)
            df["drug_b"] = df["drug_b"].astype(str)
            bad_a = df[~df["drug_a"].isin(nodes_drug_set)]
            bad_b = df[~df["drug_b"].isin(nodes_drug_set)]
            ok = True
            if len(bad_a):
                ok = False
                err(f"{os.path.basename(path_csv)}: {len(bad_a)} rows have drug_a not in nodes_drug")
                show_examples(bad_a, ["drug_a","drug_b"], n=min(limit, len(bad_a)))
            if len(bad_b):
                ok = False
                err(f"{os.path.basename(path_csv)}: {len(bad_b)} rows have drug_b not in nodes_drug")
                show_examples(bad_b, ["drug_a","drug_b"], n=min(limit, len(bad_b)))
            if ok:
                info(f"{os.path.basename(path_csv)} (id): ✅ FK PASSED ({len(df)} pairs)")
            ok_all = ok_all and ok

    # run for all splits
    for tag in ["train","val","test","pairs_all"]:
        _check(os.path.join(base, f"{tag}_pairs.csv"), uses_idx=False)
        _check(os.path.join(base, f"{tag}_pairs_idx.csv"), uses_idx=True)
    return ok_all

def main(a):
    base = a.base
    # Load node sets
    nd = load_csv(os.path.join(base, "nodes_drug_std.csv"),     required_cols=["drug_id"])
    npn= load_csv(os.path.join(base, "nodes_protein_std.csv"),  required_cols=["uniprot_id"])
    npa= load_csv(os.path.join(base, "nodes_pathway_std.csv"),  required_cols=["reactome_id"])

    drugs   = set(nd["drug_id"].astype(str))
    prots   = set(npn["uniprot_id"].astype(str))
    pathways= set(npa["reactome_id"].astype(str))

    # Edges
    ok = True
    ok &= check_edge_fk(
        os.path.join(base, "edges_drug_target_std.csv"),
        "src_drug_id", "dst_uniprot_id",
        drugs, prots, "edges_drug_target_std"
    )
    ok &= check_edge_fk(
        os.path.join(base, "edges_protein_pathway_std.csv"),
        "src_uniprot_id", "dst_reactome_id",
        prots, pathways, "edges_protein_pathway_std"
    )
    ok &= check_edge_fk(
        os.path.join(base, "edges_pathway_pathway_std.csv"),
        "src_reactome_id", "dst_reactome_id",
        pathways, pathways, "edges_pathway_pathway_std"
    )
    # Optional mechanism edges
    mech_path = os.path.join(base, "edges_mechanism_std.csv")
    if os.path.exists(mech_path):
        ok &= check_edge_fk(
            mech_path, "src_drug_id", "dst_uniprot_id",
            drugs, prots, "edges_mechanism_std"
        )
    else:
        info("edges_mechanism_std.csv: (optional) not found, skip.")

    # Optional pairs
    if a.check_pairs:
        id_maps_dir = os.path.join(base, "id_maps")
        ok &= check_pairs_fk(base, drugs, id_maps_dir=id_maps_dir)

    # Summary & exit
    if ok:
        info("FK CHECK: ✅ ALL PASSED")
        sys.exit(0)
    else:
        if a.no_fail:
            warn("FK CHECK: errors found, but continuing due to --no_fail.")
            sys.exit(0)
        else:
            err("FK CHECK: ❌ FAILED (see errors above).")
            sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="GNN/GNN_datasets", help="Base directory containing nodes_* and edges_* CSVs")
    ap.add_argument("--check_pairs", action="store_true", help="Also check {train,val,test}_pairs(.csv/.idx.csv) FK against nodes_drug")
    ap.add_argument("--no_fail", action="store_true", help="Do not exit with non-zero even if errors are found")
    args = ap.parse_args()
    main(args)