import os, json, argparse
import numpy as np
import pandas as pd

OUTDIR = "GNN/GNN_datasets"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def build_index(series):
    s = pd.Series(series).astype(str)
    uniq = pd.Index(s.unique())
    mapping = {k:i for i,k in enumerate(uniq)}
    return mapping, uniq.size

def map_col_to_idx(col, mapping):
    col = col.astype(str)
    ok = col.isin(mapping.keys())
    return col.map(mapping), ok

def deg_features(n, in_deg, out_deg):
    tot = in_deg + out_deg
    x = np.stack([
        in_deg, out_deg, tot,
        np.log1p(in_deg), np.log1p(out_deg), np.log1p(tot)
    ], axis=1).astype(np.float32)
    if x.shape[0] < n:
        pad = np.zeros((n - x.shape[0], x.shape[1]), dtype=np.float32)
        x = np.vstack([x, pad])
    return x[:n]

def project_to_256(x):
    x = x.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    xn = (x - mu) / sd
    d_in = xn.shape[1]
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1/np.sqrt(d_in), size=(d_in, 256)).astype(np.float32)
    b = np.zeros((256,), dtype=np.float32)
    y = xn @ W + b
    return np.maximum(y, 0)  # ReLU

def main(a):
    base = a.indir
    ensure_dir(base)
    id_dir   = os.path.join(base, "id_maps");     ensure_dir(id_dir)
    ie_dir   = os.path.join(base, "index_edges"); ensure_dir(ie_dir)
    feat_dir = os.path.join(base, "features");    ensure_dir(feat_dir)

    # ---- Load nodes
    nd  = pd.read_csv(os.path.join(base, "nodes_drug_std.csv"))        # drug_id
    npr = pd.read_csv(os.path.join(base, "nodes_protein_std.csv"))     # uniprot_id
    npw = pd.read_csv(os.path.join(base, "nodes_pathway_std.csv"))     # reactome_id, level

    if "drug_id" not in nd.columns:           raise ValueError("nodes_drug_std.csv missing 'drug_id'")
    if "uniprot_id" not in npr.columns:       raise ValueError("nodes_protein_std.csv missing 'uniprot_id'")
    if "reactome_id" not in npw.columns:      raise ValueError("nodes_pathway_std.csv missing 'reactome_id'")
    if "level" not in npw.columns:            raise ValueError("nodes_pathway_std.csv missing 'level'")

    d2i, N_d = build_index(nd["drug_id"])
    p2i, N_p = build_index(npr["uniprot_id"])
    w2i, N_w = build_index(npw["reactome_id"])

    with open(os.path.join(id_dir, "drug2idx.json"), "w") as f:    json.dump(d2i, f)
    with open(os.path.join(id_dir, "protein2idx.json"), "w") as f: json.dump(p2i, f)
    with open(os.path.join(id_dir, "pathway2idx.json"), "w") as f: json.dump(w2i, f)

    # ---- Map edges to index space
    rel_map = {"targets":0, "in_pathway":1, "related_to":2, "inhibits":3, "induces":4, "substrate_of":5}

    def map_edges(fname, src_col, dst_col, src_map, dst_map, default_rel):
        path = os.path.join(base, fname)
        if not os.path.exists(path): return 0
        df = pd.read_csv(path)
        if src_col not in df.columns or dst_col not in df.columns:
            raise ValueError(f"{fname} missing '{src_col}' or '{dst_col}'")
        sidx, sok = map_col_to_idx(df[src_col], src_map)
        didx, dok = map_col_to_idx(df[dst_col], dst_map)
        m = (sok & dok)
        dfm = df.loc[m].copy()
        dfm["src_idx"] = sidx[m].astype(int).values
        dfm["dst_idx"] = didx[m].astype(int).values
        if "relation" in dfm.columns:
            dfm["rel_type"] = dfm["relation"].map(lambda x: rel_map.get(str(x), default_rel)).astype(int)
        else:
            dfm["rel_type"] = default_rel
        out = dfm[["src_idx","dst_idx","rel_type"]].drop_duplicates()
        out.to_csv(os.path.join(ie_dir, default_rel_name[default_rel] + ".csv"), index=False)
        return len(out)

    default_rel_name = {0:"dt", 1:"ppw", 2:"ww", 3:"mech", 4:"mech", 5:"mech"}

    n_dt = map_edges("edges_drug_target_std.csv",    "src_drug_id","dst_uniprot_id", d2i, p2i, 0)
    n_pp = map_edges("edges_protein_pathway_std.csv","src_uniprot_id","dst_reactome_id", p2i, w2i, 1)
    n_ww = map_edges("edges_pathway_pathway_std.csv","src_reactome_id","dst_reactome_id", w2i, w2i, 2)
    n_me = 0
    if os.path.exists(os.path.join(base, "edges_mechanism_std.csv")):
        n_me = map_edges("edges_mechanism_std.csv","src_drug_id","dst_uniprot_id", d2i, p2i, 3)

    # ---- Degree features + pathway level → 256-d features
    d_in  = np.zeros(N_d, dtype=np.float32); d_out = np.zeros(N_d, dtype=np.float32)
    p_in  = np.zeros(N_p, dtype=np.float32); p_out = np.zeros(N_p, dtype=np.float32)
    w_in  = np.zeros(N_w, dtype=np.float32); w_out = np.zeros(N_w, dtype=np.float32)

    def add_degrees(edge_csv, src_kind):
        path = os.path.join(ie_dir, edge_csv)
        if not os.path.exists(path): return
        arr = pd.read_csv(path)[["src_idx","dst_idx"]].to_numpy()
        if src_kind == "drug->prot":
            for s,d in arr: d_out[s]+=1; p_in[d]+=1
        elif src_kind == "prot->path":
            for s,d in arr: p_out[s]+=1; w_in[d]+=1
        elif src_kind == "path->path":
            for s,d in arr: w_out[s]+=1; w_in[d]+=1
        elif src_kind == "drug->prot_mech":
            for s,d in arr: d_out[s]+=1; p_in[d]+=1

    if n_dt>0: add_degrees("dt.csv", "drug->prot")
    if n_pp>0: add_degrees("ppw.csv","prot->path")
    if n_ww>0: add_degrees("ww.csv", "path->path")
    if n_me>0: add_degrees("mech.csv","drug->prot_mech")

    drug_raw = deg_features(N_d, d_in, d_out)                 # [N_d, 6]
    prot_raw = deg_features(N_p, p_in, p_out)                 # [N_p, 6]
    level = npw["level"].to_numpy(dtype=np.float32).reshape(-1,1)
    path_raw = np.concatenate([deg_features(N_w, w_in, w_out), level], axis=1)  # [N_w, 7]

    np.save(os.path.join(feat_dir, "drug_feats_256.npy"),    project_to_256(drug_raw))
    np.save(os.path.join(feat_dir, "protein_feats_256.npy"), project_to_256(prot_raw))
    np.save(os.path.join(feat_dir, "pathway_feats_256.npy"), project_to_256(path_raw))

    meta = {
        "num_nodes": {"drug": int(N_d), "protein": int(N_p), "pathway": int(N_w)},
        "num_edges": {"dt": int(n_dt), "ppw": int(n_pp), "ww": int(n_ww), "mech": int(n_me)},
        "rel_map": {"targets":0, "in_pathway":1, "related_to":2, "inhibits":3, "induces":4, "substrate_of":5}
    }
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[OK] id-maps, index-edges, and 256-d features written to", base)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=OUTDIR)
    main(ap.parse_args())