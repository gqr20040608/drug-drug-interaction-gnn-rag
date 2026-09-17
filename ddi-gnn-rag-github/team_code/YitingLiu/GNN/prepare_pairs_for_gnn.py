import argparse, json, pandas as pd, os

def load_map(p): 
    with open(p,"r") as f: 
        return {k:int(v) for k,v in json.load(f).items()}

def convert(split_path, d2i):
    df = pd.read_csv(split_path)
    need = ["drug_a","drug_b","y"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{split_path} missing column: {c}")
    df["drug_a"] = df["drug_a"].astype(str)
    df["drug_b"] = df["drug_b"].astype(str)
    df["a_idx"] = df["drug_a"].map(d2i)
    df["b_idx"] = df["drug_b"].map(d2i)
    df = df[df["a_idx"].notna() & df["b_idx"].notna()].copy()
    df["a_idx"] = df["a_idx"].astype(int)
    df["b_idx"] = df["b_idx"].astype(int)
    if "weight" not in df.columns: df["weight"] = 1.0
    keep = ["drug_a","drug_b","y","weight","a_idx","b_idx"]
    return df[keep]

def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    d2i = load_map(a.drug2idx)
    for tag, p in [("train",a.train),("val",a.val),("test",a.test)]:
        df = convert(p, d2i)
        df.to_csv(os.path.join(a.outdir, f"{tag}_pairs.csv"), index=False)
        df[["a_idx","b_idx","y","weight"]].to_csv(os.path.join(a.outdir, f"{tag}_pairs_idx.csv"), index=False)
    if a.pairs_all and os.path.exists(a.pairs_all):
        df = convert(a.pairs_all, d2i)
        df.to_csv(os.path.join(a.outdir, f"pairs_all.csv"), index=False)
        df[["a_idx","b_idx","y","weight"]].to_csv(os.path.join(a.outdir, f"pairs_all_idx.csv"), index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs_all")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--nodes_drug", required=True)  # 保留接口一致性，未使用
    ap.add_argument("--drug2idx",   required=True)
    ap.add_argument("--outdir",     required=True)
    main(ap.parse_args())