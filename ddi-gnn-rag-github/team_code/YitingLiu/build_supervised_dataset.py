"""
Build supervised DDI dataset in one pass:

Inputs:
  - labels CSV (positives only): has two ID columns (drug_a, drug_b), or configurable
  - drug_features.csv: per-drug features with ecfp4/ecfp6 (FPSText), physchem, ok flag

Outputs:
  - datasets/supervised/pairs_all.csv            # with features and label y
  - datasets/supervised/train.csv
  - datasets/supervised/val.csv
  - datasets/supervised/test.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import RDLogger
from rdkit import Chem
from rdkit.DataStructs import CreateFromFPSText, TanimotoSimilarity

RDLogger.DisableLog('rdApp.*')  # quieter

def normalize_pairs(df, col_a, col_b):
    a = df[col_a].astype(str).str.strip()
    b = df[col_b].astype(str).str.strip()
    mask = a != b
    a = a[mask]; b = b[mask]
    a2 = np.minimum(a, b)
    b2 = np.maximum(a, b)
    out = pd.DataFrame({ "drug_a": a2, "drug_b": b2 })
    out = out.drop_duplicates().reset_index(drop=True)
    return out

def load_features(path, id_col, smiles_col):
    df = pd.read_csv(path)
    need = [id_col, "ecfp4", "ecfp6", "ok", "MolWt","LogP","TPSA","HBA","HBD","RotB"]
    missing = [n for n in need if n not in df.columns]
    if missing:
        raise ValueError(f"[ERR] Missing column in features: {missing}")
    df[id_col] = df[id_col].astype(str).str.strip()
    # smiles 可选：不存在就建空列，存在就保留
    if smiles_col not in df.columns:
        df[smiles_col] = pd.NA
    return df

def bitvec_or_none(fptext):
    s = str(fptext)
    if not s or s.lower()=="nan":
        return None
    try:
        return CreateFromFPSText(s)
    except Exception:
        return None

def tanimoto_from_fptexts(fp_a, fp_b):
    ba = bitvec_or_none(fp_a)
    bb = bitvec_or_none(fp_b)
    if ba is None or bb is None:
        return np.nan
    return float(TanimotoSimilarity(ba, bb))

def pairwise_features(rows, fmap, id_col):
    # join features for drug_a and drug_b
    fa = fmap.add_prefix("a_")
    fb = fmap.add_prefix("b_")
    rows = rows.merge(fmap, left_on="drug_a", right_on=id_col, how="left").merge(fmap, left_on="drug_b", right_on=id_col, how="left", suffixes=("_a","_b"))
    # rename already handled by suffixes; ensure we have columns
    # Compute Tanimoto on ecfp4/ecfp6
    rows["tani_ecfp4"] = rows.apply(lambda r: tanimoto_from_fptexts(r["ecfp4_a"], r["ecfp4_b"]), axis=1)
    rows["tani_ecfp6"] = rows.apply(lambda r: tanimoto_from_fptexts(r["ecfp6_a"], r["ecfp6_b"]), axis=1)
    # Physchem simple combos
    for col in ["MolWt","LogP","TPSA","HBA","HBD","RotB"]:
        rows[f"abs_{col}"] = (rows[f"{col}_a"] - rows[f"{col}_b"]).abs()
        rows[f"sum_{col}"] = rows[f"{col}_a"] + rows[f"{col}_b"]
        rows[f"min_{col}"] = rows[[f"{col}_a", f"{col}_b"]].min(axis=1)
        rows[f"max_{col}"] = rows[[f"{col}_a", f"{col}_b"]].max(axis=1)
    # Keep essentials; drop raw FP if想减小体积
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV of positive DDI pairs")
    ap.add_argument("--features", required=True, help="drug_features.csv")
    ap.add_argument("--outdir", default="datasets/supervised")
    ap.add_argument("--id-col", default="drug_id", help="drug ID column in features")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--labels-col-a", default="drug_a", help="column name for drug A in labels")
    ap.add_argument("--labels-col-b", default="drug_b", help="column name for drug B in labels")
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="negatives per positive")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="0.8,0.1,0.1", help="train,val,test ratios")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读正例并规范化
    pos_raw = pd.read_csv(args.labels)
    for c in [args.labels_col_a, args.labels_col_b]:
        if c not in pos_raw.columns:
            raise ValueError(f"[ERR] Missing column in labels: {c}")
    pos = normalize_pairs(pos_raw, args.labels_col_a, args.labels_col_b)
    pos["y"] = 1

    # 2) 读 per-drug 特征
    feats = load_features(args.features, args.id_col, args.smiles_col)
    drug_ids = feats[args.id_col].unique()
    drug_set = set(drug_ids)

    # 3) 过滤正例只保留两端都在 features 中的
    pos = pos[pos["drug_a"].isin(drug_set) & pos["drug_b"].isin(drug_set)].reset_index(drop=True)

    # 4) 采样负例（从所有可能对中减去正例）
    np.random.seed(args.seed)
    # 若药物太多，构造全集会爆；我们在正例邻域内采样：从出现过的药物集合取笛卡尔抽样
    used_drugs = pd.unique(pos[["drug_a","drug_b"]].values.ravel())
    used_drugs = np.array(used_drugs)
    n_pos = len(pos)
    n_neg_target = int(np.ceil(n_pos * args.neg_ratio))

    # 生成候选对（随机抽样法）
    neg_pairs = set()
    pos_set = set(map(tuple, pos[["drug_a","drug_b"]].itertuples(index=False, name=None)))
    tries = 0
    max_tries = n_neg_target * 20 + 10000
    while len(neg_pairs) < n_neg_target and tries < max_tries:
        a = np.random.choice(used_drugs, size=n_neg_target*2, replace=True)
        b = np.random.choice(used_drugs, size=n_neg_target*2, replace=True)
        for x, y in zip(a, b):
            if x == y: continue
            d1, d2 = (x, y) if x < y else (y, x)
            tp = (d1, d2)
            if tp in pos_set or tp in neg_pairs: continue
            neg_pairs.add(tp)
            if len(neg_pairs) >= n_neg_target:
                break
        tries += 1

    neg = pd.DataFrame(list(neg_pairs), columns=["drug_a","drug_b"])
    neg["y"] = 0

    # 5) 合并正负并计算对间特征
    all_pairs = pd.concat([pos, neg], ignore_index=True)
    # 与 features 连接并计算 pairwise 特征
    # 这里先把 features 列重命名为 *_a/_b 后再做 Tanimoto/physchem 组合
    feats_a = feats.rename(columns={c: f"{c}_a" for c in feats.columns})
    feats_b = feats.rename(columns={c: f"{c}_b" for c in feats.columns})
    df = all_pairs.merge(feats_a, left_on="drug_a", right_on=f"{args.id_col}_a", how="left") \
                  .merge(feats_b, left_on="drug_b", right_on=f"{args.id_col}_b", how="left")

    # Tanimoto（从 FPSText 还原）
    df["tani_ecfp4"] = df.apply(lambda r: tanimoto_from_fptexts(r["ecfp4_a"], r["ecfp4_b"]), axis=1)
    df["tani_ecfp6"] = df.apply(lambda r: tanimoto_from_fptexts(r["ecfp6_a"], r["ecfp6_b"]), axis=1)

    # Physchem 组合
    for col in ["MolWt","LogP","TPSA","HBA","HBD","RotB"]:
        df[f"abs_{col}"] = (df[f"{col}_a"] - df[f"{col}_b"]).abs()
        df[f"sum_{col}"] = df[f"{col}_a"] + df[f"{col}_b"]
        df[f"min_{col}"] = df[[f"{col}_a", f"{col}_b"]].min(axis=1)
        df[f"max_{col}"] = df[[f"{col}_a", f"{col}_b"]].max(axis=1)

    # 6) 保存全集
    out_all = outdir / "pairs_all.csv"
    df.to_csv(out_all, index=False)
    print(f"[OK] wrote {out_all}  rows={len(df)}  pos={int(df['y'].sum())}  neg={int((1-df['y']).sum())}")

    # 7) 随机划分（pair-level）
    ratios = [float(x) for x in args.splits.split(",")]
    assert abs(sum(ratios)-1.0) < 1e-6 and len(ratios)==3
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df)*ratios[0])
    n_val   = int(len(df)*ratios[1])
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    df.iloc[train_idx].to_csv(outdir / "train.csv", index=False)
    df.iloc[val_idx].to_csv(outdir / "val.csv", index=False)
    df.iloc[test_idx].to_csv(outdir / "test.csv", index=False)
    print(f"[SPLIT] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  seed={args.seed}")

if __name__ == "__main__":
    main()