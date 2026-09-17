"""
build_mech_features.py
输入:
  - run_dir = supervised_v1/<run_id>/  内有 train.csv/val.csv/test.csv
  - drug-target 边:  优先找以下其一(按存在性自动选择):
      Datasets/normalized/edges_drug_target_std.csv
      Datasets/normalized/merged/edges_drug_target.csv
      Datasets/normalized/cleaned/edges_drug_target_clean.csv
  - 可选 protein-pathway 边:
      Datasets/normalized/edges_protein_pathway.csv (或 *_clean/_std/_merged 变体)
输出:
  - 覆盖写回 train/val/test（三个文件新增机制列）
机制列:
  - deg_dA, deg_dB: 药物的靶点度
  - mech_shared_targets, jacc_targets, overlap_targets_min, overlap_targets_max, pa_targets (偏好连接)
  - 若提供 pathway 边: mech_shared_pathways, jacc_pathways, overlap_pathways_min/max, pa_pathways
"""
from pathlib import Path
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

RUN_DIR = sorted([p for p in (Path("Datasets/release/supervised_v1")).iterdir()
                  if p.is_dir() and p.name.startswith("v_")])[-1]

CAND_DT = [
    Path("Datasets/normalized/drug_layer/edges_drug_target_std.csv"),
]

CAND_PW = [
    Path("Datasets/release/v_20251008_0221/edges_protein_pathway.csv"),
]
def pick_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def build_index_dt(df_dt):
    # 期望列: drug_id, protein_id（常见命名）
    cols = df_dt.columns.str.lower()
    col_d = df_dt.columns[cols.str.contains("drug")]
    col_p = df_dt.columns[cols.str.contains("prot")]
    assert len(col_d)>0 and len(col_p)>0, "drug-target file needs columns like drug_id and protein_id"
    dcol, pcol = col_d[0], col_p[0]

    drug2prot = defaultdict(set)
    for d, p in zip(df_dt[dcol].astype(str), df_dt[pcol].astype(str)):
        drug2prot[d].add(p)
    prot2path = None
    return drug2prot, prot2path

def build_index_pw(df_pw):
    cols = df_pw.columns.str.lower()
    col_p = df_pw.columns[cols.str.contains("prot")]
    col_w = df_pw.columns[cols.str.contains("path|reactome|pw")]
    assert len(col_p)>0 and len(col_w)>0, "protein-pathway file needs columns like protein_id and pathway_id"
    pcol, wcol = col_p[0], col_w[0]

    prot2path = defaultdict(set)
    for p, w in zip(df_pw[pcol].astype(str), df_pw[wcol].astype(str)):
        prot2path[p].add(w)
    return prot2path

def pair_mech_features(drug2prot, prot2path, a, b):
    Sa, Sb = drug2prot.get(a, set()), drug2prot.get(b, set())
    inter_t = Sa & Sb
    union_t = Sa | Sb
    deg_a, deg_b = len(Sa), len(Sb)
    # targets
    shared_t = len(inter_t)
    jacc_t = (len(inter_t)/len(union_t)) if union_t else 0.0
    over_min_t = (len(inter_t)/min(deg_a,deg_b)) if min(deg_a,deg_b)>0 else 0.0
    over_max_t = (len(inter_t)/max(deg_a,deg_b)) if max(deg_a,deg_b)>0 else 0.0
    pa_t = deg_a * deg_b

    # pathways (optional)
    shared_p = jacc_p = over_min_p = over_max_p = pa_p = 0.0
    if prot2path is not None:
        Wa = set().union(*[prot2path.get(p,set()) for p in Sa]) if Sa else set()
        Wb = set().union(*[prot2path.get(p,set()) for p in Sb]) if Sb else set()
        inter_w = Wa & Wb
        union_w = Wa | Wb
        deg_w_a, deg_w_b = len(Wa), len(Wb)
        shared_p = len(inter_w)
        jacc_p = (len(inter_w)/len(union_w)) if union_w else 0.0
        over_min_p = (len(inter_w)/min(deg_w_a,deg_w_b)) if min(deg_w_a,deg_w_b)>0 else 0.0
        over_max_p = (len(inter_w)/max(deg_w_a,deg_w_b)) if max(deg_w_a,deg_w_b)>0 else 0.0
        pa_p = deg_w_a * deg_w_b

    return {
        "deg_dA": deg_a, "deg_dB": deg_b,
        "mech_shared_targets": shared_t,
        "jacc_targets": jacc_t,
        "overlap_targets_min": over_min_t,
        "overlap_targets_max": over_max_t,
        "pa_targets": pa_t,
        "mech_shared_pathways": shared_p,
        "jacc_pathways": jacc_p,
        "overlap_pathways_min": over_min_p,
        "overlap_pathways_max": over_max_p,
        "pa_pathways": pa_p,
    }

def process_split(df, drug2prot, prot2path):
    out_rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Building mech feats"):
        a = str(getattr(r, "drug_id_a", getattr(r, "drug_a")))
        b = str(getattr(r, "drug_id_b", getattr(r, "drug_b")))
        feats = pair_mech_features(drug2prot, prot2path, a, b)
        out_rows.append(feats)
    mech = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), mech], axis=1)

def main():
    run_dir = RUN_DIR
    print(f"[INFO] run_dir = {run_dir}")

    dt_path = pick_first(CAND_DT)
    if dt_path is None:
        raise FileNotFoundError("No drug-target file found in candidates.")
    print(f"[INFO] Using drug-target: {dt_path}")
    df_dt = pd.read_csv(dt_path)
    drug2prot, prot2path = build_index_dt(df_dt)

    pw_path = pick_first(CAND_PW)
    if pw_path and pw_path.exists():
        print(f"[INFO] Using protein-pathway: {pw_path}")
        df_pw = pd.read_csv(pw_path)
        prot2path = build_index_pw(df_pw)
    else:
        print("[WARN] No protein-pathway file found. Pathway features will be zeros.")

    for name in ["train","val","test"]:
        df = pd.read_csv(run_dir/f"{name}.csv")
        df2 = process_split(df, drug2prot, prot2path)
        df2.to_csv(run_dir/f"{name}.csv", index=False)  # 直接覆盖，训练脚本会自动拾取
        print(f"[OUT] wrote {run_dir/f'{name}.csv'}  (+mechanistic features)")

if __name__ == "__main__":
    main()