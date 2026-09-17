"""
check_graph.py

Purpose:
  Sanity-check the GNN_datasets graph files before touching any model code.

What it does:
  1. Inspect node csvs (drug / protein / pathway): counts and columns.
  2. Inspect edge csvs (drug-target / protein-pathway / pathway-pathway): index ranges.
  3. Inspect DDI train/val/test splits for potential leakage.
"""

import pandas as pd
from pathlib import Path

# ==============================
# 1. CONFIG: adapt to your project layout
# ==============================

# Root directory for all graph files used by the GNN baseline
DATA_DIR = Path("GNN_datasets")

# Node files (standardized versions)
NODE_FILES = {
    "drug": DATA_DIR / "nodes_drug_std.csv",
    "protein": DATA_DIR / "nodes_protein_std.csv",
    "pathway": DATA_DIR / "nodes_pathway_std.csv",
}

# Edge files (index-based versions are usually under index_edges/)
# If you actually use edges_*_std.csv directly for the GNN, switch to those instead.
EDGE_FILES = {
    "drug_target": DATA_DIR / "index_edges/dt.csv",
    "protein_protein": DATA_DIR / "index_edges/ppw.csv",
    "pathway_pathway": DATA_DIR / "index_edges/ww.csv",
}

# DDI splits (these are the supervised drug–drug pairs for training the model)
DDI_TRAIN = DATA_DIR / "train_pairs_idx.csv"
DDI_VAL   = DATA_DIR / "val_pairs_idx.csv"
DDI_TEST  = DATA_DIR / "test_pairs_idx.csv"

# Candidate column names for src/dst in edge/pair tables
SRC_CANDIDATES = [
    "src", "lhs_idx", "lhs", "drug1_idx", "drug_a_idx",
    "src_idx", "a_idx"         # <- added for your files
]

DST_CANDIDATES = [
    "dst", "rhs_idx", "rhs", "drug2_idx", "drug_b_idx",
    "dst_idx", "b_idx"         # <- added for your files
]

# Label column name candidates in DDI splits
LABEL_CANDIDATES = ["label", "y", "interaction", "ddi_label"]
# (this already covers 'y', so no change needed)


# ==============================
# 2. Helper functions
# ==============================

def find_first_column(candidates, columns):
    """Return the first column name from `candidates` that exists in `columns`,
    or None if none are found."""
    for c in candidates:
        if c in columns:
            return c
    return None


# ==============================
# 3. Node inspection
# ==============================
def inspect_nodes():
    print("=== [Nodes] ===")
    total_nodes = 0
    for ntype, path in NODE_FILES.items():
        if not path.exists():
            print(f"[WARN] Node file for {ntype} not found: {path}")
            continue

        df = pd.read_csv(path)
        n_count = len(df)
        total_nodes += n_count

        print(f"[{ntype}] rows={n_count}")
        print(f"  columns: {list(df.columns)}")

    print(f"[Global] total_nodes (sum over node tables) = {total_nodes}")
    print()


# ==============================
# 4. Edge inspection
# ==============================
def inspect_edges():
    print("=== [Edges] ===")

    for etype, path in EDGE_FILES.items():
        if not path.exists():
            print(f"[WARN] Edge file for {etype} not found: {path}")
            continue

        df = pd.read_csv(path)
        cols = list(df.columns)
        print(f"[{etype}] rows={len(df)}")
        print(f"  columns: {cols}")

        src_col = find_first_column(SRC_CANDIDATES, cols)
        dst_col = find_first_column(DST_CANDIDATES, cols)

        if src_col is None or dst_col is None:
            print("  [WARN] Could not find clear src/dst columns "
                  f"(looked for {SRC_CANDIDATES} / {DST_CANDIDATES}).")
            print("  Skipping range checks for this file.\n")
            continue

        s_min, s_max = df[src_col].min(), df[src_col].max()
        d_min, d_max = df[dst_col].min(), df[dst_col].max()

        print(f"  Using src_col='{src_col}', dst_col='{dst_col}'")
        print(f"  src index range: [{s_min}, {s_max}]")
        print(f"  dst index range: [{d_min}, {d_max}]")
        print()

    print()


# ==============================
# 5. DDI train/val/test split checks
# ==============================
def load_ddi_split(path, split_name):
    if not path.exists():
        print(f"[WARN] DDI {split_name} file not found: {path}")
        return None

    df = pd.read_csv(path)
    cols = list(df.columns)
    print(f"[{split_name}] rows={len(df)}")
    print(f"  columns: {cols}")

    src_col = find_first_column(SRC_CANDIDATES, cols)
    dst_col = find_first_column(DST_CANDIDATES, cols)
    label_col = find_first_column(LABEL_CANDIDATES, cols)

    if src_col is None or dst_col is None:
        print(f"  [ERROR] Could not find src/dst columns in {path}")
        return None

    return df, src_col, dst_col, label_col


def canonical_pair(a, b):
    """Return an unordered pair so (a,b) and (b,a) map to the same key."""
    return (a, b) if a <= b else (b, a)


def inspect_ddi_splits():
    print("=== [DDI Splits] ===")

    train_res = load_ddi_split(DDI_TRAIN, "train")
    val_res   = load_ddi_split(DDI_VAL, "val")
    test_res  = load_ddi_split(DDI_TEST, "test")

    if train_res is None or val_res is None or test_res is None:
        print("[INFO] Some DDI split files are missing or invalid, skipping leakage checks.")
        print()
        return

    train_df, train_src, train_dst, train_label = train_res
    val_df,   val_src,   val_dst,   val_label   = val_res
    test_df,  test_src,  test_dst,  test_label  = test_res

    def to_pair_set(df, s_col, d_col):
        pairs = [canonical_pair(a, b) for a, b in zip(df[s_col], df[d_col])]
        return set(pairs)

    train_pairs = to_pair_set(train_df, train_src, train_dst)
    val_pairs   = to_pair_set(val_df,   val_src,   val_dst)
    test_pairs  = to_pair_set(test_df,  test_src,  test_dst)

    print(f"[train] #unique_pairs={len(train_pairs)}")
    print(f"[val]   #unique_pairs={len(val_pairs)}")
    print(f"[test]  #unique_pairs={len(test_pairs)}")

    # Check intersections (data leakage)
    inter_train_val  = train_pairs & val_pairs
    inter_train_test = train_pairs & test_pairs
    inter_val_test   = val_pairs & test_pairs

    if inter_train_val:
        print(f"[ERROR] train & val share {len(inter_train_val)} pairs (data leakage)!")
    if inter_train_test:
        print(f"[ERROR] train & test share {len(inter_train_test)} pairs (data leakage)!")
    if inter_val_test:
        print(f"[ERROR] val & test share {len(inter_val_test)} pairs (data leakage)!")

    # Optional: label distribution
    for name, df, lab_col in [
        ("train", train_df, train_label),
        ("val",   val_df,   val_label),
        ("test",  test_df,  test_label),
    ]:
        if lab_col is not None and lab_col in df.columns:
            print(f"\n[{name}] label distribution (column '{lab_col}'):")
            print(df[lab_col].value_counts())
        else:
            print(f"\n[{name}] has no label column detected; skipped label distribution.")

    print()


# ==============================
# 6. main
# ==============================
if __name__ == "__main__":
    print("=== Running graph sanity checks on GNN_datasets ===\n")
    inspect_nodes()
    inspect_edges()
    inspect_ddi_splits()
    print("=== Done. ===")
