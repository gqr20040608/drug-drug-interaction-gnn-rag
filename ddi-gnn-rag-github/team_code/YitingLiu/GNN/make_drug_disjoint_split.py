import os
import numpy as np
import pandas as pd

SPLIT_SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main(indir="GNN_datasets"):
    pairs_path = os.path.join(indir, "pairs_all_idx.csv")
    df = pd.read_csv(pairs_path)

    # ------------ 你文件里的真实列名是这三个 -----------
    u_col = "a_idx"
    v_col = "b_idx"
    label_col = "y"
    # ----------------------------------------------------

    print("Detected columns:", df.columns.tolist())
    print(f"Using: u={u_col}, v={v_col}, label={label_col}")

    # 1) 收集所有药物 ID
    drugs = np.unique(np.concatenate([df[u_col].values, df[v_col].values]))

    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(drugs)

    n = len(drugs)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_drugs = set(drugs[:n_train])
    val_drugs = set(drugs[n_train:n_train+n_val])
    test_drugs = set(drugs[n_train+n_val:])

    print(f"Total drugs={n}")
    print(f"  Train={len(train_drugs)}, Val={len(val_drugs)}, Test={len(test_drugs)}")

    # 2) 分配 split：只有两端都在同一个 split 才保留
    def assign_split(row):
        u = row[u_col]
        v = row[v_col]
        if (u in train_drugs) and (v in train_drugs):
            return "train"
        if (u in val_drugs) and (v in val_drugs):
            return "val"
        if (u in test_drugs) and (v in test_drugs):
            return "test"
        return "cross"

    df["split"] = df.apply(assign_split, axis=1)

    df_train = df[df["split"] == "train"].drop(columns=["split"])
    df_val = df[df["split"] == "val"].drop(columns=["split"])
    df_test = df[df["split"] == "test"].drop(columns=["split"])

    dropped = len(df) - len(df_train) - len(df_val) - len(df_test)

    print(f"Pairs total={len(df)}")
    print(f"  Train pairs={len(df_train)}")
    print(f"  Val pairs  ={len(df_val)}")
    print(f"  Test pairs ={len(df_test)}")
    print(f"  Dropped cross-split pairs (cannot be assigned)={dropped}")

    # 3) 保存
    df_train.to_csv(os.path.join(indir, "train_pairs_idx_drugdisjoint.csv"), index=False)
    df_val.to_csv(os.path.join(indir, "val_pairs_idx_drugdisjoint.csv"), index=False)
    df_test.to_csv(os.path.join(indir, "test_pairs_idx_drugdisjoint.csv"), index=False)

    print("\nSaved to:")
    print("  train_pairs_idx_drugdisjoint.csv")
    print("  val_pairs_idx_drugdisjoint.csv")
    print("  test_pairs_idx_drugdisjoint.csv")


if __name__ == "__main__":
    main()