import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-feats",
        required=True,
        help="Path to base drug features (e.g., Morgan/RDKit), shape (N, d_base)",
    )
    parser.add_argument(
        "--chemberta-feats",
        required=True,
        help="Path to ChemBERTa embeddings, shape (N, d_chemberta)",
    )
    parser.add_argument(
        "--out-npy",
        required=True,
        help="Output .npy path for fused PCA-compressed features",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=256,
        help="Output feature dimension after PCA (default: 256)",
    )
    args = parser.parse_args()

    base_path = args.base_feats
    chem_path = args.chemberta_feats
    out_path = args.out_npy
    out_dim = args.out_dim

    print("[INFO] loading base drug features from:", base_path)
    base = np.load(base_path)  # (N, d_base)
    print(f"[INFO] base shape = {base.shape}")

    print("[INFO] loading ChemBERTa drug features from:", chem_path)
    chem = np.load(chem_path)  # (N, d_chemberta)
    print(f"[INFO] chemberta shape = {chem.shape}")

    if base.shape[0] != chem.shape[0]:
        raise ValueError(
            f"Row mismatch: base has {base.shape[0]} rows, "
            f"chemberta has {chem.shape[0]} rows."
        )

    # 1. 拼接： (N, d_base + d_chemberta)
    print("[INFO] concatenating along feature dimension ...")
    X = np.concatenate([base, chem], axis=1)
    print(f"[INFO] concatenated shape = {X.shape}")

    # 2. 标准化：每一维 zero-mean / unit-variance
    print("[INFO] standardizing concatenated features ...")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 3. PCA 降维到 out_dim
    print(f"[INFO] running PCA -> {out_dim} dims ...")
    pca = PCA(n_components=out_dim, random_state=42)
    X_pca = pca.fit_transform(X_std)

    print("[INFO] first 10 dims, explained variance ratio:")
    print(pca.explained_variance_ratio_[:10])
    print(
        f"[INFO] total explained variance (sum over {out_dim} comps): "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # 4. 保存为 float32
    X_pca = X_pca.astype("float32")
    np.save(out_path, X_pca)
    print(f"[OK] saved fused drug features to {out_path}, shape = {X_pca.shape}")


if __name__ == "__main__":
    main()