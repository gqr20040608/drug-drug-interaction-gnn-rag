"""
Build new protein feature matrix from simple text-based embeddings.

Inputs:
  - nodes_protein_std.csv   (must contain columns: uniprot_id, gene, name)

Assumptions:
  - Row index in nodes_protein_std.csv corresponds to protein node index
    (i.e., protein 0 is row 0, protein 1 is row 1, ...), consistent with your
    existing protein_feats_256.npy.

Output:
  - features/protein_feats_256.npy  (shape: [num_proteins, 256], dtype: float32)

Usage example (from GNN/ directory):

  python3 build_protein_feats_text.py \
    --indir GNN_datasets \
    --dim 256
"""

import argparse
import os
import re
import hashlib

import numpy as np
import pandas as pd


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--indir",
        type=str,
        default="GNN_datasets",
        help="Root directory of the GNN dataset (contains nodes_protein_std.csv, features/).",
    )
    ap.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Feature dimension for the protein embeddings.",
    )
    return ap


_token_re = re.compile(r"[A-Za-z0-9]+")


def text_to_hash_vec(text: str, dim: int) -> np.ndarray:
    """
    Very lightweight text embedding: bag-of-tokens with hashing trick.

    Steps:
      - lower-case the text
      - extract alphanumeric tokens
      - for each token, hash it and map to an index in [0, dim)
      - increment the corresponding feature dimension
      - L2-normalize the vector

    This is not SOTA, but it is:
      - deterministic
      - requires no external libraries
      - usually better than a constant / random feature vector
    """
    vec = np.zeros((dim,), dtype=np.float32)
    if not isinstance(text, str):
        return vec

    tokens = _token_re.findall(text.lower())
    if not tokens:
        return vec

    for tok in tokens:
        # Use a stable hash (md5) instead of Python's built-in hash (which is salted)
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


def main():
    args = build_argparser().parse_args()
    indir = args.indir
    dim = args.dim

    nodes_path = os.path.join(indir, "nodes_protein_std.csv")
    out_path = os.path.join(indir, "features", f"protein_feats_{dim}.npy")

    print(f"[INFO] Reading protein node table: {nodes_path}")
    df = pd.read_csv(nodes_path)

    required_cols = {"uniprot_id", "gene", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"nodes_protein_std.csv is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    num_proteins = len(df)
    print(f"[INFO] num_proteins = {num_proteins}")
    print(f"[INFO] Building feature matrix of shape ({num_proteins}, {dim})")

    feats = np.zeros((num_proteins, dim), dtype=np.float32)

    for i, row in df.iterrows():
        # Build a simple text description for each protein
        desc = f"UNIPROT={row['uniprot_id']} GENE={row['gene']} NAME={row['name']}"
        feats[i] = text_to_hash_vec(desc, dim)

        if (i + 1) % 1000 == 0 or (i + 1) == num_proteins:
            print(f"[INFO] Processed {i + 1}/{num_proteins} proteins", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Optional: backup old feature file if exists
    if os.path.exists(out_path):
        backup_path = out_path + ".bak"
        print(f"[INFO] Existing feature file found. Backing up to: {backup_path}")
        os.replace(out_path, backup_path)

    np.save(out_path, feats)
    print(f"[OK] Saved new protein features to: {out_path}")
    print(f"[OK] Shape: {feats.shape}, dtype: {feats.dtype}")


if __name__ == "__main__":
    main()