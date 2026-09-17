"""
Build new drug feature matrix using RDKit Morgan fingerprints.

Inputs:
  - nodes_drug_std.csv       (must contain columns: drug_id, smiles)
  - id_maps/drug2idx.json    (maps drug_id -> row index)

Output:
  - features/drug_feats_256.npy  (shape: [num_drugs, 256], dtype: float32)

Usage example (from GNN/ directory):

  python3 build_drug_feats_rdkit.py \
    --indir GNN_datasets \
    --n_bits 256
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--indir",
        type=str,
        default="GNN_datasets",
        help="Root directory of the GNN dataset (contains nodes_*.csv, id_maps/, features/).",
    )
    ap.add_argument(
        "--n_bits",
        type=int,
        default=256,
        help="Number of bits for the Morgan fingerprint (feature dimension).",
    )
    return ap


def canonical_morgan_fp(smiles: str, n_bits: int) -> np.ndarray:
    """
    Convert a SMILES string into a fixed-length Morgan fingerprint.

    Returns a float32 numpy array of shape [n_bits].
    If parsing fails, returns an all-zero vector.
    """
    arr = np.zeros((n_bits,), dtype=np.float32)
    if not isinstance(smiles, str) or smiles.strip() == "":
        return arr

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Failed to parse SMILES
        return arr

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    # RDKit BitVect -> numpy array
    on_bits = list(fp.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def main():
    args = build_argparser().parse_args()
    indir = args.indir
    n_bits = args.n_bits

    nodes_path = os.path.join(indir, "nodes_drug_std.csv")
    drug2idx_path = os.path.join(indir, "id_maps", "drug2idx.json")
    out_path = os.path.join(indir, "features", f"drug_feats_{n_bits}.npy")

    print(f"[INFO] Reading node table: {nodes_path}")
    df = pd.read_csv(nodes_path)

    required_cols = {"drug_id", "smiles"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"nodes_drug_std.csv is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Reading drug2idx mapping: {drug2idx_path}")
    with open(drug2idx_path, "r") as f:
        drug2idx = json.load(f)

    # Sanity check: the sets of drug_ids should match
    node_ids = set(df["drug_id"].astype(str).tolist())
    map_ids = set(drug2idx.keys())

    only_in_nodes = node_ids - map_ids
    only_in_map = map_ids - node_ids

    if only_in_nodes:
        print(f"[WARN] {len(only_in_nodes)} drug_ids appear in nodes_drug_std.csv "
              f"but not in drug2idx.json (examples: {list(list(only_in_nodes)[:5])})")
    if only_in_map:
        print(f"[WARN] {len(only_in_map)} drug_ids appear in drug2idx.json "
              f"but not in nodes_drug_std.csv (examples: {list(list(only_in_map)[:5])})")

    num_drugs = len(drug2idx)
    print(f"[INFO] num_drugs from drug2idx = {num_drugs}")
    print(f"[INFO] Building feature matrix of shape ({num_drugs}, {n_bits})")

    feats = np.zeros((num_drugs, n_bits), dtype=np.float32)

    # Build a quick lookup: drug_id -> smiles
    id_to_smiles = (
        df[["drug_id", "smiles"]]
        .astype({"drug_id": str})
        .drop_duplicates(subset=["drug_id"])
        .set_index("drug_id")["smiles"]
        .to_dict()
    )

    n_missing_smiles = 0
    n_failed = 0

    for drug_id, idx in drug2idx.items():
        idx = int(idx)
        smiles = id_to_smiles.get(str(drug_id), "")
        if not smiles:
            n_missing_smiles += 1
            feats[idx] = np.zeros((n_bits,), dtype=np.float32)
            continue

        vec = canonical_morgan_fp(smiles, n_bits)
        if vec.sum() == 0.0:
            n_failed += 1
        feats[idx] = vec

    print(f"[INFO] Finished building features.")
    print(f"       Missing/empty SMILES count: {n_missing_smiles}")
    print(f"       SMILES parse failed count:  {n_failed}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Optional: backup old file if exists
    if os.path.exists(out_path):
        backup_path = out_path + ".bak"
        print(f"[INFO] Existing feature file found. Backing up to: {backup_path}")
        os.replace(out_path, backup_path)

    np.save(out_path, feats)
    print(f"[OK] Saved new drug features to: {out_path}")
    print(f"[OK] Shape: {feats.shape}, dtype: {feats.dtype}")


if __name__ == "__main__":
    main()