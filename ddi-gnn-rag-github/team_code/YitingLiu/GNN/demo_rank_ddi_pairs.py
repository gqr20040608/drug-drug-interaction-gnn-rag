"""
Demo script: rank candidate DDI partners for a given drug using a trained RGCN encoder.

This script:
  1) Builds the graph tensors from {indir} (same layout as rgcn_runner.py)
  2) Loads a trained RGCNEncoder from rgcn_encoder.pt
  3) Computes drug embeddings Z_drug
  4) For a given anchor drug, scores all possible partner drugs and prints top-K

Usage examples (from GNN/ directory):

  # Rank partners for a specific drug_id (must exist in drug2idx.json)
  python3 demo_rank_ddi_pairs.py \
    --indir GNN_datasets \
    --encoder_ckpt GNN_datasets/rgcn_encoder.pt \
    --drug_id DB00331 \
    --top_k 20

  # Or, rank partners for a given drug index (0-based)
  python3 demo_rank_ddi_pairs.py \
    --indir GNN_datasets \
    --encoder_ckpt GNN_datasets/rgcn_encoder.pt \
    --drug_idx 123 \
    --top_k 20
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import torch

# We reuse the encoder + graph builder from rgcn_runner.py
from rgcn_runner import RGCNEncoder, DistMult, BiAffine, build_homo_from_hetero


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--indir",
        type=str,
        default="GNN_datasets",
        help="Root directory of the GNN dataset (same as in rgcn_runner.py).",
    )
    ap.add_argument(
        "--encoder_ckpt",
        type=str,
        default="GNN_datasets/rgcn_encoder.pt",
        help="Path to trained RGCN encoder checkpoint.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--drug_id",
        type=str,
        help="Anchor drug_id (must be a key in id_maps/drug2idx.json).",
    )
    group.add_argument(
        "--drug_idx",
        type=int,
        help="Anchor drug index (0-based, must be < Nd).",
    )
    ap.add_argument(
        "--decoder",
        type=str,
        choices=["distmult", "biaffine"],
        default="distmult",
        help="Decoder form used for scoring (must match training choice logically).",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top partner drugs to display.",
    )
    return ap.parse_args()


def load_drug_metadata(indir: str):
    """
    Load nodes_drug_std.csv and drug2idx.json so we can map between:
      - drug_id <-> index
      - index -> human-readable name / InChIKey

    Returns:
      drug2idx: dict[str, int]
      idx2drug: list[str] (index -> drug_id)
      meta_df:  pandas DataFrame indexed by drug_id (for names etc.)
    """
    nodes_path = os.path.join(indir, "nodes_drug_std.csv")
    mapping_path = os.path.join(indir, "id_maps", "drug2idx.json")

    print(f"[INFO] Reading drug node table: {nodes_path}")
    df = pd.read_csv(nodes_path)

    required_cols = {"drug_id", "name", "inchikey"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"[WARN] nodes_drug_std.csv is missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    print(f"[INFO] Reading drug2idx mapping: {mapping_path}")
    with open(mapping_path, "r") as f:
        drug2idx = {k: int(v) for k, v in json.load(f).items()}

    # DataFrame indexed by drug_id for fast lookup
    df = df.astype({"drug_id": str}).drop_duplicates(subset=["drug_id"])
    meta_df = df.set_index("drug_id")

    # Build inverse mapping: index -> drug_id
    idx2drug = [None] * len(drug2idx)
    for d, i in drug2idx.items():
        if i < 0 or i >= len(idx2drug):
            raise ValueError(f"Invalid index {i} in drug2idx for drug_id={d}")
        idx2drug[i] = d

    return drug2idx, idx2drug, meta_df


def build_encoder_from_ckpt(encoder_ckpt: str, graph, decoder_name: str):
    """
    Build an RGCNEncoder with the same hyperparameters as training and load weights
    from the given checkpoint. Also build a fresh decoder for scoring.
    """
    device = graph["X"].device
    ckpt = torch.load(encoder_ckpt, map_location=device)

    in_dim = ckpt.get("in_dim", int(graph["X"].size(1)))
    hidden = ckpt.get("hidden", 128)
    out_dim = ckpt.get("out", 128)
    num_rels = ckpt.get("num_rels", int(graph["rel_type"].max().item()) + 1)
    num_bases = ckpt.get("num_bases", 8)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.2)

    print("[INFO] Building RGCNEncoder with hyperparameters:")
    print(f"       in_dim={in_dim}, hidden={hidden}, out={out_dim}, "
          f"num_rels={num_rels}, num_bases={num_bases}, "
          f"num_layers={num_layers}, dropout={dropout}")

    enc = RGCNEncoder(
        in_dim=in_dim,
        hidden=hidden,
        out=out_dim,
        num_rels=num_rels,
        num_bases=num_bases,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state = ckpt.get("encoder_state_dict", ckpt.get("state_dict", None))
    if state is None:
        raise RuntimeError(
            "Could not find encoder_state_dict in checkpoint. "
            "Make sure rgcn_runner.py saved the file."
        )

    enc.load_state_dict(state, strict=False)
    enc.eval()

    # Build decoder for scoring. This decoder is not trained, but for demo ranking
    # it is sufficient to induce a scoring function over embeddings.
    if decoder_name == "biaffine":
        dec = BiAffine(out_dim).to(device)
    else:
        dec = DistMult(out_dim).to(device)

    return enc, dec


@torch.no_grad()
def compute_drug_embeddings(enc: RGCNEncoder, graph):
    """
    Run the encoder on the full graph and extract the drug sub-embeddings.

    Returns:
      Z_drug: tensor [Nd, d]
    """
    X = graph["X"]
    edge_index = graph["edge_index"]
    rel_type = graph["rel_type"]

    print("[INFO] Running encoder to compute node embeddings...")
    Z = enc(X, edge_index, rel_type)  # [N_total, d]

    off_d = graph["off_d"]
    Nd = graph["Nd"]
    Z_drug = Z[off_d : off_d + Nd].contiguous()
    print(f"[INFO] Z_drug shape: {tuple(Z_drug.shape)}")
    return Z_drug


@torch.no_grad()
def rank_partners_for_drug(
    Z_drug: torch.Tensor,
    anchor_idx: int,
    dec: torch.nn.Module,
    top_k: int = 20,
):
    """
    Score all partner drugs for a given anchor drug index and return sorted results.

    Returns:
      indices: numpy array of shape [top_k] with partner indices
      scores:  numpy array of shape [top_k] with scores
    """
    device = Z_drug.device
    Nd, dim = Z_drug.shape

    if anchor_idx < 0 or anchor_idx >= Nd:
        raise ValueError(f"anchor_idx {anchor_idx} is out of range [0, {Nd})")

    anchor_vec = Z_drug[anchor_idx].unsqueeze(0)  # [1, d]
    # Broadcast anchor to all pairs
    anchor_mat = anchor_vec.expand(Nd, dim)       # [Nd, d]

    # Score with the decoder
    logits = dec(anchor_mat, Z_drug)              # [Nd]
    scores = torch.sigmoid(logits).detach().cpu().numpy()

    # Do not rank self
    scores[anchor_idx] = -np.inf

    order = np.argsort(-scores)  # descending
    top_indices = order[:top_k]
    top_scores = scores[top_indices]

    return top_indices, top_scores


def pretty_print_results(
    anchor_id: str,
    anchor_name: str,
    top_indices,
    top_scores,
    idx2drug,
    meta_df: pd.DataFrame,
):
    print("\n====================")
    print(f"Anchor drug: {anchor_id}  ({anchor_name})")
    print("====================")
    print(f"{'Rank':>4}  {'drug_idx':>8}  {'drug_id':>15}  {'name':<40}  {'score':>8}")
    print("-" * 90)

    for rank, (idx, s) in enumerate(zip(top_indices, top_scores), start=1):
        d_id = idx2drug[idx]
        if d_id in meta_df.index:
            row = meta_df.loc[d_id]
            name = str(row.get("name", ""))
        else:
            name = ""
        print(f"{rank:>4}  {idx:>8}  {d_id:>15}  {name[:38]:<40}  {s:8.4f}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load drug metadata (id <-> idx, names)
    drug2idx, idx2drug, meta_df = load_drug_metadata(args.indir)
    Nd = len(idx2drug)
    print(f"[INFO] Number of drug nodes (Nd): {Nd}")

    # 2) Determine anchor_idx from either drug_id or drug_idx
    if args.drug_id is not None:
        d_id = args.drug_id
        if d_id not in drug2idx:
            raise ValueError(
                f"drug_id '{d_id}' not found in drug2idx.json."
            )
        anchor_idx = drug2idx[d_id]
        anchor_id = d_id
    else:
        anchor_idx = args.drug_idx
        if anchor_idx < 0 or anchor_idx >= Nd:
            raise ValueError(
                f"drug_idx {anchor_idx} out of range [0, {Nd})."
            )
        anchor_id = idx2drug[anchor_idx]

    if anchor_id in meta_df.index:
        anchor_name = str(meta_df.loc[anchor_id].get("name", ""))
    else:
        anchor_name = ""

    print(f"[INFO] Anchor drug_id={anchor_id}, anchor_idx={anchor_idx}, name={anchor_name}")

    # 3) Build graph tensors (reuse helper from rgcn_runner.py)
    print("[INFO] Building graph tensors from directory:", args.indir)
    graph = build_homo_from_hetero(args.indir, device)

    # Sanity check: graph["Nd"] should match Nd from drug2idx
    if graph["Nd"] != Nd:
        print(
            f"[WARN] Nd mismatch: graph['Nd']={graph['Nd']} vs len(drug2idx)={Nd}. "
            "Make sure your dataset is consistent."
        )

    # 4) Build encoder + decoder, load encoder weights
    enc, dec = build_encoder_from_ckpt(args.encoder_ckpt, graph, args.decoder)

    # 5) Compute drug embeddings
    Z_drug = compute_drug_embeddings(enc, graph)

    # 6) Rank partners
    top_indices, top_scores = rank_partners_for_drug(
        Z_drug=Z_drug,
        anchor_idx=anchor_idx,
        dec=dec,
        top_k=args.top_k,
    )

    # 7) Pretty-print results
    pretty_print_results(
        anchor_id=anchor_id,
        anchor_name=anchor_name,
        top_indices=top_indices,
        top_scores=top_scores,
        idx2drug=idx2drug,
        meta_df=meta_df,
    )


if __name__ == "__main__":
    main()