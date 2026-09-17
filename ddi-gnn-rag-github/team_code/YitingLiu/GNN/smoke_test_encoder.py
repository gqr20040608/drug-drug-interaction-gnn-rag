# """
# smoke_test_encoder.py

# Goal:
#   - Build the homogeneous graph from GNN_datasets
#   - Instantiate RGCNEncoder with correct input dim and num_rels
#   - Run a single forward pass: Z = enc(X, edge_index, rel_type)
#   - Print shapes and basic sanity info

# Run from the GNN folder:

#     cd /mnt/c/Users/jason/OneDrive/Desktop/DS_340W_Project/GNN
#     python3 smoke_test_encoder.py
# """

# import inspect
# from pathlib import Path

# import torch

# # NOTE: we import directly from rgcn_runner to avoid any circular import
# from rgcn_runner import RGCNEncoder, build_homo_from_hetero
# from train_rgcn_baseline import RGCNEncoder, build_homo_from_hetero

# def main():
#     # 1. Print the encoder signature (for sanity)
#     print("=== RGCNEncoder signature ===")
#     sig = inspect.signature(RGCNEncoder.__init__)
#     print(sig)
#     print()

#     # 2. Build graph from GNN_datasets
#     root = Path(__file__).resolve().parent
#     indir = root / "GNN_datasets"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"Loading graph from: {indir}")
#     graph = build_homo_from_hetero(str(indir), device)

#     X = graph["X"]
#     edge_index = graph["edge_index"]
#     rel_type = graph["rel_type"]

#     Nd = graph["Nd"]
#     off_d = graph["off_d"]
#     total = graph["total"]

#     print(f"X shape:          {tuple(X.shape)}")           # [N_total, F]
#     print(f"edge_index shape: {tuple(edge_index.shape)}")  # [2, E]
#     print(f"rel_type shape:   {tuple(rel_type.shape)}")    # [E]
#     print(f"Nd (num drugs):   {Nd}")
#     print(f"total nodes:      {total}")
#     print()

#     # 3. Instantiate encoder with correct in_dim and num_rels
#     in_dim = int(X.size(1))
#     num_rels = int(rel_type.max().item()) + 1  # infer from data

#     print(f"Instantiating RGCNEncoder(in_dim={in_dim}, hidden=128, out=128, "
#           f"num_rels={num_rels}, num_bases=8, num_layers=2, dropout=0.2)")
#     enc = RGCNEncoder(
#         in_dim=in_dim,
#         hidden=128,
#         out=128,
#         num_rels=num_rels,
#         num_bases=8,
#         num_layers=2,
#         dropout=0.2,
#     ).to(device)

#     # 4. Forward pass
#     enc.eval()
#     with torch.no_grad():
#         Z = enc(X, edge_index, rel_type)

#     print()
#     print(f"Z shape:          {tuple(Z.shape)}")  # [N_total, out_dim]
#     Zd = Z[off_d: off_d + Nd]
#     print(f"Z_drug shape:     {tuple(Zd.shape)}")
#     print()

#     # 5. Print a small snippet of embeddings
#     print("First 5 drug embeddings (rows 0–4):")
#     print(Zd[:5].cpu())
#     print("\n[OK] RGCNEncoder forward smoke test completed.")


# if __name__ == "__main__":
#     main()






import inspect
from pathlib import Path

import torch

from train_rgcn_baseline import RGCNEncoder, build_homo_from_hetero


def main():
    # 1. Print the encoder signature (for sanity)
    print("=== RGCNEncoder signature ===")
    sig = inspect.signature(RGCNEncoder.__init__)
    print(sig)
    print()

    # 2. Build graph from GNN_datasets
    root = Path(__file__).resolve().parent
    indir = root / "GNN_datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading graph from: {indir}")
    graph = build_homo_from_hetero(str(indir), device)

    X = graph["X"]
    edge_index = graph["edge_index"]
    rel_type = graph["rel_type"]

    Nd = graph["Nd"]
    off_d = graph["off_d"]
    total = graph["total"]

    print(f"X shape:          {tuple(X.shape)}")           # [N_total, F]
    print(f"edge_index shape: {tuple(edge_index.shape)}")  # [2, E]
    print(f"rel_type shape:   {tuple(rel_type.shape)}")    # [E]
    print(f"Nd (num drugs):   {Nd}")
    print(f"total nodes:      {total}")
    print()

    # 3. Instantiate encoder with correct in_dim and num_rels
    in_dim = int(X.size(1))
    num_rels = int(rel_type.max().item()) + 1  # robust: infer from data

    print(f"Instantiating RGCNEncoder(in_dim={in_dim}, hidden=128, out=128, "
          f"num_rels={num_rels}, num_bases=8, num_layers=2, dropout=0.2)")
    enc = RGCNEncoder(
        in_dim=in_dim,
        hidden=128,
        out=128,
        num_rels=num_rels,
        num_bases=8,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    # 4. Forward pass
    enc.eval()
    with torch.no_grad():
        Z = enc(X, edge_index, rel_type)

    print()
    print(f"Z shape:          {tuple(Z.shape)}")  # [N_total, out_dim]
    Zd = Z[off_d: off_d + Nd]
    print(f"Z_drug shape:     {tuple(Zd.shape)}")
    print()

    # 5. Print a small snippet of embeddings
    print("First 5 drug embeddings (rows 0–4):")
    print(Zd[:5].cpu())
    print("\n[OK] RGCNEncoder forward smoke test completed.")


if __name__ == "__main__":
    main()
