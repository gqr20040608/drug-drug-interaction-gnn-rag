import os, sys, torch

CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CUR)
if CUR not in sys.path: sys.path.insert(0, CUR)
if ROOT not in sys.path: sys.path.insert(0, ROOT)

ENC_PATH = os.path.join(CUR, "GNN_datasets", "rgcn_encoder.pt")
ART_PATH = os.path.join(CUR, "GNN_datasets", "eval_artifacts.pt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ENC_PATH):
        raise FileNotFoundError(f"missing: {ENC_PATH}")

    ckpt = torch.load(ENC_PATH, map_location=device)
    
    from train_rgcn_baseline import RGCNEncoder
    encoder = RGCNEncoder(
    in_dim=256, hidden=128, out=128,
    num_rels=6, num_bases=8,    
    num_layers=2, dropout=0.2   
    ).to(device)

    
    state = None
    if isinstance(ckpt, dict):
        for k in ("encoder_state_dict", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
        if state is None and "encoder" in ckpt and hasattr(ckpt["encoder"], "state_dict"):
            state = ckpt["encoder"].state_dict()
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        raise RuntimeError("cannot find state_dict in checkpoint")

    
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:   print("  missing (head):", missing[:8])
        if unexpected:print("  unexpected (head):", unexpected[:8])

    encoder.eval()
    print("[OK] encoder reconstructed & weights loaded.")
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"[INFO] params: {total_params:,}")

    
    if os.path.exists(ART_PATH):
        pack = torch.load(ART_PATH, map_location=device)
        data = pack.get("data", None)
        if data is None:
            print("[WARN] eval_artifacts.pt has no 'data'; skipping forward.")
            return
        with torch.no_grad():
            try:
                Z = encoder(data.to(device))
            except AttributeError:
                Z = encoder(data)
        if torch.is_tensor(Z):
            print(f"[OK] forward done. Z shape: {tuple(Z.shape)}  dtype: {Z.dtype}")
        else:
            print(f"[OK] forward done. output type: {type(Z)}")
    else:
        print("[INFO] no eval_artifacts.pt; load-only test completed.")

if __name__ == "__main__":
    main()