import os, json, argparse, math, random
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from torch import nn

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:
    average_precision_score = None
    roc_auc_score = None

from torch_geometric.nn import RGCNConv

# ===================== 新增：控制是否使用 drug-disjoint split =====================
USE_DRUG_DISJOINT = True          # 现在我们默认用 drug-disjoint 版本
PAIR_SUFFIX = "_drugdisjoint" if USE_DRUG_DISJOINT else ""
# ======================================================================


# ---------------------------  
# Utils  
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log(s: str):
    print(s, flush=True)


# ---------------------------  
# Data utilities  
# ---------------------------

def load_idx_pairs(path_idx: str, path_raw: str, drug2idx: Dict[str,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prefer *_pairs_idx.csv (a_idx,b_idx,y,weight?),
    otherwise read *_pairs.csv (drug_a,drug_b,y,weight?) and map via drug2idx.
    """
    if os.path.exists(path_idx):
        df = pd.read_csv(path_idx)
        for c in ("a_idx","b_idx","y"):
            if c not in df.columns:
                raise ValueError(f"{path_idx} missing column: {c}")
        a = df["a_idx"].to_numpy(dtype=np.int64)
        b = df["b_idx"].to_numpy(dtype=np.int64)
        y = df["y"].to_numpy(dtype=np.float32)
        w = df["weight"].to_numpy(dtype=np.float32) if "weight" in df.columns else np.ones_like(y, dtype=np.float32)
        return a, b, y, w

    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"neither {path_idx} nor {path_raw} exists.")

    df = pd.read_csv(path_raw)
    for c in ("drug_a","drug_b","y"):
        if c not in df.columns:
            raise ValueError(f"{path_raw} missing column: {c}")
    a = df["drug_a"].astype(str).map(drug2idx)
    b = df["drug_b"].astype(str).map(drug2idx)
    m = a.notna() & b.notna()
    if (~m).any():
        log(f"[WARN] {int((~m).sum())} rows dropped due to missing mapping in drug2idx.")
    df = df[m].copy()
    a = df["drug_a"].astype(str).map(drug2idx).astype(np.int64).to_numpy()
    b = df["drug_b"].astype(str).map(drug2idx).astype(np.int64).to_numpy()
    y = df["y"].to_numpy(dtype=np.float32)
    w = df["weight"].to_numpy(dtype=np.float32) if "weight" in df.columns else np.ones_like(y, dtype=np.float32)
    return a, b, y, w

def build_homo_from_hetero(indir: str, device: torch.device) -> Dict[str, Any]:
    """
    Build a homogeneous graph from (Drug, Protein, Pathway) heterogeneous graph.

      X [N, D], edge_index [2, E], rel_type [E], and offsets/sizes.
    """
    meta = json.load(open(os.path.join(indir, "meta.json"), "r"))
    Nd = int(meta["num_nodes"]["drug"])
    Np = int(meta["num_nodes"]["protein"])
    Nw = int(meta["num_nodes"]["pathway"])

    fd = torch.from_numpy(np.load(os.path.join(indir,"features","drug_feats_256.npy"))).float()
    fp = torch.from_numpy(np.load(os.path.join(indir,"features","protein_feats_256.npy"))).float()
    fw = torch.from_numpy(np.load(os.path.join(indir,"features","pathway_feats_256.npy"))).float()
    assert fd.size(1) == fp.size(1) == fw.size(1), "feature dims mismatch"

    off_d = 0
    off_p = Nd
    off_w = Nd + Np
    total = Nd + Np + Nw

    X = torch.zeros((total, fd.size(1)), dtype=torch.float32)
    X[off_d:off_d+Nd] = fd
    X[off_p:off_p+Np] = fp
    X[off_w:off_w+Nw] = fw

    def load_edges(csv: str, src_off: int, dst_off: int):
        p = os.path.join(indir, "index_edges", csv)
        arr = pd.read_csv(p)[["src_idx","dst_idx","rel_type"]].to_numpy()
        ei = torch.from_numpy(arr[:, :2].T).long()
        ei[0] += src_off
        ei[1] += dst_off
        et = torch.from_numpy(arr[:, 2]).long()
        return ei, et

    dt_ei, dt_et = load_edges("dt.csv",  off_d, off_p)  # D -> T
    pp_ei, pp_et = load_edges("ppw.csv", off_p, off_w)  # T -> P
    ww_ei, ww_et = load_edges("ww.csv",  off_w, off_w)  # P -> P

    mech_path = os.path.join(indir, "index_edges", "mech.csv")
    has_mech = os.path.exists(mech_path)
    if has_mech:
        me_ei, me_et = load_edges("mech.csv", off_d, off_p)
        edge_index = torch.cat([dt_ei, pp_ei, ww_ei, me_ei], dim=1)
        rel_type   = torch.cat([dt_et, pp_et, ww_et, me_et], dim=0)
    else:
        edge_index = torch.cat([dt_ei, pp_ei, ww_ei], dim=1)
        rel_type   = torch.cat([dt_et, pp_et, ww_et], dim=0)

    return {
        "X": X.to(device),
        "edge_index": edge_index.to(device),
        "rel_type": rel_type.to(device),
        "Nd": Nd, "Np": Np, "Nw": Nw,
        "off_d": off_d, "off_p": off_p, "off_w": off_w, "total": total
    }


# ---------------------------  
# Model  
# ---------------------------

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=8, dropout=0.2, use_bn=True, residual=True):
        super().__init__()
        self.conv = RGCNConv(in_dim, out_dim, num_rels, num_bases=num_bases)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x, edge_index, etype):
        h = self.conv(x, edge_index, etype)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            h = h + x
        return h

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim=256, hidden=128, out=128, num_rels=6, num_bases=8, num_layers=2, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out]
        self.layers = nn.ModuleList([
            RGCNLayer(dims[i], dims[i+1], num_rels,
                      num_bases=num_bases, dropout=dropout,
                      use_bn=True, residual=True)
            for i in range(len(dims) - 1)
        ])

    def forward(self, X, edge_index, rel_type):
        h = X
        for g in self.layers:
            h = g(h, edge_index, rel_type)
        return h  # [N_total, out]

class DistMult(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.r, -0.1, 0.1)
    def forward(self, a, b):
        return (a * self.r * b).sum(dim=-1)

class BiAffine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self, a, b):
        return (a @ self.W * b).sum(dim=-1)


# ---------------------------  
# Metrics  
# ---------------------------

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, topk: int = 10) -> Dict[str, float]:
    out = {}
    if average_precision_score is not None:
        out["PR-AUC"] = float(average_precision_score(y_true, scores))
    if roc_auc_score is not None:
        try:
            out["ROC-AUC"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            out["ROC-AUC"] = float("nan")

    # Global approximate Hits@K / MRR (not per-query)
    order = np.argsort(-scores)
    top = y_true[order][:topk]
    out[f"Hits@{topk}"] = float(top.sum() > 0)
    ranks = np.where(y_true[order] == 1)[0]
    out["MRR"] = (1.0 / float(ranks[0] + 1)) if len(ranks) else 0.0
    return out


# ---------------------------  
# Train / Eval  
# ---------------------------

@torch.no_grad()
def score_pairs(Z_drug: torch.Tensor, a_idx_np: np.ndarray, b_idx_np: np.ndarray,
                decoder: nn.Module, device: torch.device) -> np.ndarray:
    a_idx = torch.from_numpy(a_idx_np).long().to(device)
    b_idx = torch.from_numpy(b_idx_np).long().to(device)
    s = decoder(Z_drug[a_idx], Z_drug[b_idx]).detach().cpu().numpy().astype(np.float64)
    return s

def run(indir: str,
        hidden: int, out_dim: int, layers: int,
        num_bases: int, dropout: float,
        lr: float, weight_decay: float,
        epochs: int, eval_every: int,
        decoder_name: str, seed: int,
        export_artifacts: str = None,
        no_train: bool = False,
        eval_only: bool = False,
        encoder_ckpt: str = None,
        patience: int = 6):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    # --- load drug2idx
    with open(os.path.join(indir, "id_maps", "drug2idx.json"), "r") as f:
        drug2idx = {k: int(v) for k, v in json.load(f).items()}

    # --- load pairs（这里改成支持 drug-disjoint 版本）
    train_idx_path = os.path.join(indir, f"train_pairs_idx{PAIR_SUFFIX}.csv")
    val_idx_path   = os.path.join(indir, f"val_pairs_idx{PAIR_SUFFIX}.csv")
    test_idx_path  = os.path.join(indir, f"test_pairs_idx{PAIR_SUFFIX}.csv")

    train_raw_path = os.path.join(indir, "train_pairs.csv")
    val_raw_path   = os.path.join(indir, "val_pairs.csv")
    test_raw_path  = os.path.join(indir, "test_pairs.csv")

    tr_a, tr_b, tr_y, tr_w = load_idx_pairs(train_idx_path, train_raw_path, drug2idx)
    va_a, va_b, va_y, va_w = load_idx_pairs(val_idx_path,   val_raw_path,   drug2idx)
    te_a, te_b, te_y, te_w = load_idx_pairs(test_idx_path,  test_raw_path,  drug2idx)

    # --- class imbalance handling: compute pos_weight
    pos_ratio = float(tr_y.mean())
    neg_ratio = 1.0 - pos_ratio
    pos_weight_value = neg_ratio / max(pos_ratio, 1e-6)
    log(f"[INFO] pos_ratio={pos_ratio:.4f}, pos_weight={pos_weight_value:.4f}")

    # --- build graph tensors
    graph = build_homo_from_hetero(indir, device)
    in_dim = int(graph["X"].size(1))
    num_rels = int(graph["rel_type"].max().item()) + 1
    log(f"[INFO] inferred num_rels={num_rels}")

    # --- model
    enc = RGCNEncoder(in_dim=in_dim, hidden=hidden, out=out_dim,
                      num_rels=num_rels, num_bases=num_bases,
                      num_layers=layers, dropout=dropout).to(device)
    dec = BiAffine(out_dim).to(device) if decoder_name == "biaffine" else DistMult(out_dim).to(device)

    # --- export artifacts (always allowed)
    if export_artifacts:
        pack = {
            "X": graph["X"].detach().cpu(),
            "edge_index": graph["edge_index"].detach().cpu(),
            "rel_type": graph["rel_type"].detach().cpu(),
            "drug2idx": drug2idx,
            "offsets": {"off_d": graph["off_d"], "Nd": graph["Nd"]}
        }
        os.makedirs(os.path.dirname(export_artifacts), exist_ok=True)
        torch.save(pack, export_artifacts)
        log(f"[OK] export artifacts -> {export_artifacts}")

    # --- eval-only path
    if eval_only:
        if encoder_ckpt is None:
            raise ValueError("--eval_only 需要提供 --encoder_ckpt")
        ckpt = torch.load(encoder_ckpt, map_location=device)
        state = ckpt.get("encoder_state_dict", ckpt.get("state_dict", None))
        if state is None and "encoder" in ckpt and hasattr(ckpt["encoder"], "state_dict"):
            state = ckpt["encoder"].state_dict()
        if state is None:
            raise RuntimeError("无法在 checkpoint 中找到 encoder state_dict")

        enc.load_state_dict(state, strict=False)
        enc.eval()
        with torch.no_grad():
            Z = enc(graph["X"], graph["edge_index"], graph["rel_type"])
        Zd = Z[graph["off_d"]: graph["off_d"] + graph["Nd"]]

        val_scores  = score_pairs(Zd, va_a, va_b, dec, device)
        test_scores = score_pairs(Zd, te_a, te_b, dec, device)

        val_metrics  = compute_metrics(va_y, val_scores)
        test_metrics = compute_metrics(te_y, test_scores)
        log(f"[EVAL-ONLY] val: {val_metrics}  test: {test_metrics}")
        return

    if no_train:
        log("[INFO] --no_train: 构图和工件导出完成，不进入训练。")
        return

    # --- optimizer / loss (note: train encoder + decoder together)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # --- early stopping
    best_val = -1.0
    best_state = None
    patience_cnt = 0

    # --- training loop
    tr_a_t = torch.from_numpy(tr_a).long().to(device)
    tr_b_t = torch.from_numpy(tr_b).long().to(device)
    tr_y_t = torch.from_numpy(tr_y).float().to(device)
    tr_w_t = torch.from_numpy(tr_w).float().to(device)

    for ep in range(1, epochs + 1):
        enc.train()
        dec.train()
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            Z = enc(graph["X"], graph["edge_index"], graph["rel_type"])
            Zd = Z[graph["off_d"]: graph["off_d"] + graph["Nd"]]
            logits = dec(Zd[tr_a_t], Zd[tr_b_t])
            loss = (bce(logits, tr_y_t) * tr_w_t).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if ep % eval_every == 0 or ep == epochs:
            enc.eval()
            dec.eval()
            with torch.no_grad():
                Z = enc(graph["X"], graph["edge_index"], graph["rel_type"])
                Zd = Z[graph["off_d"]: graph["off_d"] + graph["Nd"]]
                val_scores  = dec(Zd[torch.from_numpy(va_a).to(device)],
                                  Zd[torch.from_numpy(va_b).to(device)]).sigmoid().cpu().numpy()
                test_scores = dec(Zd[torch.from_numpy(te_a).to(device)],
                                  Zd[torch.from_numpy(te_b).to(device)]).sigmoid().cpu().numpy()

            val_metrics  = compute_metrics(va_y,  val_scores)
            test_metrics = compute_metrics(te_y,  test_scores)
            log(f"[E{ep:03d}] loss={loss.item():.4f}  "
                f"PR-AUC: val={val_metrics.get('PR-AUC', float('nan')):.4f}  "
                f"test={test_metrics.get('PR-AUC', float('nan')):.4f}")

            # early stop on val PR-AUC
            cur = val_metrics.get("PR-AUC", -1.0)
            if cur > best_val:
                best_val = cur
                best_state = enc.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    log(f"[EARLY-STOP] patience reached (best val PR-AUC={best_val:.4f}).")
                    break

    # --- save encoder checkpoint（保存最佳）
    if best_state is not None:
        enc.load_state_dict(best_state, strict=False)
    out_ckpt = os.path.join(indir, "rgcn_encoder.pt")
    torch.save({
        "encoder_state_dict": enc.state_dict(),
        "in_dim": in_dim, "hidden": hidden, "out": out_dim,
        "num_rels": num_rels, "num_bases": num_bases,
        "num_layers": layers, "dropout": dropout
    }, out_ckpt)
    log(f"[OK] saved encoder -> {out_ckpt}")

    # --- final eval with best encoder
    enc.eval()
    dec.eval()
    with torch.no_grad():
        Z = enc(graph["X"], graph["edge_index"], graph["rel_type"])
        Zd = Z[graph["off_d"]: graph["off_d"] + graph["Nd"]]
        val_scores  = dec(Zd[torch.from_numpy(va_a).to(device)],
                          Zd[torch.from_numpy(va_b).to(device)]).sigmoid().cpu().numpy()
        test_scores = dec(Zd[torch.from_numpy(te_a).to(device)],
                          Zd[torch.from_numpy(te_b).to(device)]).sigmoid().cpu().numpy()
    val_metrics  = compute_metrics(va_y,  val_scores)
    test_metrics = compute_metrics(te_y,  test_scores)
    log(f"[BEST] val: {val_metrics}  test: {test_metrics}")

    # --- save scores & labels for offline analysis ---
    out_scores = os.path.join(indir, "ddi_scores_best.npz")
    np.savez(
        out_scores,
        val_a_idx=va_a,
        val_b_idx=va_b,
        val_y=va_y,
        val_scores=val_scores,
        test_a_idx=te_a,
        test_b_idx=te_b,
        test_y=te_y,
        test_scores=test_scores,
    )
    log(f"[OK] saved best val/test scores -> {out_scores}")


# ---------------------------  
# CLI  
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="GNN_datasets", help="数据目录根路径")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out",    type=int, default=128)
    ap.add_argument("--layers", type=int, default=2, help="R-GCN 层数")
    ap.add_argument("--num_bases", type=int, default=8, help="Basis 分解数（需与训练一致）")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--decoder", choices=["distmult","biaffine"], default="distmult")
    ap.add_argument("--seed", type=int, default=42)

    # modes
    ap.add_argument("--export_artifacts", default=None, help="导出评估工件到指定 .pt（含X/edge_index/rel_type/drug2idx/offsets）")
    ap.add_argument("--no_train", action="store_true", help="只构图与导出工件，不训练")
    ap.add_argument("--eval_only", action="store_true", help="仅评估（需提供 --encoder_ckpt）")
    ap.add_argument("--encoder_ckpt", default=None, help="已训练 encoder 的 checkpoint 路径（eval_only 时必需）")

    # early stopping
    ap.add_argument("--patience", type=int, default=6, help="早停耐心（按 val PR-AUC）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(indir=args.indir,
        hidden=args.hidden, out_dim=args.out, layers=args.layers,
        num_bases=args.num_bases, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, eval_every=args.eval_every,
        decoder_name=args.decoder, seed=args.seed,
        export_artifacts=args.export_artifacts,
        no_train=args.no_train,
        eval_only=args.eval_only,
        encoder_ckpt=args.encoder_ckpt,
        patience=args.patience)