

import os, sys, json, argparse, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


try:
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
    from sklearn.manifold import TSNE
except Exception:
    precision_recall_curve = None
    average_precision_score = None
    roc_auc_score = None
    TSNE = None

CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CUR)
if CUR not in sys.path: sys.path.insert(0, CUR)
if ROOT not in sys.path: sys.path.insert(0, ROOT)


class DistMult(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.r, -0.1, 0.1)
    def forward(self, a, b):
        return (a * self.r * b).sum(dim=-1)

# ------------------------------
# Utils
# ------------------------------
def log(msg): print(msg, flush=True)

def load_artifacts(indir):
    p = os.path.join(indir, "eval_artifacts.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}\n"
                                f"python3 GNN/rgcn_runner.py --indir {indir} "
                                f"--export_artifacts {p} --no_train")
    pack = torch.load(p, map_location="cpu")

    for k in ("X","edge_index","rel_type","drug2idx","offsets"):
        if k not in pack:
            raise RuntimeError(f"{p} '{k}'")
    return pack

def infer_num_bases_from_state(state):
    nb = None
    for k,t in state.items():
        if k.endswith("layers.0.conv.weight") and t.ndim==3:
            nb = t.shape[0]
            break
    return nb if nb is not None else 8

def build_encoder_from_ckpt(ckpt_path, rel_type, device):

    from GNN.rgcn_runner import RGCNEncoder  
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state = ckpt.get("encoder_state_dict", ckpt.get("state_dict", None))
        in_dim  = int(ckpt.get("in_dim", 256))
        hidden  = int(ckpt.get("hidden", 128))
        out     = int(ckpt.get("out", 128))
        num_rels= int(ckpt.get("num_rels", int(rel_type.max().item())+1))
        num_bases = int(ckpt.get("num_bases", infer_num_bases_from_state(state or {})))
        num_layers= int(ckpt.get("num_layers", 2))
        dropout  = float(ckpt.get("dropout", 0.2))
    else:
        raise RuntimeError("")
    enc = RGCNEncoder(in_dim=in_dim, hidden=hidden, out=out,
                      num_rels=num_rels, num_bases=num_bases,
                      num_layers=num_layers, dropout=dropout).to(device)
    if state is None:
        raise RuntimeError("encoder_state_dict/state_dict")
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        log(f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:   log("  missing (head): " + ", ".join(missing[:6]))
        if unexpected:log("  unexpected (head): " + ", ".join(unexpected[:6]))
    enc.eval()
    return enc, out

def fetch_pairs(indir, split):

    p_idx = os.path.join(indir, f"{split}_pairs_idx.csv")
    p_raw = os.path.join(indir, f"{split}_pairs.csv")
    if os.path.exists(p_idx):
        df = pd.read_csv(p_idx)
        for c in ("a_idx","b_idx","y"):
            if c not in df.columns: raise ValueError(f"{p_idx} miss {c}")
        return df["a_idx"].to_numpy(dtype=np.int64), df["b_idx"].to_numpy(dtype=np.int64), df["y"].to_numpy(dtype=np.int32), "idx"
    if os.path.exists(p_raw):
        df = pd.read_csv(p_raw)
        for c in ("drug_a","drug_b","y"):
            if c not in df.columns: raise ValueError(f"{p_raw} miss {c}")
        return df["drug_a"].astype(str).to_numpy(), df["drug_b"].astype(str).to_numpy(), df["y"].to_numpy(dtype=np.int32), "id"
    raise FileNotFoundError(f"didn't find {split}{p_idx}  {p_raw}")

def ids_to_idx(arr, drug2idx):
    out = []
    miss = 0
    for s in arr:
        if s in drug2idx:
            out.append(drug2idx[s])
        else:
            out.append(-1); miss += 1
    return np.array(out, dtype=np.int64), miss

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_artifacts(args.indir)
    X = pack["X"].to(device)
    edge_index = pack["edge_index"].to(device)
    rel_type = pack["rel_type"].to(device)
    drug2idx = pack["drug2idx"]
    off_d = int(pack["offsets"]["off_d"]); Nd = int(pack["offsets"]["Nd"])


    enc, emb_dim = build_encoder_from_ckpt(args.encoder_ckpt, rel_type, device)
    with torch.no_grad():
        Z = enc(X, edge_index, rel_type)    # [N_total, D]
    Z_drug = Z[off_d: off_d+Nd].detach()  
    Zd_cpu = Z_drug.cpu().numpy()

  
    deg = np.zeros(Nd, dtype=np.int32)
    ei = edge_index.detach().cpu().numpy()

    src = ei[0]
    m = (src >= off_d) & (src < off_d + Nd)
    drug_src = src[m] - off_d
    if drug_src.size > 0:
        binc = np.bincount(drug_src, minlength=Nd)
        deg[:len(binc)] += binc


    va_a, va_b, va_y, vt = fetch_pairs(args.indir, "val")
    te_a, te_b, te_y, tt = fetch_pairs(args.indir, "test")
    if vt == "id":
        
        va_a2, va_b2 = [], []
        miss = 0
        for ai,bi in zip(va_a, va_b):
            if ai > bi: ai,bi = bi,ai
            va_a2.append(ai); va_b2.append(bi)
        va_a_idx, m1 = ids_to_idx(va_a2, drug2idx)
        va_b_idx, m2 = ids_to_idx(va_b2, drug2idx)
        miss += m1 + m2
        if miss: log(f"[WARN] val has {miss}")
    else:
        va_a_idx, va_b_idx = va_a, va_b

    if tt == "id":
        te_a2, te_b2 = [], []
        miss = 0
        for ai,bi in zip(te_a, te_b):
            if ai > bi: ai,bi = bi,ai
            te_a2.append(ai); te_b2.append(bi)
        te_a_idx, m1 = ids_to_idx(te_a2, drug2idx)
        te_b_idx, m2 = ids_to_idx(te_b2, drug2idx)
        miss += m1 + m2
        if miss: log(f"[WARN] test has {miss} ")
    else:
        te_a_idx, te_b_idx = te_a, te_b

    # 过滤掉 -1
    def filt(a,b,y):
        m = (a>=0) & (b>=0)
        return a[m], b[m], y[m]
    va_a_idx, va_b_idx, va_y = filt(va_a_idx, va_b_idx, va_y)
    te_a_idx, te_b_idx, te_y = filt(te_a_idx, te_b_idx, te_y)

    # 4) 两种打分：Dot & DistMult
    def dot_scores(a_idx, b_idx):
        A = torch.from_numpy(a_idx).long().to(device)
        B = torch.from_numpy(b_idx).long().to(device)
        with torch.no_grad():
            s = (Z_drug[A] * Z_drug[B]).sum(dim=1)
            s = torch.sigmoid(s)
        return s.detach().cpu().numpy()

    dm = DistMult(Z_drug.size(1)).to(device).eval()
    def dm_scores(a_idx, b_idx):
        A = torch.from_numpy(a_idx).long().to(device)
        B = torch.from_numpy(b_idx).long().to(device)
        with torch.no_grad():
            s = dm(Z_drug[A], Z_drug[B])
            s = torch.sigmoid(s)
        return s.detach().cpu().numpy()

    val_dot = dot_scores(va_a_idx, va_b_idx); val_dm = dm_scores(va_a_idx, va_b_idx)
    test_dot= dot_scores(te_a_idx, te_b_idx); test_dm= dm_scores(te_a_idx, te_b_idx)

    # 5) 画 PR 曲线（若 sklearn 可用）
    outdir = os.path.join(args.indir, "viz")
    ensure_dir(os.path.join(outdir, "x.png"))

    if precision_recall_curve is not None and average_precision_score is not None:
        def pr_plot(y, s, title, png):
            P, R, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            plt.figure(figsize=(6,5))
            plt.plot(R, P, lw=2, label=f'AP={ap:.3f}')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(title)
            plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(png, dpi=160); plt.close()
            log(f"[OK] saved {png}")

        pr_plot(va_y, val_dot, "VAL PR Curve (Dot)", os.path.join(outdir, "pr_val_dot.png"))
        pr_plot(va_y, val_dm,  "VAL PR Curve (DistMult)", os.path.join(outdir, "pr_val_distmult.png"))
        pr_plot(te_y, test_dot,"TEST PR Curve (Dot)", os.path.join(outdir, "pr_test_dot.png"))
        pr_plot(te_y, test_dm, "TEST PR Curve (DistMult)", os.path.join(outdir, "pr_test_distmult.png"))
    else:
        log("[WARN] sklearn ")

    # 6) t-SNE（抽样 2000 个药物）
    if TSNE is not None:
        rng = np.random.default_rng(42)
        n = min(2000, Nd)
        idx = rng.choice(Nd, size=n, replace=False)
        emb2d = TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(Zd_cpu[idx])

        plt.figure(figsize=(6,5))
        sc = plt.scatter(emb2d[:,0], emb2d[:,1],
                         c=deg[idx], s=8, alpha=0.8)
        cbar = plt.colorbar(sc)
        cbar.set_label("Drug out-degree (D→T)")
        plt.title("t-SNE of Drug Embeddings (color=degree)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "tsne_drugs.png"), dpi=160); plt.close()
        log(f"[OK] saved {os.path.join(outdir, 'tsne_drugs.png')}")
    else:
        log("[WARN] sklearn.manifold.TSNE ")

    # 7) 相似药物检索 demo（余弦相似度）
    if args.query_id is not None and args.query_idx is not None:
        log("[WARN] ")
    q_idx = None
    if args.query_id is not None:
        if args.query_id in drug2idx:
            q_idx = int(drug2idx[args.query_id])
        else:
            log(f"[WARN]")
    elif args.query_idx is not None:
        if 0 <= args.query_idx < Nd:
            q_idx = int(args.query_idx)
        else:
            log(f"[WARN]。")

    if q_idx is not None:
        q = Z_drug[q_idx].detach().cpu().numpy()
        Q = q / (np.linalg.norm(q) + 1e-9)
        Znorm = Zd_cpu / (np.linalg.norm(Zd_cpu, axis=1, keepdims=True) + 1e-9)
        sims = Znorm @ Q
        order = np.argsort(-sims)
        topk = int(args.topk)
        top_idx = order[:topk+1]  

        idx2drug = {v:k for k,v in drug2idx.items()}
        rows = []
        for j in top_idx:
            dname = idx2drug.get(j, f"IDX_{j}")
            rows.append((j, dname, float(sims[j])))
        df = pd.DataFrame(rows, columns=["drug_idx","drug_id","cosine_sim"])
        out_csv = os.path.join(outdir, f"neighbors_{idx2drug.get(q_idx, f'IDX_{q_idx}')}.csv")
        df.to_csv(out_csv, index=False)
        log(f"[OK] nearest neighbors saved -> {out_csv}")
        log(df.head(min(10, len(df))).to_string(index=False))


    order = np.argsort(-test_dm)
    k = min(1000, len(order))
    top = pd.DataFrame({
        "a_idx": te_a_idx[order][:k],
        "b_idx": te_b_idx[order][:k],
        "y":     te_y[order][:k],
        "score_dm": test_dm[order][:k],
        "score_dot": test_dot[order][:k],
    })
    top_path = os.path.join(outdir, "topk_predictions.csv")
    top.to_csv(top_path, index=False)
    log(f"[OK] top predictions -> {top_path}")

    log("Done.")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="GNN/GNN_datasets", help="")
    ap.add_argument("--encoder_ckpt", default="GNN/GNN_datasets/rgcn_encoder.pt")
    ap.add_argument("--query_id", type=str, default=None, help="")
    ap.add_argument("--query_idx", type=int, default=None, help="")
    ap.add_argument("--topk", type=int, default=15, help="")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)