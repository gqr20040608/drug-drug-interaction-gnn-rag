import os
import numpy as np
import argparse
from collections import defaultdict

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt


def log(msg: str):
    print(msg, flush=True)


def load_scores(indir: str):
    path = os.path.join(indir, "ddi_scores_best.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Did you run train_rgcn_baseline.py and save ddi_scores_best.npz?"
        )
    data = np.load(path)
    return data


# ---------- 1. 阈值相关指标 ----------

def scan_thresholds(y, scores, split_name="test"):
    """
    扫描一系列阈值，计算不同 operating point 下的指标，
    并挑出：best-F1 / high-precision / high-recall 三个代表点。
    """
    log(f"\n=== [{split_name.upper()}] Threshold scan ===")

    # 用 PR 曲线上的阈值作为候选点
    precisions, recalls, thresholds = precision_recall_curve(y, scores)
    # 注意：precision/recall 比 thresholds 多 1 个点，这里直接按长度截齐
    n_thr = len(thresholds)
    precisions = precisions[:n_thr]
    recalls = recalls[:n_thr]

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)

    # 1) best-F1
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    # 2) high-precision (P >= 0.9) 中 F1 最大的点
    mask_hp = precisions >= 0.90
    hp_thr = None
    if mask_hp.any():
        idxs = np.where(mask_hp)[0]
        idx_hp = idxs[int(np.nanargmax(f1s[idxs]))]
        hp_thr = float(thresholds[idx_hp])

    # 3) high-recall (R >= 0.9) 中 F1 最大的点
    mask_hr = recalls >= 0.90
    hr_thr = None
    if mask_hr.any():
        idxs = np.where(mask_hr)[0]
        idx_hr = idxs[int(np.nanargmax(f1s[idxs]))]
        hr_thr = float(thresholds[idx_hr])

    def report_at_thr(name, thr):
        if thr is None:
            log(f"[{name}] no threshold satisfies the condition.")
            return None

        y_pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        acc = (tp + tn) / (tp + tn + fp + fn)

        log(f"\n[{name}] threshold = {thr:.4f}")
        log(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        log(f"  Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
        return {
            "thr": thr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "acc": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    best_info = report_at_thr("Best-F1", best_thr)
    hp_info = report_at_thr("High-Precision (P>=0.9)", hp_thr)
    hr_info = report_at_thr("High-Recall (R>=0.9)", hr_thr)

    return {
        "best": best_info,
        "high_precision": hp_info,
        "high_recall": hr_info,
    }


def summarize_global_metrics(y, scores, split_name="test"):
    """整体 ROC-AUC / PR-AUC + best-F1 operating point。"""
    log(f"\n=== [{split_name.upper()}] Global metrics ===")
    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)
    log(f"ROC-AUC: {roc:.4f}")
    log(f"PR-AUC : {pr:.4f}")

    # 顺带跑一遍阈值扫描（里面会打印 best-F1 等信息）
    thr_info = scan_thresholds(y, scores, split_name=split_name)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "thresholds": thr_info,
    }


# ---------- 2. Top-K 排名指标 ----------

def precision_recall_at_k(y, scores, ks=(100, 500, 1000), split_name="test"):
    """全局排序下的 Precision@K & Recall@K。"""
    order = np.argsort(-scores)
    y_sorted = y[order]
    n = len(y)
    total_pos = y.sum()

    log(f"\n=== [{split_name.upper()}] Precision@K / Recall@K (global ranking) ===")
    for k in ks:
        k = min(k, n)
        if k <= 0:
            continue
        topk = y_sorted[:k]
        hits = topk.sum()
        p_at_k = hits / k
        r_at_k = hits / (total_pos + 1e-12)
        log(f"@{k:4d}: Precision={p_at_k:.4f}, Recall={r_at_k:.4f}, Hits={int(hits)}")


# ---------- 3. 以 anchor drug 为 query 的 MAP / Recall@K ----------

def per_anchor_map_recall_at_k(a_idx, y, scores, ks=(5, 10, 20), split_name="test"):
    """
    以每个 anchor drug 为查询，计算：
      - 该 anchor 的 AP（average precision）
      - 该 anchor 的 Recall@K
    最后在所有 anchor 上取平均，得到 MAP 和 mean Recall@K。
    """
    log(f"\n=== [{split_name.upper()}] Per-anchor MAP & Recall@K ===")

    groups = defaultdict(list)
    for a, label, score in zip(a_idx, y, scores):
        groups[int(a)].append((score, label))

    ap_list = []
    recall_at_k = {K: [] for K in ks}

    for anchor, items in groups.items():
        labels = np.array([lab for (_, lab) in items], dtype=int)
        scores_anchor = np.array([sc for (sc, _) in items], dtype=float)

        num_pos = labels.sum()
        if num_pos == 0:
            # 这个 anchor 在该 split 里没有正样本，跳过
            continue

        order = np.argsort(-scores_anchor)
        labels_sorted = labels[order]

        # AP
        cum_tp = 0
        prec_sum = 0.0
        for rank, rel in enumerate(labels_sorted, start=1):
            if rel == 1:
                cum_tp += 1
                prec_sum += cum_tp / rank
        ap = prec_sum / num_pos
        ap_list.append(ap)

        # Recall@K
        for K in ks:
            eff_k = min(K, len(labels_sorted))
            if eff_k == 0:
                continue
            topk = labels_sorted[:eff_k]
            recall_k = topk.sum() / num_pos
            recall_at_k[K].append(recall_k)

    if not ap_list:
        log("No anchors with positive labels in this split; cannot compute MAP / Recall@K.")
        return

    map_score = float(np.mean(ap_list))
    log(f"MAP (mean AP over anchors): {map_score:.4f}")
    for K in ks:
        values = recall_at_k[K]
        if not values:
            continue
        r = float(np.mean(values))
        log(f"Mean Recall@{K} over anchors: {r:.4f}")


# ---------- 4. 曲线 & 校准可视化 ----------

def plot_curves_and_calibration(y, scores, split_name="test", outdir="viz", tag: str = ""):
    os.makedirs(outdir, exist_ok=True)

    prefix = f"{tag}_" if tag else ""

    # ROC
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({split_name})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(outdir, f"{prefix}roc_{split_name}.png")
    plt.savefig(roc_path)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = average_precision_score(y, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, lw=2, label=f"AP={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR ({split_name})")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(outdir, f"{prefix}pr_{split_name}.png")
    plt.savefig(pr_path)
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", lw=2, label=f"Brier={brier:.4f}")
    plt.plot([0, 1], [0, 1], "--", lw=1, label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"Calibration ({split_name})")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    calib_path = os.path.join(outdir, f"{prefix}calib_{split_name}.png")
    plt.savefig(calib_path)
    plt.close()

    log(f"\n[{split_name.upper()}] Curves saved to {outdir}:")
    log(f"  ROC  -> {roc_path}")
    log(f"  PR   -> {pr_path}")
    log(f"  CAL  -> {calib_path}")


# ---------- 5. 主函数 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="GNN_datasets",
                    help="Directory containing ddi_scores_best.npz")
    ap.add_argument("--outdir", default="viz_fused",
                    help="Directory to save visualizations (default: viz_fused)")
    ap.add_argument("--tag", default="fused_esm_drugdisjoint",
                    help="Tag prefix for output image filenames")
    args = ap.parse_args()

    data = load_scores(args.indir)

    # unpack
    val_y = data["val_y"]
    val_scores = data["val_scores"]
    val_a_idx = data["val_a_idx"]

    test_y = data["test_y"]
    test_scores = data["test_scores"]
    test_a_idx = data["test_a_idx"]

    # 1) global metrics (val/test) + 阈值扫描
    summarize_global_metrics(val_y, val_scores, split_name="val")
    summarize_global_metrics(test_y, test_scores, split_name="test")

    # 2) global ranking Precision@K + Recall@K（只在 test 上看就够了）
    precision_recall_at_k(test_y, test_scores, ks=(100, 500, 1000), split_name="test")

    # 3) per-anchor ranking metrics (MAP, Recall@K)
    per_anchor_map_recall_at_k(val_a_idx, val_y, val_scores, ks=(5, 10, 20), split_name="val")
    per_anchor_map_recall_at_k(test_a_idx, test_y, test_scores, ks=(5, 10, 20), split_name="test")

    # 4) ROC / PR / Calibration 曲线（val + test）
    plot_curves_and_calibration(val_y, val_scores, split_name="val",
                                outdir=args.outdir, tag=args.tag)
    plot_curves_and_calibration(test_y, test_scores, split_name="test",
                                outdir=args.outdir, tag=args.tag)


if __name__ == "__main__":
    main()


# ---------- 5. 主函数 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="GNN_datasets",
                    help="Directory containing ddi_scores_best.npz")
    args = ap.parse_args()

    data = load_scores(args.indir)

    # unpack
    val_y = data["val_y"]
    val_scores = data["val_scores"]
    val_a_idx = data["val_a_idx"]

    test_y = data["test_y"]
    test_scores = data["test_scores"]
    test_a_idx = data["test_a_idx"]

    # 1) global metrics (val/test) + 阈值扫描
    summarize_global_metrics(val_y, val_scores, split_name="val")
    summarize_global_metrics(test_y, test_scores, split_name="test")

    # 2) global ranking Precision@K + Recall@K（只在 test 上看就够了）
    precision_recall_at_k(test_y, test_scores, ks=(100, 500, 1000), split_name="test")

    # 3) per-anchor ranking metrics (MAP, Recall@K)
    per_anchor_map_recall_at_k(val_a_idx, val_y, val_scores, ks=(5, 10, 20), split_name="val")
    per_anchor_map_recall_at_k(test_a_idx, test_y, test_scores, ks=(5, 10, 20), split_name="test")

    # 4) ROC / PR / Calibration 曲线（val + test）
    plot_curves_and_calibration(val_y, val_scores, split_name="val", outdir="viz")
    plot_curves_and_calibration(test_y, test_scores, split_name="test", outdir="viz")


if __name__ == "__main__":
    main()