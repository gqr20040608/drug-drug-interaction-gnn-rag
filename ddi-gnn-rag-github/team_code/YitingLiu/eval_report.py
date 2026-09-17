import argparse, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Matplotlib: headless save (works in servers/WSL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    precision_recall_fscore_support, confusion_matrix
)
from joblib import load


# ========== tiny helpers (boring but essential) ==========

def latest_run_dir(root: Path) -> Path:
    """Pick the newest run folder like v_YYYYMMDD_HHMMSS. If none, tell me nicely."""
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("v_")])
    if not runs:
        raise FileNotFoundError(f"No versioned run dirs under: {root}. "
                                f"Did you run verify_and_split.py?")
    return runs[-1]

def pick_model(run_dir: Path, model_file: str|None) -> Path:
    """We prefer XGB (model_xgb.pkl). If user pins a file, trust them."""
    if model_file:
        p = Path(model_file)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p
    p_xgb = run_dir / "model_xgb.pkl"
    p_lr  = run_dir / "model.pkl"
    if p_xgb.exists(): return p_xgb
    if p_lr.exists():  return p_lr
    raise FileNotFoundError(f"No model_xgb.pkl or model.pkl found in {run_dir}. "
                            f"Run train_xgb.py or train_baseline.py first.")

def load_splits(run_dir: Path):
    """Input CSVs we made during split. If this fails, we broke earlier steps."""
    train = pd.read_csv(run_dir/"train.csv")
    val   = pd.read_csv(run_dir/"val.csv")
    test  = pd.read_csv(run_dir/"test.csv")
    return train, val, test

def load_features_meta(run_dir: Path):
    """We saved FEATURES.json when training. Use it to reproduce the exact preprocessing order."""
    meta = json.loads((run_dir/"FEATURES.json").read_text())
    cont_cols = meta.get("continuous", [])
    cat_cols  = meta.get("bucket_onehot", [])
    if not cont_cols and not cat_cols:
        raise ValueError("FEATURES.json is empty or missing required keys. "
                         "Re-train to regenerate it.")
    return cont_cols, cat_cols

def ensure_reports_dir(run_dir: Path) -> Path:
    out = run_dir / "reports"
    out.mkdir(exist_ok=True)
    return out

def get_pipe_model_steps(pipe):
    """Our model is a sklearn Pipeline: ('pre', ColumnTransformer), ('clf', estimator)."""
    pre = pipe.named_steps.get("pre", None)
    clf = pipe.named_steps.get("clf", None)
    return pre, clf

def feature_names_from_preprocessor(pre, cont_cols, cat_cols):
    """
    Build transformed feature names in order:
    - numeric block keeps original names (after StandardScaler).
    - categorical block expands via OneHotEncoder.categories_ into col=value labels.
    """
    names = []
    if pre is None:
        return cont_cols + cat_cols  # bare model (unlikely here), fallback safely
    if "num" in pre.named_transformers_:
        names += list(cont_cols)
    if "cat" in pre.named_transformers_ and len(cat_cols) > 0:
        enc = pre.named_transformers_["cat"]
        cats = enc.categories_
        # map: each bucket col -> its possible values (0/1/2/3)
        for col, cat_list in zip(cat_cols, cats):
            for v in cat_list:
                names.append(f"{col}={v}")
    return names

def best_threshold_by_f1(y_true, proba):
    """
    We pick threshold on VALIDATION only (no peeking at test).
    Why F1? It balances precision & recall and is robust under class imbalance.
    Feel free to switch to Youden J or best PR-F1 if domain requires.
    """
    grid = np.linspace(0.05, 0.95, 19)
    best = {"thr":0.5, "f1":-1, "prec":0.0, "rec":0.0}
    for t in grid:
        pred = (proba >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"thr":float(t), "f1":float(f1), "prec":float(prec), "rec":float(rec)}
    return best

def metrics_block(y_true, proba, thr):
    """Consistent metric dict we can dump to JSON and compare across runs."""
    auc  = roc_auc_score(y_true, proba)
    aupr = average_precision_score(y_true, proba)
    pred = (proba >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred)
    return {
        "ROC_AUC": float(auc),
        "PR_AUC": float(aupr),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "ConfusionMatrix": cm.tolist(),
        "Threshold": float(thr),
    }

def _bucket_by_quantiles_from_train(train_s: pd.Series, s: pd.Series) -> pd.Series:
    q = train_s.quantile([0.25, 0.5, 0.75]).values
    bins = [-np.inf, q[0], q[1], q[2], np.inf]
    return pd.cut(s, bins=bins, labels=[0,1,2,3]).astype(int)

def ensure_bucket_columns(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                          cont_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    If bucket columns (e.g., 'tani_ecfp4_bucket') are missing, rebuild them
    from the base columns (e.g., 'tani_ecfp4') using TRAIN quantiles.
    We modify & return copies of train/val/test.
    """
    train = train.copy(); val = val.copy(); test = test.copy()

    # map bucket -> base column name (assumes pattern {base}_bucket)
    bucket_bases = {}
    for c in cat_cols:
        if c.endswith("_bucket"):
            base = c[:-7]  # strip "_bucket"
            bucket_bases[c] = base

    # which buckets are missing?
    missing = [c for c in cat_cols if c not in train.columns or c not in val.columns or c not in test.columns]
    if not missing:
        return train, val, test

    for bcol, base in bucket_bases.items():
        if base not in train.columns:
            # can't rebuild if base is absent; skip gracefully
            continue
        # create bucket in all splits using TRAIN quantiles
        train[bcol] = _bucket_by_quantiles_from_train(train[base], train[base])
        if base in val.columns:
            val[bcol]   = _bucket_by_quantiles_from_train(train[base], val[base])
        if base in test.columns:
            test[bcol]  = _bucket_by_quantiles_from_train(train[base], test[base])

    return train, val, test

def _bucket_by_quantiles_from_train(train_s, s):
    q = train_s.quantile([0.25, 0.5, 0.75]).values
    bins = [-np.inf, q[0], q[1], q[2], np.inf]
    return pd.cut(s, bins=bins, labels=[0,1,2,3]).astype(int)

def ensure_bucket_columns(train, val, test, cont_cols, cat_cols):
    train = train.copy(); val = val.copy(); test = test.copy()
    bucket_bases = {c: c[:-7] for c in cat_cols if c.endswith("_bucket")}
    for bcol, base in bucket_bases.items():
        if bcol not in train.columns or bcol not in val.columns or bcol not in test.columns:
            if base not in train.columns:
                continue  
            train[bcol] = _bucket_by_quantiles_from_train(train[base], train[base])
            if base in val.columns:
                val[bcol] = _bucket_by_quantiles_from_train(train[base], val[base])
            if base in test.columns:
                test[bcol] = _bucket_by_quantiles_from_train(train[base], test[base])
    return train, val, test

# ----- plotting helpers (each makes ONE figure; simple, no seaborn magic) -----

def plot_roc(ax, y, proba, title):
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y, proba):.3f}")
    ax.plot([0,1],[0,1],"--", lw=1, label="random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")

def plot_pr(ax, y, proba, title):
    prec, rec, _ = precision_recall_curve(y, proba)
    ax.plot(rec, prec, lw=2, label=f"AUPRC={average_precision_score(y, proba):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left")

def plot_calibration(ax, y, proba, bins=10):
    # Reliability diagram: predicted prob (x) vs observed positive rate (y)
    df = pd.DataFrame({"y": y, "p": proba})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")  # quantile bins → roughly balanced counts
    g = df.groupby("bin")
    x = g["p"].mean(); ycal = g["y"].mean()
    ax.plot([0,1],[0,1],"--", lw=1, label="perfect")
    ax.plot(x, ycal, marker="o", lw=2, label="observed")
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration (Reliability)")
    ax.legend(loc="upper left")

def plot_prob_hist(ax, y, proba):
    ax.hist(proba[y==0], bins=30, alpha=0.5, label="neg")
    ax.hist(proba[y==1], bins=30, alpha=0.5, label="pos")
    ax.legend(); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title("Score distribution by class")

def plot_confusion(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(title)

def plot_lift_gain(ax_lift, ax_gain, y, proba, bins=10):
    """
    Lift/Gain: how well does the ranking concentrate positives at the top?
    Good for business-ish sense-making: "Top 20% captures X% positives".
    """
    df = pd.DataFrame({"y": y, "p": proba}).sort_values("p", ascending=False)
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False) + 1
    grouped = df.groupby("bucket")
    cum_positives = grouped["y"].sum().cumsum()
    total_positives = df["y"].sum()
    total = len(df)

    cum_pct_samples   = (grouped.size().cumsum() / total).values
    cum_pct_positives = (cum_positives / total_positives).values

    # Gain chart
    ax_gain.plot(cum_pct_samples, cum_pct_positives, lw=2, label="model")
    ax_gain.plot([0,1],[0,1],"--", lw=1, label="baseline")
    ax_gain.set_xlabel("Cumulative % of samples"); ax_gain.set_ylabel("Cumulative % of positives")
    ax_gain.set_title("Cumulative Gain"); ax_gain.legend(loc="lower right")

    # Lift chart
    lift = cum_pct_positives / np.clip(cum_pct_samples, 1e-9, 1.0)
    ax_lift.plot(cum_pct_samples, lift, lw=2)
    ax_lift.set_xlabel("Cumulative % of samples"); ax_lift.set_ylabel("Lift")
    ax_lift.set_title("Lift Curve")

def feature_importance(pipe, cont_cols, cat_cols, topk=30):
    """
    Handle both XGB (feature_importances_) and LogisticRegression (coef magnitude).
    We map transformed feature order back to readable names.
    """
    pre, clf = get_pipe_model_steps(pipe)
    names = feature_names_from_preprocessor(pre, cont_cols, cat_cols)

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):         # LR path: use absolute coefficient as importance proxy
        coef = clf.coef_.ravel()
        imp = np.abs(coef)
    else:
        # Not all models expose importances; return zeros to keep UX smooth.
        imp = np.zeros(len(names), dtype=float)

    # guard: sometimes transformers add/drop columns → align lengths
    L = min(len(names), len(imp))
    df = pd.DataFrame({"feature": names[:L], "importance": imp[:L]})
    df = df.sort_values("importance", ascending=False).head(topk)
    return df

def example_pair_predict(pipe, df_source, pair_str, cont_cols, cat_cols):
    """
    Tiny convenience: ask for a real pair like "DB00072,DB00364" and get a probability.
    We look inside the split CSVs (so features/preprocessing match exactly).
    """
    if pair_str is None:
        return None, "No pair provided."
    a, b = [x.strip() for x in pair_str.split(",")]
    # we support either (drug_a,drug_b) or (drug_id_a,drug_id_b)
    for cols in [("drug_a","drug_b"), ("drug_id_a","drug_id_b")]:
        if set(cols).issubset(df_source.columns):
            hit = df_source[(df_source[cols[0]]==a) & (df_source[cols[1]]==b)]
            if not len(hit):
                hit = df_source[(df_source[cols[0]]==b) & (df_source[cols[1]]==a)]
            if len(hit):
                X = hit[cont_cols + cat_cols]
                p = pipe.predict_proba(X)[:,1]
                y = hit["y"].values.tolist()
                return {"pair": f"{a},{b}", "proba": float(p[0]), "y": y[0] if len(y) else None}, None
    return None, f"Pair {a},{b} not found in splits. Tip: check spelling or try another pair."

# ========== main (the path you actually run) ==========

def main():
    # CLI with sane defaults (latest run; prefer XGB)
    ap = argparse.ArgumentParser(description="Evaluate baseline model + plots + PDF report (friendly comments).")
    ap.add_argument("--run-dir", type=str, default=None, help="Path to supervised_v1 run dir (default: latest).")
    ap.add_argument("--model-file", type=str, default=None, help="Specific model .pkl path (overrides auto-pick).")
    ap.add_argument("--pair", type=str, default=None, help="Example pair 'DBxxxx,DByyyy' to score.")
    args = ap.parse_args()

    root = Path("Datasets/release/supervised_v1")
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(root)
    reports_dir = ensure_reports_dir(run_dir)

    # Load model + feature schema (critical for consistent transforms)
    model_path = pick_model(run_dir, args.model_file)
    pipe = load(model_path)
    cont_cols, cat_cols = load_features_meta(run_dir)

    # Snapshot hyperparams (I like to diff these across runs)
    pre, clf = get_pipe_model_steps(pipe)
    hyper = clf.get_params(deep=True)
    (reports_dir/"hyperparams.json").write_text(json.dumps(hyper, indent=2))

    # Read splits
    train, val, test = load_splits(run_dir)
    cont_cols, cat_cols = load_features_meta(run_dir)


    train, val, test = ensure_bucket_columns(train, val, test, cont_cols, cat_cols)
    # Predict probs on each split (no threshold yet; keep it pure)
    tqdm.write("[STAGE] Inference on train/val/test ...")
    pv_tr = pipe.predict_proba(train[cont_cols + cat_cols])[:,1]
    pv_va = pipe.predict_proba(val[cont_cols + cat_cols])[:,1]
    pv_te = pipe.predict_proba(test[cont_cols + cat_cols])[:,1]

    # Choose decision threshold on VALIDATION (max F1) — key to honest evaluation
    best = best_threshold_by_f1(val["y"].values.astype(int), pv_va)
    thr = best["thr"]

    # Compute metrics blocks (train/val/test) at the chosen threshold
    m_train = metrics_block(train["y"].astype(int).values, pv_tr, thr)
    m_val   = metrics_block(val["y"].astype(int).values,   pv_va, thr)
    m_test  = metrics_block(test["y"].astype(int).values,  pv_te, thr)

    # Dump metrics to JSON/TXT for reproducibility + report writing
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "threshold_val_maxF1": best,
        "train": m_train,
        "val":   m_val,
        "test":  m_test,
    }
    (reports_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir/"metrics.txt").write_text(json.dumps(metrics, indent=2))

    # Export ranked test predictions (useful for error analysis / case studies)
    cols_pair = ["drug_a","drug_b"] if {"drug_a","drug_b"}.issubset(test.columns) else ["drug_id_a","drug_id_b"]
    out_pred = test[cols_pair + ["y"]].copy()
    out_pred["proba"] = pv_te
    out_pred = out_pred.sort_values("proba", ascending=False)
    out_pred.to_csv(reports_dir/"top_predictions.csv", index=False)

    # Feature importance (XGB or LR). If nothing available, we still write an empty CSV.
    fi = feature_importance(pipe, cont_cols, cat_cols, topk=30)
    fi.to_csv(reports_dir/"feature_importance.csv", index=False)

    # ---- Make all plots + bundle into one PDF ----
    pdf_path = reports_dir/"report.pdf"
    with PdfPages(pdf_path) as pdf:
        # ROC
        fig, ax = plt.subplots(figsize=(6,5))
        plot_roc(ax, test["y"].values, pv_te, f"ROC (TEST)")
        fig.tight_layout(); fig.savefig(reports_dir/"curves_roc.png", dpi=220); pdf.savefig(fig); plt.close(fig)

        # PR
        fig, ax = plt.subplots(figsize=(6,5))
        plot_pr(ax, test["y"].values, pv_te, f"PR (TEST)")
        fig.tight_layout(); fig.savefig(reports_dir/"curves_pr.png", dpi=220); pdf.savefig(fig); plt.close(fig)

        # Calibration (aka reliability)
        fig, ax = plt.subplots(figsize=(6,5))
        plot_calibration(ax, test["y"].values, pv_te, bins=10)
        fig.tight_layout(); fig.savefig(reports_dir/"calibration.png", dpi=220); pdf.savefig(fig); plt.close(fig)

        # Probability histogram
        fig, ax = plt.subplots(figsize=(6,5))
        plot_prob_hist(ax, test["y"].values, pv_te)
        fig.tight_layout(); fig.savefig(reports_dir/"prob_hist.png", dpi=220); pdf.savefig(fig); plt.close(fig)

        # Confusion matrix @thr
        cm = np.array(m_test["ConfusionMatrix"])
        fig, ax = plt.subplots(figsize=(5,4))
        plot_confusion(ax, cm, f"Confusion @thr={thr:.2f}")
        fig.tight_layout(); fig.savefig(reports_dir/"confusion_matrix.png", dpi=220); pdf.savefig(fig); plt.close(fig)

        # Lift & Gain (ranking view)
        fig, ax = plt.subplots(figsize=(6,4))
        fig2, ax2 = plt.subplots(figsize=(6,4))
        plot_lift_gain(ax, ax2, test["y"].values, pv_te, bins=10)
        fig.tight_layout(); fig.savefig(reports_dir/"lift_curve.png", dpi=220); pdf.savefig(fig); plt.close(fig)
        fig2.tight_layout(); fig2.savefig(reports_dir/"gain_curve.png", dpi=220); pdf.savefig(fig2); plt.close(fig2)

        # Feature importance chart (if something to show)
        if len(fi):
            fig, ax = plt.subplots(figsize=(7,8))
            ax.barh(range(len(fi)), fi["importance"][::-1].values)
            ax.set_yticks(range(len(fi))); ax.set_yticklabels(fi["feature"][::-1].values)
            ax.set_title("Top feature importance")
            fig.tight_layout(); fig.savefig(reports_dir/"feature_importance.png", dpi=240); pdf.savefig(fig); plt.close(fig)

    # Optional: quick example prediction (use test split to guarantee columns/transform)
    if args.pair:
        ex, err = example_pair_predict(pipe, test, args.pair, cont_cols, cat_cols)
        (reports_dir/"example_prediction.txt").write_text(err if err else json.dumps(ex, indent=2))

    # Final console summary — snack-size for eyeballing
    print("\n=== Evaluation Summary ===")
    print(f"Run dir     : {run_dir}")
    print(f"Model       : {model_path.name}")
    print(f"Val BestThr : {metrics['threshold_val_maxF1']}")
    print(f"TEST        : AUC={m_test['ROC_AUC']:.4f}  AUPRC={m_test['PR_AUC']:.4f}  "
          f"F1={m_test['F1']:.4f}  P={m_test['Precision']:.4f}  R={m_test['Recall']:.4f}")
    print(f"Reports saved to: {reports_dir}")
    print(f"PDF: {pdf_path}")

    # --------- extension hooks (future you will thank present you) ---------
    # TODO: add k-fold cross-validation and aggregate metrics (mean±std).
    # TODO: add SHAP value explanations for XGB to show local feature contributions.
    # TODO: add "drug-level split" loader variant for stricter generalization eval.
    # TODO: export calibration curve with isotonic/Platt scaling comparison.

if __name__ == "__main__":
    main()