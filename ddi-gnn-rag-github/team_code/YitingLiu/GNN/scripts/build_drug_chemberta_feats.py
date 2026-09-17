#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_drug_chemberta_feats.py

从 nodes_drug_std.csv 读取药物 + SMILES，
用 HuggingFace 的 ChemBERTa 生成每个药物的 embedding，
输出对齐的 N × D 特征矩阵（.npy），以及一个简单的质量报告（.csv）。

用法示例：
    python build_drug_chemberta_feats.py \
        --csv GNN_datasets/nodes_drug_std.csv \
        --smiles-col smiles \
        --id-col drug_id \
        --out-npy GNN_datasets/features_drug/drug_feats_chemberta_768.npy \
        --out-report GNN_datasets/features_drug/drug_feats_chemberta_report.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ChemBERTa drug embeddings from SMILES."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to nodes_drug_std.csv（药物主表）",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="SMILES 列名（默认：smiles）",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="drug_id",
        help="内部药物 ID 列名（默认：drug_id）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="ChemBERTa 模型名称（HuggingFace hub 路径）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小（默认 64）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="SMILES 序列的最大 token 长度（默认 256）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备：auto / cpu / cuda（默认 auto）",
    )
    parser.add_argument(
        "--out-npy",
        type=str,
        required=True,
        help="输出的 N×D 特征矩阵 .npy 路径",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=None,
        help="可选：输出质量报告 .csv 路径（每个 drug 一行）",
    )
    parser.add_argument(
        "--drop-duplicate-ids",
        action="store_true",
        help="如指定，则按 id-col 去重，仅保留第一行（谨慎使用，可能破坏现有索引对齐）",
    )
    return parser.parse_args()


def choose_device(arg_device: str) -> torch.device:
    if arg_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[WARN] --device cuda 指定但不存在 GPU，回退到 CPU。", file=sys.stderr)
            return torch.device("cpu")
    elif arg_device == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def load_and_clean_drug_table(
    csv_path: str,
    smiles_col: str,
    id_col: str,
    drop_duplicate_ids: bool = False,
) -> pd.DataFrame:
    print(f"[INFO] 读取药物主表: {csv_path}")
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        raise ValueError(f"[ERROR] 找不到 id 列: {id_col}")
    if smiles_col not in df.columns:
        raise ValueError(f"[ERROR] 找不到 SMILES 列: {smiles_col}")

    original_n = len(df)
    print(f"[INFO] 原始药物行数: {original_n}")

    # 可选：按 ID 去重（默认不开，避免破坏现有索引）
    if drop_duplicate_ids:
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        print(
            f"[INFO] 按 {id_col} 去重后行数: {len(df)} "
            f"(去掉 {original_n - len(df)} 行重复)"
        )
    else:
        # 只做一个提醒
        dup_count = df.duplicated(subset=[id_col]).sum()
        if dup_count > 0:
            print(
                f"[WARN] 发现 {dup_count} 个重复 {id_col}（未去重，仅提示）。",
                file=sys.stderr,
            )

    # 统一 SMILES 格式
    df[smiles_col] = df[smiles_col].astype(str).str.strip()

    # RDKit 合法性检查 + canonical SMILES
    smiles_clean = []
    is_valid = []
    invalid_reason = []

    print("[INFO] 使用 RDKit 校验和 canonical SMILES...")
    for s in tqdm(df[smiles_col].tolist(), desc="RDKit parsing", ncols=100):
        if s == "" or s.lower() == "nan":
            smiles_clean.append("")
            is_valid.append(False)
            invalid_reason.append("empty_or_nan")
            continue

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            smiles_clean.append("")
            is_valid.append(False)
            invalid_reason.append("rdkit_parse_fail")
        else:
            can_s = Chem.MolToSmiles(mol, canonical=True)
            smiles_clean.append(can_s)
            is_valid.append(True)
            invalid_reason.append("ok")

    df["smiles_clean"] = smiles_clean
    df["is_valid_smiles"] = is_valid
    df["invalid_reason"] = invalid_reason

    n_valid = sum(is_valid)
    n_invalid = len(df) - n_valid
    print(
        f"[INFO] RDKit 校验完成：合法 {n_valid} 条，非法/缺失 {n_invalid} 条。"
    )

    return df


def build_chemberta_embeddings(
    df: pd.DataFrame,
    smiles_col_clean: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    print(f"[INFO] 加载 ChemBERTa 模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    smiles_list = df[smiles_col_clean].tolist()
    n = len(smiles_list)

    all_vecs = []
    first_batch_hidden_size = None

    print(
        f"[INFO] 开始生成 ChemBERTa embedding，总计 {n} 个药物，batch_size={batch_size}。"
    )

    for start in tqdm(range(0, n, batch_size), desc="ChemBERTa", ncols=100):
        end = min(start + batch_size, n)
        batch_smiles = smiles_list[start:end]

        # 即使是空字符串也直接喂给 tokenizer，后面用 mask 统一置零
        inputs = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 简单做 mean pooling: 对 last_hidden_state 在 token 维度上求平均
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        emb = last_hidden.mean(dim=1)  # [B, H]

        emb_np = emb.cpu().numpy()
        if first_batch_hidden_size is None:
            first_batch_hidden_size = emb_np.shape[1]

        all_vecs.append(emb_np)

    feats = np.vstack(all_vecs)  # (N, H)
    assert feats.shape[0] == n, "Embedding 行数和药物数量不一致！"

    print(f"[INFO] 生成 embedding 矩阵形状: {feats.shape} (N, H)")
    return feats


def apply_invalid_mask(feats: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    对非法 / 缺失 SMILES 的行，用全 0 向量占位。
    """
    mask_valid = df["is_valid_smiles"].to_numpy().astype(bool)
    n_invalid = (~mask_valid).sum()
    if n_invalid > 0:
        print(
            f"[INFO] 将 {n_invalid} 条非法/缺失 SMILES 的 embedding 置为全 0。"
        )
        feats[~mask_valid, :] = 0.0
    else:
        print("[INFO] 所有 SMILES 都合法，无需置零处理。")
    return feats


def save_outputs(
    feats: np.ndarray,
    df: pd.DataFrame,
    out_npy: str,
    out_report: str | None,
    id_col: str,
):
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, feats)
    print(f"[INFO] 已保存 embedding 矩阵到: {out_npy}")

    if out_report is not None:
        os.makedirs(os.path.dirname(out_report), exist_ok=True)
        report_cols = [id_col, "smiles_clean", "is_valid_smiles", "invalid_reason"]
        # 如果 id_col 不在这些列中，说明用户改了列名，仍然按传入的 id_col 存
        report_cols = [c for c in report_cols if c in df.columns]

        df[report_cols].to_csv(out_report, index=False)
        print(f"[INFO] 已保存质量报告到: {out_report}")


def main():
    args = parse_args()
    device = choose_device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 1. 读取 & 清洗药物表（不改变原始行顺序）
    df = load_and_clean_drug_table(
        csv_path=args.csv,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        drop_duplicate_ids=args.drop_duplicate_ids,
    )

    # 2. ChemBERTa 生成 embedding（按 df 当前行顺序）
    feats = build_chemberta_embeddings(
        df=df,
        smiles_col_clean="smiles_clean",
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # 3. 对非法/缺失 SMILES 用全 0 占位，保证行数对齐
    feats = apply_invalid_mask(feats, df)

    # 4. 保存 .npy + 报告
    save_outputs(
        feats=feats,
        df=df,
        out_npy=args.out_npy,
        out_report=args.out_report,
        id_col=args.id_col,
    )

    print("[INFO] 全部完成 ✅")


if __name__ == "__main__":
    main()