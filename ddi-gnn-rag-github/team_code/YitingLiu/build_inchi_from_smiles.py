#Compute/Fill InChI and InChIKey from SMILES using RDKit, robust & chunked.


import argparse
import sys
import math
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi, MolToInchiKey

def compute_inchi_from_smiles(smiles: str):
    """Return (inchi, inchikey) or (None, None) if failed."""
    if not isinstance(smiles, str) or smiles.strip() == "" or smiles.strip().upper() == "NA":
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None, None
        # 标准 InChI（默认选项）
        inchi = MolToInchi(mol)
        inchikey = MolToInchiKey(mol)
        return inchi, inchikey
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input CSV path (e.g., drugs_master_clean.csv)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    ap.add_argument("--inchi-col", default="InChI", help="existing InChI column name")
    ap.add_argument("--inchikey-col", default="InChIKey", help="existing InChIKey column name")
    ap.add_argument("--id-col", default=None, help="Drug ID column (e.g., DrugBank_ID), optional")
    ap.add_argument("--mode", choices=["fill", "recompute"], default="fill",
                   help="'fill' only fills missing; 'recompute' overwrites all InChI/Key")
    ap.add_argument("--chunksize", type=int, default=50000, help="Read/process in chunks")
    ap.add_argument("--dedupe", action="store_true", help="Also write potential duplicates by InChIKey")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 统计
    total = 0
    computed = 0
    filled_inchi = 0
    filled_key = 0
    failed = 0

    dup_records = []  # for dedupe reporting

    # 首次写出时带表头，后续 append
    first_chunk = True

    print(f"[INFO] Input: {inp}")
    print(f"[INFO] Output: {out}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Chunk size: {args.chunksize}")

    # 逐块读取
    for df in pd.read_csv(inp, chunksize=args.chunksize):
        total += len(df)

        # 确保列存在
        for col, default in [(args.smiles_col, None), (args.inchi_col, None), (args.inchikey_col, None)]:
            if col not in df.columns:
                # 若缺列则创建空列
                df[col] = pd.Series([pd.NA]*len(df), index=df.index)

        # 结果列（新计算值）
        new_inchi = []
        new_inchikey = []
        ok_mask = []

        # 逐行计算
        for i, row in df.iterrows():
            smi = row[args.smiles_col]
            inchi_old = row[args.inchi_col]
            key_old = row[args.inchikey_col]

            # 是否需要计算？
            need_inchi = (args.mode == "recompute") or pd.isna(inchi_old) or str(inchi_old).strip() in ("", "NA")
            need_key   = (args.mode == "recompute") or pd.isna(key_old) or str(key_old).strip() in ("", "NA")

            if need_inchi or need_key:
                inchi_new, key_new = compute_inchi_from_smiles(smi)
                if inchi_new is None or key_new is None:
                    failed += 1
                    new_inchi.append(pd.NA)
                    new_inchikey.append(pd.NA)
                    ok_mask.append(False)
                else:
                    computed += 1
                    new_inchi.append(inchi_new)
                    new_inchikey.append(key_new)
                    ok_mask.append(True)
            else:
                new_inchi.append(pd.NA)
                new_inchikey.append(pd.NA)
                ok_mask.append(True)  # 不需要计算也视为“OK”

        df["_new_inchi"] = new_inchi
        df["_new_inchikey"] = new_inchikey

        # 填充或覆盖
        if args.mode == "recompute":
            df[args.inchi_col] = df["_new_inchi"]
            df[args.inchikey_col] = df["_new_inchikey"]
            filled_inchi += df["_new_inchi"].notna().sum()
            filled_key += df["_new_inchikey"].notna().sum()
        else:
            # fill 模式：只在原值缺失时用新值补
            need_inchi_mask = df[args.inchi_col].isna() | (df[args.inchi_col].astype(str).str.strip().isin(["", "NA"]))
            need_key_mask   = df[args.inchikey_col].isna() | (df[args.inchikey_col].astype(str).str.strip().isin(["", "NA"]))

            df.loc[need_inchi_mask & df["_new_inchi"].notna(), args.inchi_col] = df["_new_inchi"]
            df.loc[need_key_mask & df["_new_inchikey"].notna(), args.inchikey_col] = df["_new_inchikey"]

            filled_inchi += (need_inchi_mask & df["_new_inchi"].notna()).sum()
            filled_key += (need_key_mask & df["_new_inchikey"].notna()).sum()

        # 可选：统计潜在重复（同一 InChIKey 出现多次）
        if args.dedupe:
            key_series = df[args.inchikey_col].astype(str).str.strip()
            dup_keys = key_series[key_series != ""].value_counts()
            dup_keys = dup_keys[dup_keys > 1]
            if len(dup_keys) > 0:
                # 记录重复的行（简要）
                idcol = args.id_col if args.id_col and args.id_col in df.columns else None
                for k in dup_keys.index.tolist():
                    rows = df[df[args.inchikey_col].astype(str).str.strip() == k]
                    items = rows[idcol].astype(str).tolist() if idcol else rows.index.astype(str).tolist()
                    dup_records.append({"InChIKey": k, "Count": len(rows), "IDs": "|".join(items)[:500]})

        # 清理临时列
        df.drop(columns=["_new_inchi", "_new_inchikey"], inplace=True)

        # 追加写出
        df.to_csv(out, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    print(f"[STATS] rows processed: {total}")
    print(f"[STATS] computed (attempted & got both): {computed}")
    print(f"[STATS] filled InChI: {filled_inchi}")
    print(f"[STATS] filled InChIKey: {filled_key}")
    print(f"[STATS] failed computations: {failed}")
    print(f"[RESULT] wrote: {out}")

    if args.dedupe and len(dup_records) > 0:
        dup_df = pd.DataFrame(dup_records).sort_values(["Count", "InChIKey"], ascending=[False, True])
        dup_out = out.with_suffix(".dup_inchikey.csv")
        dup_df.to_csv(dup_out, index=False)
        print(f"[DEDUP] potential duplicates by InChIKey: {len(dup_df)} groups")
        print(f"[DEDUP] wrote: {dup_out}")

if __name__ == "__main__":
    sys.exit(main())