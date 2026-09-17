from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import inchi as rdInchi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
OUT  = DATA / "normalized"
P_DRUGS = OUT / "drugs.csv"

def log(msg): print(f"[rdkit] {msg}")

def calc_inchikey(smiles: str) -> str:
    """把 SMILES 转成 InChIKey；失败则返回空串。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return ""
        # 可选：规范化/中和等预处理，这里先保持最简单
        ik = rdInchi.MolToInchiKey(mol)
        return ik or ""
    except Exception:
        return ""

def main():
    if not P_DRUGS.exists():
        raise FileNotFoundError(f"not found: {P_DRUGS}")

    log(f"reading: {P_DRUGS}")
    df = pd.read_csv(P_DRUGS, dtype=str, keep_default_na=False)

    # 只看非 biotech，且 smiles 有值而 inchi_key 为空
    mask_non_biotech = df["type"].str.lower() != "biotech"
    mask_need = mask_non_biotech & (df["smiles"].str.len() > 0) & (df["inchi_key"].str.len() == 0)

    need_n = int(mask_need.sum())
    total_nonbiotech = int(mask_non_biotech.sum())
    log(f"rows to compute (non-biotech, missing InChIKey but has SMILES): {need_n}/{total_nonbiotech}")

    if need_n == 0:
        log("nothing to do.")
        return

    # 计算
    patched = 0
    for idx in df[mask_need].index:
        ik = calc_inchikey(df.at[idx, "smiles"])
        if ik:
            df.at[idx, "inchi_key"] = ik
            patched += 1
        # 可按需打印进度：
        # if patched % 1000 == 0: log(f"patched {patched}")

    # 覆盖率统计（前后）
    before_missing_ik = int((df[mask_non_biotech]["inchi_key"] == "").sum()) + 0  # 在这之前其实是“后”，但下面我们对比更直观
    # 保存
    backup = P_DRUGS.with_suffix(".csv.bak_rdkit")
    if not backup.exists():
        df_backup = pd.read_csv(P_DRUGS, dtype=str, keep_default_na=False)
        df_backup.to_csv(backup, index=False)
        log(f"backup saved: {backup}")

    df.to_csv(P_DRUGS, index=False)
    after_missing_ik = int((df[mask_non_biotech]["inchi_key"] == "").sum())
    log(f"patched InChIKeys: +{patched}")
    log(f"coverage (non-biotech) InChIKey missing: {after_missing_ik}/{total_nonbiotech}")
    log("Done.")

if __name__ == "__main__":
    main()