from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
OUT  = DATA / "normalized"

P_DRUGS = OUT / "drugs.csv"
SDF_PATH = DATA / "DrugBank" / "structures.sdf"

def log(msg): print(f"[sdf] {msg}")

def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s

ID_KEYS      = {"drugbank_id","drugbank","drugbank_id_primary","drugbank-id","drugbank_identifier"}
SMILES_KEYS  = {"smiles","canonical_smiles"}
INCHI_KEYS   = {"inchi_key","inchikey","standard_inchi_key","standard_inchikey"}

def parse_sdf(sdf_path: Path):
    """纯文本解析 SDF，返回 list(dict)；每个 dict 至少包含 drug_id / smiles / inchi_key 中的若干。"""
    records = []
    if not sdf_path.exists():
        raise FileNotFoundError(f"Cannot find SDF: {sdf_path}")

    with sdf_path.open("r", encoding="utf-8", errors="ignore") as f:
        cur_props = {}
        cur_field = None
        buf = []

        def flush_field():
            nonlocal cur_field, buf, cur_props
            if cur_field is not None:
                val = "\n".join(buf).strip()
                cur_props[cur_field] = val
            cur_field = None
            buf = []

        def finalize_record():
            nonlocal cur_props
            if cur_props:
                # 归一化 keys
                props_norm = {}
                for k, v in cur_props.items():
                    nk = norm_key(k)
                    props_norm[nk] = v.strip()

                # 抽取 DrugBank ID（SDF 有时会含列表，这里取第一行/第一个 ID）
                drug_id = ""
                for cand in ID_KEYS:
                    if cand in props_norm and props_norm[cand]:
                        first = props_norm[cand].splitlines()[0].split(";")[0].strip()
                        # 规范化形如 DB00001
                        m = re.search(r"(DB\d{5})", first, re.IGNORECASE)
                        drug_id = m.group(1).upper() if m else first
                        break

                smiles = ""
                for cand in SMILES_KEYS:
                    if cand in props_norm and props_norm[cand].strip():
                        smiles = props_norm[cand].splitlines()[0].strip()
                        break

                inchi_key = ""
                for cand in INCHI_KEYS:
                    if cand in props_norm and props_norm[cand].strip():
                        inchi_key = props_norm[cand].splitlines()[0].strip()
                        break

                if drug_id:
                    records.append({"drug_id": drug_id, "smiles": smiles, "inchi_key": inchi_key})

            cur_props = {}

        for line in f:
            line = line.rstrip("\n")
            if line == "$$$$":  # 记录结束
                flush_field()
                finalize_record()
                continue

            # 属性起始：>  <FIELD_NAME>
            if line.startswith(">"):
                flush_field()
                m = re.match(r">+\s*<([^>]+)>", line)
                cur_field = m.group(1).strip() if m else None
                continue

            # 属性值或分子块
            if cur_field is not None:
                buf.append(line)
            else:
                # 分子块内容忽略（坐标等）
                pass

        # 文件尾巴
        flush_field()
        finalize_record()

    return records

def main():
    log(f"reading drugs.csv: {P_DRUGS}")
    drugs = pd.read_csv(P_DRUGS, dtype=str, keep_default_na=False)

    # 统计补丁前覆盖（仅非 biotech）
    pre_non_biotech = drugs[drugs["type"].str.lower()!="biotech"]
    pre_missing_smiles = int((pre_non_biotech["smiles"]=="").sum())
    pre_missing_inchi  = int((pre_non_biotech["inchi_key"]=="").sum())
    pre_total          = len(pre_non_biotech)

    log(f"parsing SDF: {SDF_PATH}")
    recs = parse_sdf(SDF_PATH)
    log(f"parsed {len(recs)} records from SDF")

    if not recs:
        log("no records parsed; abort")
        return

    sdf_df = pd.DataFrame(recs, dtype=str).fillna("")
    # 对每个 drug_id 聚合：取第一个非空
    def first_nonempty(series):
        for v in series:
            if isinstance(v, str) and v.strip():
                return v
        return ""
    sdf_df = sdf_df.groupby("drug_id", as_index=False).agg({"smiles": first_nonempty, "inchi_key": first_nonempty})

    # merge & patch（只补空）
    merged = drugs.merge(sdf_df, on="drug_id", how="left", suffixes=("", "__sdf")).fillna("")
    need_smi = (merged["smiles"].eq(""))    & merged["smiles__sdf"].ne("")
    need_ik  = (merged["inchi_key"].eq("")) & merged["inchi_key__sdf"].ne("")

    patched_smiles = int(need_smi.sum())
    patched_inchi  = int(need_ik.sum())

    merged.loc[need_smi, "smiles"]     = merged.loc[need_smi, "smiles__sdf"]
    merged.loc[need_ik,  "inchi_key"]  = merged.loc[need_ik,  "inchi_key__sdf"]

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__sdf")], errors="ignore")
    merged = merged.replace("None","").fillna("")
    merged.to_csv(P_DRUGS, index=False)

    # 覆盖率统计（补丁后）
    post_non_biotech = merged[merged["type"].str.lower()!="biotech"]
    post_missing_smiles = int((post_non_biotech["smiles"]=="").sum())
    post_missing_inchi  = int((post_non_biotech["inchi_key"]=="").sum())
    post_total          = len(post_non_biotech)

    log(f"patched from SDF: +{patched_smiles} SMILES, +{patched_inchi} InChIKeys")
    log(f"coverage (non-biotech):")
    log(f"  before missing  SMILES: {pre_missing_smiles}/{pre_total}")
    log(f"  after  missing  SMILES: {post_missing_smiles}/{post_total}")
    log(f"  before missing InChIKey: {pre_missing_inchi}/{pre_total}")
    log(f"  after  missing InChIKey: {post_missing_inchi}/{post_total}")
    log("Done.")

if __name__ == "__main__":
    main()