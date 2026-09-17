#!/usr/bin/env python3
import pandas as pd
from collections import Counter
from pathlib import Path

root = Path("Datasets/normalized")
drugs = pd.read_csv(root/"drugs.csv", dtype=str, keep_default_na=False)
syns  = pd.read_csv(root/"drug_synonym.csv", dtype=str, keep_default_na=False)

# 统计每个 drug_id 的同义词频次；回填最短/最常见
name_by_id = {}
for did, group in syns.groupby("drug_id"):
    c = Counter(s.strip() for s in group["synonym"] if s.strip())
    if not c: 
        continue
    # 先取出现最多的；并在 tie 时选最短的
    best = sorted(c.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
    name_by_id[did] = best

mask = (drugs["preferred_name"].eq("")) | (drugs["preferred_name"].isna())
filled = 0
for i, row in drugs[mask].iterrows():
    did = row["drug_id"]
    if did in name_by_id:
        drugs.at[i, "preferred_name"] = name_by_id[did]
        filled += 1

drugs.to_csv(root/"drugs.csv", index=False)
print(f"[names] filled {filled} preferred_name cells from synonyms.")