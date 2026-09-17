import pandas as pd
from pathlib import Path

# 路径
root = Path("Datasets/Reactome")
src  = root / "ReactomePathwaysRelation.csv"
dst  = root / "ReactomePathwaysRelation_comma.csv"   # 输出文件名

# 读取原文件（tab 分隔）
df = pd.read_csv(src, sep="\t", dtype=str, on_bad_lines="skip", engine="python")

# 清除空格
df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

# 保存为逗号分隔
df.to_csv(dst, sep=",", index=False)

print(f"✅ 已将 {src.name} 从 tab 分隔转换为 comma 分隔，保存为 {dst.name}")