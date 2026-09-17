import pandas as pd
from pathlib import Path

# 路径
root = Path("Datasets/Reactome")
src = root / "ReactomePathwaysRelation.csv"
dst = root / "ReactomePathwaysRelation_comma.csv"

# 读取原文件（tab 分隔）
df = pd.read_csv(src, sep="\t", dtype=str, on_bad_lines="skip", engine="python")

# 清理空格（防止ID前后有空白）
df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

# 保存为逗号分隔格式
df.to_csv(dst, sep=",", index=False)

print(f"✅ 已将 {src.name} 从 tab 转换为 comma 分隔，输出到 {dst.name}")