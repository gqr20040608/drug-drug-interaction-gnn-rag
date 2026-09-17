from pathlib import Path
import csv

root = Path("Datasets/Reactome")
src  = root / "ReactomePathways_comma.csv"
dst  = root / "ReactomePathways_human.csv"

with open(src, "r", encoding="utf-8", newline="") as fin, \
     open(dst, "w", encoding="utf-8", newline="") as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    for row in reader:
        # 跳过空行
        if not row or len(row) < 3:
            continue
        # 第三列精确匹配 Homo sapiens
        if row[2].strip() == "Homo sapiens":
            writer.writerow(row)

print(f"✅ 已完成筛选，只保留第三列为 Homo sapiens 的行，保存为 {dst.name}")