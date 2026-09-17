# preview_normalized_heads.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
NORM = ROOT / "Datasets" / "normalized"
OUT  = NORM / "_preview"
OUT.mkdir(parents=True, exist_ok=True)

def safe_read_head(path, n=5):
    # 兼容 \N、空值、杂编码；只读前 n 行避免大文件占内存
    try:
        df = pd.read_csv(
            path,
            nrows=n,
            dtype=str,
            keep_default_na=True,
            na_values=["\\N", ""],
            encoding_errors="ignore",
            low_memory=False,
        )
        return df
    except Exception as e:
        return e

report_lines = []
csvs = sorted(NORM.glob("*.csv"))
if not csvs:
    print(f"[error] 没在 {NORM} 里找到任何 .csv")
    raise SystemExit(1)

for i, p in enumerate(csvs, 1):
    df_or_err = safe_read_head(p, n=5)
    print("="*88)
    print(f"[{i}/{len(csvs)}] {p.name}")
    if isinstance(df_or_err, Exception):
        print("  -> 读取失败：", repr(df_or_err))
        report_lines.append(f"### {p.name}\n读取失败：`{repr(df_or_err)}`\n")
        continue

    df = df_or_err
    # 打印基本信息
    print("  rows preview:", len(df))
    print("  columns     :", list(df.columns))
    # 打印前 5 行（尽量横向精简）
    with pd.option_context("display.max_columns", 50, "display.width", 180):
        print(df)

    # 输出到 _preview/
    out_csv = OUT / f"{p.stem}.head.csv"
    df.to_csv(out_csv, index=False)
    # 也写入 markdown 片段，后面可汇总查看
    md = df.to_markdown(index=False)
    report_lines.append(f"### {p.name}\n\n{md}\n")

# 汇总一个 markdown 报告
report_md = OUT / "HEADS_REPORT.md"
report_md.write_text("# normalized CSV Heads (first 5 rows each)\n\n" + "\n".join(report_lines), encoding="utf-8")

print("="*88)
print(f"✅ 预览完成：已在 {OUT} 下生成每个文件的 .head.csv，以及汇总 {report_md.name}")