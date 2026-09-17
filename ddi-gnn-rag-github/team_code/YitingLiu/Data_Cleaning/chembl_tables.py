from pathlib import Path
import sqlite3
import pandas as pd

# -------------------- paths --------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
OUT  = DATA / "normalized"

CH_EMBL_DB = DATA / "ChEMBL" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"

# -------------------- ensure output dir --------------------
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- connect to sqlite --------------------
conn = sqlite3.connect(f"file:{CH_EMBL_DB}?mode=ro", uri=True)
cur = conn.cursor()

# -------------------- fetch table names --------------------
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [r[0] for r in cur.fetchall()]

# -------------------- collect schema, counts, and samples --------------------
table_info = []
for t in tables:
    try:
        # row count
        cur.execute(f"SELECT COUNT(*) FROM '{t}'")
        n = cur.fetchone()[0]
    except Exception as e:
        n = f"ERROR: {e}"

    # column names
    cur.execute(f"PRAGMA table_info('{t}')")
    cols = [r[1] for r in cur.fetchall()]

    # first 5 rows
    try:
        sample_df = pd.read_sql_query(f"SELECT * FROM '{t}' LIMIT 5", conn)
        sample_str = sample_df.to_csv(index=False)
    except Exception as e:
        sample_str = f"ERROR: {e}"

    table_info.append({
        "table_name": t,
        "row_count": n,
        "columns": ", ".join(cols),
        "sample_rows": sample_str.strip()
    })

# -------------------- save to CSV --------------------
df = pd.DataFrame(table_info)
out_csv = OUT / "chembl_tables_with_samples.csv"
df.to_csv(out_csv, index=False)

print(f"Exported {len(df)} tables with schema + samples to {out_csv}")
conn.close()