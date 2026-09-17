"""
Extract specific tables from a PostgreSQL plain-text dump (.sql) that uses COPY ... FROM stdin.
Writes each requested table to a TSV with a header row. No DB server required.

Usage:
  python3 extract_tables_from_pg_dump.py \
      --sql Datasets/DrugCentral/drugcentral.dump.11012023.sql \
      --tables structures xref synonyms \
      --outdir Datasets/DrugCentral/extracted
"""
import argparse, os, re, io

COPY_RE = re.compile(
    r"^COPY\s+(?P<schema>\w+)\.(?P<table>\w+)\s*\((?P<cols>[^)]+)\)\s+FROM\s+stdin;[^\n]*\n",
    re.MULTILINE
)

def parse_copy_blocks(sql_text):
    """
    Yields dicts: {'schema','table','cols':[...],'data': 'raw tsv text'}
    """
    for m in COPY_RE.finditer(sql_text):
        start = m.end()
        # data until a line containing '\.'
        end = sql_text.find("\n\\.\n", start)
        if end == -1:
            raise RuntimeError("COPY block without terminator '\\.' found.")
        block = sql_text[start:end]  # includes a trailing newline before \.
        cols = [c.strip() for c in m.group("cols").split(",")]
        yield {
            "schema": m.group("schema"),
            "table": m.group("table"),
            "cols": cols,
            "data": block
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", required=True, help="Path to DrugCentral .sql dump")
    ap.add_argument("--tables", nargs="+", required=True,
                    help="Table names to extract (e.g., structures xref synonyms)")
    ap.add_argument("--outdir", required=True, help="Output directory for TSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with io.open(args.sql, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    found = 0
    for blk in parse_copy_blocks(text):
        if blk["table"] not in args.tables:
            continue
        out_path = os.path.join(args.outdir, f"{blk['table']}.tsv")
        with io.open(out_path, "w", encoding="utf-8") as out:
            out.write("\t".join(blk["cols"]) + "\n")
            out.write(blk["data"])
        found += 1
        print(f"[ok] wrote {out_path}  (cols={len(blk['cols'])})")

    if found == 0:
        print("[warn] None of the requested tables were found. "
              "Run again with --tables all to see what's available.")

if __name__ == "__main__":
    main()