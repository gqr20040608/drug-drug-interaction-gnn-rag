#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST builder for proteins_master (Human-only, line-by-line streaming).
- Streams HUGE STRING files and keeps only 9606.* rows.
- Avoids loading all species into memory (10x faster under WSL/OneDrive).
- Shows byte-level progress bars with ETA.

Inputs (relative to repo root):
  - Datasets/normalized/protein_layer/edges_protein_pathway.csv   (needs column: uniprot_id)
  - Datasets/STRING/protein.aliases.v12.0.txt                     (STRING→UniProt mapping; source==UniProt_AC)
  - Datasets/STRING/protein.info.v12.0.txt                        (STRING protein metadata)

Output (NOTE: renamed as requested):
  - Datasets/normalized/protein_layer/proteins_master_human_fast.csv
    columns: uniprot_id,string_id,preferred_name,protein_size,annotation

Run:
  python3 build_proteins_master_fast.py
"""
from __future__ import annotations
from pathlib import Path
import os
import csv
from collections import defaultdict
from typing import Dict, Tuple
from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None

# ----------------- config -----------------
HUMAN_PREFIX = "9606."
ALIAS_SOURCE = "UniProt_AC"
CHUNK_PRINT_EVERY = 250_000  # only used for occasional debug prints

# ----------------- utils ------------------

def here() -> Path:
    return Path(__file__).resolve().parent


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def stream_aliases_human_uniprot(path: Path) -> Dict[str, str]:
    """Return mapping: uniprot_id -> string_protein_id (choose first seen).
    Reads file line-by-line; keeps only HUMAN_PREFIX + ALIAS_SOURCE rows.
    """
    total = os.stat(path).st_size
    mapping: Dict[str, str] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f, \
         tqdm(total=total, unit="B", unit_scale=True, desc="Aliases (stream)") as pbar:
        header_skipped = False
        for line in f:
            pbar.update(len(line.encode("utf-8", "ignore")))
            if not header_skipped:
                # skip commented header lines
                if line.startswith("#"):
                    continue
                header_skipped = True  # first non-# line encountered; columns are tab-separated
            # very fast prefilter by prefix position
            if not line.startswith(HUMAN_PREFIX):
                continue
            # split minimal columns (0:string_id, 1:alias, 2:source)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]
            if source != ALIAS_SOURCE:
                continue
            # first-come mapping for each UniProt AC
            if alias not in mapping:
                mapping[alias] = string_id
    return mapping


def stream_info_human(path: Path) -> Dict[str, Tuple[str, str, str]]:
    """Return mapping: string_protein_id -> (preferred_name, protein_size, annotation).
    Reads line-by-line and keeps only HUMAN_PREFIX rows.
    """
    total = os.stat(path).st_size
    meta: Dict[str, Tuple[str, str, str]] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f, \
         tqdm(total=total, unit="B", unit_scale=True, desc="Info (stream)") as pbar:
        header_skipped = False
        for line in f:
            pbar.update(len(line.encode("utf-8", "ignore")))
            if not header_skipped:
                if line.startswith("#"):
                    continue
                header_skipped = True
            if not line.startswith(HUMAN_PREFIX):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            string_id, preferred_name, protein_size, annotation = parts[:4]
            meta[string_id] = (preferred_name, protein_size, annotation)
    return meta


# ----------------- main -------------------

def main() -> None:
    base = here()
    outdir = base / "Datasets" / "normalized" / "protein_layer"
    string_dir = base / "Datasets" / "STRING"

    edges_pwy = outdir / "edges_protein_pathway.csv"
    aliases_txt = string_dir / "protein.aliases.v12.0.txt"
    info_txt    = string_dir / "protein.info.v12.0.txt"

    if not edges_pwy.exists():
        raise FileNotFoundError(f"Missing {edges_pwy}")
    if not aliases_txt.exists():
        raise FileNotFoundError(f"Missing {aliases_txt}")
    if not info_txt.exists():
        raise FileNotFoundError(f"Missing {info_txt}")

    # 1) Reactome proteins (uniprot set)
    reactome_prots = pd.read_csv(edges_pwy, dtype=str, usecols=["uniprot_id"], engine="python").drop_duplicates()

    # 2) Stream aliases → map uniprot -> string_id (human + UniProt_AC only)
    print("[Stage 1/3] Streaming aliases (human-only UniProt_AC) …")
    uni_to_string = stream_aliases_human_uniprot(aliases_txt)

    # 3) Stream info → map string_id -> meta
    print("[Stage 2/3] Streaming STRING info (human-only) …")
    string_to_meta = stream_info_human(info_txt)

    # 4) Build master rows (left-join on Reactome set) and write out
    print("[Stage 3/3] Writing proteins_master_human_fast.csv …")
    out_path = outdir / "proteins_master_human_fast.csv"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as fw:
        w = csv.writer(fw)
        w.writerow(["uniprot_id", "string_id", "preferred_name", "protein_size", "annotation"])
        # Iterate reactome set to ensure full coverage
        for up in tqdm(reactome_prots["uniprot_id"].tolist(), desc="Writing rows"):
            sid = uni_to_string.get(up)
            if sid is not None:
                pref, size, anno = string_to_meta.get(sid, ("", "", ""))
            else:
                pref = size = anno = ""
            w.writerow([up, sid or "", pref, size, anno])

    # Summary
    n_total = len(reactome_prots)
    n_with_sid = sum(1 for up in reactome_prots["uniprot_id"] if up in uni_to_string)
    print("\n=== FAST Build Summary ===")
    print(f"Reactome proteins:      {n_total:,}")
    print(f"With STRING string_id:  {n_with_sid:,} ({n_with_sid/max(1,n_total):.2%})")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
