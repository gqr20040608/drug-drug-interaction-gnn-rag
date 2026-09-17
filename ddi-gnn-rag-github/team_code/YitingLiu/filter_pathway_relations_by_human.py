import pandas as pd
from pathlib import Path

BASE = Path("Datasets/normalized/protein_layer")
relations = pd.read_csv(BASE / "edges_pathway_pathway.csv")
nodes = pd.read_csv(BASE / "nodes_pathway.csv")

human_ids = set(nodes["pathway_id"])
filtered = relations[
    relations["parent_pathway"].isin(human_ids) &
    relations["child_pathway"].isin(human_ids)
]
filtered.to_csv(BASE / "edges_pathway_pathway_human.csv", index=False)
print(f"✅ Saved {len(filtered):,} human-only relations.")