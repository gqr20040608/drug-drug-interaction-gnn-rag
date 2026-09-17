import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

def inchikey_from_mol(mol):
    """Return an InChIKey for a molecule using whatever is available in this RDKit build."""
    if mol is None:
        return None
    # 1) Try the direct function (absent in some builds)
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        pass
    # 2) Compute InChI string, then convert to key
    try:
        inchi_str = inchi.MolToInchi(mol)
        if inchi_str:
            return inchi.InchiToInchiKey(inchi_str)
    except Exception:
        pass
    # 3) Give up
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Datasets/normalized/drugs_patched.csv")
    ap.add_argument("--sdf", default="Datasets/DrugBank/structures.sdf")
    ap.add_argument("--out", default="Datasets/normalized/drugs_patched2.csv")
    args = ap.parse_args()

    # Load current master and treat "\N" as missing
    df = pd.read_csv(args.csv, dtype=str).replace({"\\N": None})
    if "type" not in df.columns:
        df["type"] = ""

    # We only patch rows missing BOTH and not biotech
    need = (df["smiles"].isna()) & (df["inchi_key"].isna()) & (df["type"].str.lower() != "biotech")
    if not need.any():
        print("[info] nothing to patch (no eligible rows).")
        df.to_csv(args.out, index=False)
        print(f"[done] wrote {args.out}")
        return

    # Build a map: DrugBank ID -> (SMILES, InChIKey)
    suppl = Chem.SDMolSupplier(args.sdf, sanitize=True, removeHs=False)
    patch = {}
    seen = 0
    for mol in suppl:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        # Common DrugBank ID property names
        did = (
            props.get("DRUGBANK_ID")
            or props.get("DRUGBANK_IDs")
            or props.get("DB_ID")
            or props.get("drugbank_id")
        )
        if not did:
            continue
        # Some entries have multiple IDs like "DB00001; DB12345" -> take each
        ids = [s.strip() for s in str(did).replace(",", ";").split(";") if s.strip()]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:
            smi = None
        ik = None
        if smi:
            ik = inchikey_from_mol(mol)  # robust IK computation
        for one_id in ids:
            if smi:
                patch[one_id] = (smi, ik)
                seen += 1

    print(f"[info] indexed {seen} DrugBank ID -> structure pairs from SDF "
          f"({len(patch)} unique DrugBank IDs).")

    # Apply patch only to rows still missing
    rows = df.index[need]
    smi_fill, ik_fill = [], []
    for did in df.loc[rows, "drug_id"].astype(str):
        smi, ik = patch.get(did, (None, None))
        smi_fill.append(smi)
        ik_fill.append(ik)

    df.loc[rows, "smiles"] = smi_fill
    df.loc[rows, "inchi_key"] = ik_fill

    # Report remaining missing
    still_missing = int((df["smiles"].isna() & df["inchi_key"].isna()
                         & (df["type"].str.lower() != "biotech")).sum())
    print(f"[report] still missing small molecules after DrugBank SDF: {still_missing}")

    df.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()