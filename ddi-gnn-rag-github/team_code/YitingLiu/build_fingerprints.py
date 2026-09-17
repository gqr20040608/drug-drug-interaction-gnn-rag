import argparse, pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from rdkit.DataStructs import BitVectToFPSText

def fp_hex(mol, r):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=2048, includeChirality=False)
    bv = gen.GetFingerprint(mol)
    return BitVectToFPSText(bv)  # hex-like compact text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default="drug_id")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    cols = []
    for need in [args.id_col, args.smiles_col]:
        if need not in df.columns: df[need] = pd.NA
    out_rows = []
    for _, row in df.iterrows():
        smi = row[args.smiles_col]
        did = row[args.id_col]
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            out_rows.append({
                args.id_col: did, "ok": 0,
                "ecfp4": "", "ecfp6": "",
                "MolWt": "", "LogP": "", "TPSA": "",
                "HBA": "", "HBD": "", "RotB": ""
            })
            continue
        out_rows.append({
            args.id_col: did, "ok": 1,
            "ecfp4": fp_hex(mol, 2),     # ECFP4
            "ecfp6": fp_hex(mol, 3),     # ECFP6
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        })
    pd.DataFrame(out_rows).to_csv(args.out, index=False)

if __name__ == "__main__":
    main()