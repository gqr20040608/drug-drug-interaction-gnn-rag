from lxml import etree
import os, csv, argparse

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def text(el, default=None): return el.text.strip() if (el is not None and el.text) else default
def collect(parent, path): return [text(x) for x in parent.findall(path) if text(x)]

def get_calc_prop(drug_el, kind_name: str):
    for p in drug_el.findall("./{*}calculated-properties/{*}property"):
        if text(p.find("./{*}kind")) == kind_name:
            v = text(p.find("./{*}value"))
            if v: return v
    return None

def iter_drugs(xml_path):
    # 关键：命名空间通配
    ctx = etree.iterparse(xml_path, events=("end",), tag="{*}drug", huge_tree=True, recover=True)
    for _, elem in ctx:
        yield elem
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

def parse_drug(elem):
    # 取主 ID（primary='true' 优先）
    dbid = None
    for id_el in elem.findall("./{*}drugbank-id"):
        if (id_el.get("primary") == "true") or (dbid is None):
            dbid = text(id_el)
    if not dbid: return None

    name  = text(elem.find("./{*}name"))
    dtype = elem.get("type")
    groups = collect(elem, "./{*}groups/{*}group")
    approval_status = "|".join(groups) if groups else None

    inchikey = get_calc_prop(elem, "InChIKey")
    smiles   = get_calc_prop(elem, "SMILES")

    drug_row = dict(drug_id=dbid, preferred_name=name, type=dtype,
                    approval_status=approval_status, inchi_key=inchikey, smiles=smiles)

    synonyms = [(dbid, s) for s in collect(elem, "./{*}synonyms/{*}synonym")]
    atcs = []
    for a in elem.findall("./{*}atc-codes/{*}atc-code"):
        code = a.get("code")
        if code: atcs.append((dbid, code))

    xrefs = []
    for x in elem.findall("./{*}external-identifiers/{*}external-identifier"):
        res = text(x.find("./{*}resource")); ident = text(x.find("./{*}identifier"))
        if res and ident: xrefs.append((dbid, res, ident))

    props = []
    for p in elem.findall("./{*}calculated-properties/{*}property"):
        kind = text(p.find("./{*}kind")); val = text(p.find("./{*}value"))
        if kind and val: props.append((dbid, kind, val, "DrugBank"))

    targets = []
    for t in elem.findall("./{*}targets/{*}target"):
        tname = text(t.find("./{*}name"))
        actions = collect(t, "./{*}actions/{*}action")
        action_str = "|".join(actions) if actions else None
        up = t.find("./{*}polypeptide")
        uniprot = None
        if up is not None:
            src = (up.get("source") or "").lower()
            if ("swiss" in src) or ("uniprot" in src): uniprot = up.get("id")
        if uniprot:
            targets.append((dbid, uniprot, tname, action_str, "DrugBank"))
    return drug_row, synonyms, atcs, xrefs, props, targets

def write_headers(out_dir):
    fpaths = {
        "drugs":                os.path.join(out_dir, "drugs.csv"),
        "syn":                  os.path.join(out_dir, "drug_synonym.csv"),
        "atc":                  os.path.join(out_dir, "drug_atc.csv"),
        "xref":                 os.path.join(out_dir, "drug_xref.csv"),
        "props":                os.path.join(out_dir, "drug_properties.csv"),
        "targets":              os.path.join(out_dir, "drug_targets.csv"),
    }
    files = {
        "drugs": open(fpaths["drugs"], "w", newline="", encoding="utf-8"),
        "syn":   open(fpaths["syn"], "w", newline="", encoding="utf-8"),
        "atc":   open(fpaths["atc"], "w", newline="", encoding="utf-8"),
        "xref":  open(fpaths["xref"], "w", newline="", encoding="utf-8"),
        "props": open(fpaths["props"], "w", newline="", encoding="utf-8"),
        "targets": open(fpaths["targets"], "w", newline="", encoding="utf-8"),
    }
    writers = {
        "drugs":   csv.DictWriter(files["drugs"], fieldnames=["drug_id","preferred_name","type","approval_status","inchi_key","smiles"]),
        "syn":     csv.writer(files["syn"]),
        "atc":     csv.writer(files["atc"]),
        "xref":    csv.writer(files["xref"]),
        "props":   csv.writer(files["props"]),
        "targets": csv.writer(files["targets"]),
    }
    writers["drugs"].writeheader()
    writers["syn"].writerow(["drug_id","synonym"])
    writers["atc"].writerow(["drug_id","atc_code"])
    writers["xref"].writerow(["drug_id","xref_db","xref_id"])
    writers["props"].writerow(["drug_id","kind","value","source"])
    writers["targets"].writerow(["drug_id","uniprot_id","target_name","action","source_db"])
    return files, writers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to DrugBank full XML (e.g., 'full database.xml')")
    ap.add_argument("--out", required=True, help="Output directory for CSVs")
    args = ap.parse_args()

    ensure_dir(args.out)
    files, writers = write_headers(args.out)

    n = 0
    kept = 0
    try:
        for drug_el in iter_drugs(args.xml):
            n += 1
            parsed = parse_drug(drug_el)
            if not parsed:
                continue
            drug_row, syns, atcs, xrefs, props, targets = parsed
            writers["drugs"].writerow(drug_row)
            for r in syns:    writers["syn"].writerow(r)
            for r in atcs:    writers["atc"].writerow(r)
            for r in xrefs:   writers["xref"].writerow(r)
            for r in props:   writers["props"].writerow(r)
            for r in targets: writers["targets"].writerow(r)
            kept += 1
            if kept % 1000 == 0:
                print(f"[info] processed {kept} drugs...")
    finally:
        for f in files.values():
            f.close()
    print(f"[done] parsed {kept} drugs (visited {n} <drug> elements).")
    print(f"[out] CSVs written to: {args.out}")

if __name__ == "__main__":
    main()