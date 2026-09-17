import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import RGCNConv
from rdkit import Chem
from rdkit.Chem import Draw
import os
import json

# ================= MODULE IMPORT =================
# Import the literature summarization module (RAG)
# Ensure summarize_interactions.py is in the same folder as this script.
import summarize_interactions 
print("System: Successfully imported summarize_interactions module.")
# ==============================================

# ================= CONFIGURATION =================
# 1. DATA DIRECTORY
# This should point to the folder containing 'meta.json', 'features/', and 'index_edges/'.
# Based on your training script, this is likely in 'GNN/GNN_datasets'.
DATA_DIR = "."

# 2. MODEL FILE PATH
# UPDATED: Pointing to the 'Approach6' folder inside 'ddi_project'.
MODEL_PATH = os.path.join("Approach6", "rgcn_encoder.pt")

# 3. DRUG LIST PATH
# The CSV containing the list of drugs (Drug ID, Name, SMILES).
FILE_NODES = "nodes_drug_final.csv" 
# =================================================

print("System: Initializing RGCN Demo...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------
# 1. Model Definition
#    (Exact copy of the classes from train_rgcn_baseline.py)
# -----------------------------------------------------------

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=8,
                 dropout=0.2, use_bn=True, residual=True):
        super().__init__()
        self.conv = RGCNConv(in_dim, out_dim, num_rels, num_bases=num_bases)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x, edge_index, etype):
        h = self.conv(x, edge_index, etype)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            h = h + x
        return h

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim=256, hidden=128, out=128,
                 num_rels=6, num_bases=8, num_layers=2, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out]
        self.layers = nn.ModuleList([
            RGCNLayer(
                dims[i], dims[i + 1], num_rels,
                num_bases=num_bases,
                dropout=dropout,
                use_bn=True,
                residual=True,
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, X, edge_index, rel_type):
        h = X
        for layer in self.layers:
            h = layer(h, edge_index, rel_type)
        return h

# -----------------------------------------------------------
# 2. Graph Construction Logic
#    (Loads features from .npy files and builds the graph)
# -----------------------------------------------------------
def build_graph_for_demo(indir, device):
    """
    Constructs the homogeneous graph required by RGCN.
    Reads meta.json and features from the specified DATA_DIR.
    """
    import pathlib
    indir = pathlib.Path(indir)
    
    # Validation: Check if meta.json exists
    if not (indir / "meta.json").exists():
        raise FileNotFoundError(f"meta.json not found in DATA_DIR: {indir}. Please check the path.")
        
    with open(indir / "meta.json", "r") as f:
        meta = json.load(f)

    Nd = int(meta["num_nodes"]["drug"])
    Np = int(meta["num_nodes"]["protein"])
    Nw = int(meta["num_nodes"]["pathway"])

    # Load Features
    # The RGCN baseline expects 256-dimension features
    try:
        # Load Drug Features (Prioritize 'fused', fallback to standard)
        if (indir / "features/drug_feats_256_fused.npy").exists():
            fd = torch.from_numpy(np.load(indir / "features/drug_feats_256_fused.npy")).float()
        else:
            fd = torch.from_numpy(np.load(indir / "features/drug_feats_256.npy")).float()
        
        # Load Protein Features (Prioritize 'Approach6', fallback to standard)
        if (indir / "features/protein_feats_256_approach6.npy").exists():
            fp = torch.from_numpy(np.load(indir / "features/protein_feats_256_approach6.npy")).float()
        else:
            fp = torch.from_numpy(np.load(indir / "features/protein_feats_256.npy")).float()
            
        # Load Pathway Features
        fw = torch.from_numpy(np.load(indir / "features/pathway_feats_256.npy")).float()
        
    except FileNotFoundError as e:
        print(f"Error: Missing specific feature files in {indir}. Ensure .npy files exist.")
        raise e

    # Concatenate all features into matrix X
    off_d, off_p, off_w = 0, Nd, Nd + Np
    total = Nd + Np + Nw
    X = torch.zeros((total, fd.size(1)), dtype=torch.float32)
    X[off_d:off_d + Nd] = fd
    X[off_p:off_p + Np] = fp
    X[off_w:off_w + Nw] = fw

    # Helper function to load edges from CSVs
    def load_edges(csv_name, src_off, dst_off):
        p = indir / "index_edges" / csv_name
        if not p.exists(): 
            return torch.empty((2,0), dtype=torch.long), torch.empty((0), dtype=torch.long)
        
        df = pd.read_csv(p)[["src_idx", "dst_idx", "rel_type"]].to_numpy()
        ei = torch.from_numpy(df[:, :2].T).long()
        # Adjust indices based on node type offsets
        ei[0] += src_off
        ei[1] += dst_off
        et = torch.from_numpy(df[:, 2]).long()
        return ei, et

    # Load interaction networks
    dt_ei, dt_et = load_edges("dt.csv", off_d, off_p)
    pp_ei, pp_et = load_edges("ppw.csv", off_p, off_w)
    ww_ei, ww_et = load_edges("ww.csv", off_w, off_w)
    me_ei, me_et = load_edges("mech.csv", off_d, off_p) 

    # Combine edges into single tensors
    edge_index = torch.cat([dt_ei, pp_ei, ww_ei, me_ei], dim=1)
    rel_type = torch.cat([dt_et, pp_et, ww_et, me_et], dim=0)

    return X.to(device), edge_index.to(device), rel_type.to(device), Nd

# -----------------------------------------------------------
# 3. System Initialization
# -----------------------------------------------------------

# Load Drug List for UI Dropdown
print("System: Loading drug list...")
if not os.path.exists(FILE_NODES):
    print(f"Error: {FILE_NODES} not found. Please ensure the CSV file is in the root directory.")
    exit()

df_nodes = pd.read_csv(FILE_NODES)
name_col = 'name' if 'name' in df_nodes.columns else 'drugbank_id'
if name_col not in df_nodes.columns: name_col = 'drug_id'

df_nodes['label'] = df_nodes[name_col].astype(str) + " (ID:" + df_nodes['drug_id'].astype(str) + ")"
df_nodes_sorted = df_nodes.sort_values(by='label')
all_drug_options = df_nodes_sorted['label'].tolist()

# Create mappings
id_map = dict(zip(df_nodes['label'], df_nodes['drug_id']))
smiles_map = dict(zip(df_nodes['drug_id'], df_nodes['smiles']))
clean_name_map = dict(zip(df_nodes['label'], df_nodes[name_col].astype(str)))

# Build Graph
print(f"System: Building RGCN Graph from {DATA_DIR}...")
try:
    X, edge_index, rel_type, num_drugs = build_graph_for_demo(DATA_DIR, device)
except Exception as e:
    print(f"\nCRITICAL ERROR: Failed to build graph. {e}")
    exit()

print(f"System: Graph built successfully (Nodes: {X.shape[0]}, Edges: {edge_index.shape[1]})")

# Load Model Weights
print(f"System: Loading RGCN model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}.")
    print("Please verify the file path.")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=device)
encoder_state = checkpoint['encoder_state_dict']

# Extract hyperparameters from checkpoint
in_dim = checkpoint.get('in_dim', 256)
hidden = checkpoint.get('hidden', 128)
out_dim = checkpoint.get('out', 128)
num_rels = checkpoint.get('num_rels', 6)
num_layers = checkpoint.get('num_layers', 2)

# Initialize Model
model = RGCNEncoder(
    in_dim=in_dim, 
    hidden=hidden, 
    out=out_dim, 
    num_rels=num_rels,
    num_bases=8,
    num_layers=num_layers
).to(device)

model.load_state_dict(encoder_state)
model.eval()
print("System: RGCN Model Loaded Successfully!")

# Pre-compute Drug Embeddings
print("System: Pre-computing embeddings...")
with torch.no_grad():
    Z = model(X, edge_index, rel_type)
    drug_embeddings = Z[:num_drugs] 

# -----------------------------------------------------------
# 4. Core Logic
# -----------------------------------------------------------
def search_drug_options(query):
    """Filters dropdown options based on search query."""
    if not query: return gr.Dropdown(choices=[], value=None, label="No results found")
    query = query.lower()
    matches = [opt for opt in all_drug_options if query in opt.lower()]
    if len(matches) > 50: matches = matches[:50]
    if not matches: return gr.Dropdown(choices=[], value=None, label="No matches found")
    return gr.Dropdown(choices=matches, value=matches[0], label=f"Select from {len(matches)} matches", interactive=True)

def full_analysis(drug1_label, drug2_label):
    """Main analysis function."""
    if not drug1_label or not drug2_label:
        return "Please complete selection.", "", None, None
    
    id1 = id_map[drug1_label]
    id2 = id_map[drug2_label]
    name1 = clean_name_map[drug1_label]
    name2 = clean_name_map[drug2_label]
    
    # ID Validation
    if id1 >= len(drug_embeddings) or id2 >= len(drug_embeddings):
        return f"Error: Drug ID {max(id1, id2)} exceeds model range.", "", None, None

    # Prediction
    emb1 = drug_embeddings[id1]
    emb2 = drug_embeddings[id2]
    score = (emb1 * emb2).sum().sigmoid().item()
    
    pred_report = f"Drug A: {name1}\nDrug B: {name2}\n\n"
    pred_report += f"Model Architecture: RGCN (Approach 6)\n"
    pred_report += f"Predicted Score: {score:.4f}\n"
    if score > 0.5:
        pred_report += "Result: High Risk"
    else:
        pred_report += "Result: Low Risk"

    # Literature Summary
    lit_summary = "Generating summary..."
    try:
        lit_summary = summarize_interactions.summarize_interaction_effects(name1, name2)
    except Exception as e:
        lit_summary = f"Summary Error: {str(e)}"

    # Visualization
    mol1 = Chem.MolFromSmiles(smiles_map.get(id1, ""))
    mol2 = Chem.MolFromSmiles(smiles_map.get(id2, ""))
    img1 = Draw.MolToImage(mol1, size=(300, 300)) if mol1 else None
    img2 = Draw.MolToImage(mol2, size=(300, 300)) if mol2 else None
    
    return pred_report, lit_summary, img1, img2

# -----------------------------------------------------------
# 5. Gradio Interface
# -----------------------------------------------------------
with gr.Blocks(title="DDI Analysis (RGCN Approach 6)") as demo:
    gr.Markdown("# DDI Analysis System: RGCN Approach 6")
    gr.Markdown("Prediction based on Relational Graph Convolutional Networks using updated model artifacts.")
    
    with gr.Row():
        with gr.Column():
            search_a = gr.Textbox(label="Drug A Search")
            btn_a = gr.Button("Search")
            dd_a = gr.Dropdown(label="Select Drug A", choices=[], interactive=True)
        with gr.Column():
            search_b = gr.Textbox(label="Drug B Search")
            btn_b = gr.Button("Search")
            dd_b = gr.Dropdown(label="Select Drug B", choices=[], interactive=True)

    btn_pred = gr.Button("Analyze Interaction", variant="primary")

    with gr.Row():
        out_text = gr.Textbox(label="RGCN Prediction", lines=5)
        summary_out = gr.Textbox(label="Literature Review", lines=10)

    with gr.Row():
        img1_out = gr.Image(label="Drug A Structure", type="pil")
        img2_out = gr.Image(label="Drug B Structure", type="pil")

    btn_a.click(search_drug_options, search_a, dd_a)
    btn_b.click(search_drug_options, search_b, dd_b)
    btn_pred.click(full_analysis, inputs=[dd_a, dd_b], outputs=[out_text, summary_out, img1_out, img2_out])

if __name__ == "__main__":
    demo.launch()
