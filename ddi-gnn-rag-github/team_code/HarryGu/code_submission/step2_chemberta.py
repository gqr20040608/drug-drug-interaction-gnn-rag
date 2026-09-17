import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条

# ================= CONFIGURATION =================
INPUT_FILE = 'nodes_drug_final.csv'
OUTPUT_FILE = 'drug_features.pt'   # This will overwrite the old fingerprint file
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 32  # Process 32 drugs at a time to save memory
# ===============================================

# 1. Setup Device (Mac M2 MPS acceleration or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f">>> Using device: {device}")

# 2. Load Data
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

print(f">>> Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# CRITICAL: Sort by drug_id to ensure row alignment!
# We must ensure row 0 of the tensor corresponds to drug_id 0.
if 'drug_id' in df.columns:
    df.sort_values('drug_id', inplace=True)
else:
    print("⚠️ Warning: 'drug_id' column missing. Assuming file is already sorted.")

smiles_list = df['smiles'].tolist()
print(f"   Total drugs to process: {len(smiles_list)}")

# 3. Load Pre-trained ChemBERTa Model
print(f">>> Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval() # Set to evaluation mode (no training)

# 4. Processing Loop (Batch Processing)
print(f">>> Starting embedding extraction (Output Dim: 768)...")

all_embeddings = []

# Process in batches
for i in tqdm(range(0, len(smiles_list), BATCH_SIZE)):
    batch_smiles = smiles_list[i : i + BATCH_SIZE]
    
    # Tokenize
    # padding=True: pad short strings to match the longest in the batch
    # truncation=True: cut off extremely long strings (rare for drugs)
    inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to GPU (MPS)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract the [CLS] token embedding (first token)
        # Shape: [batch_size, seq_len, 768] -> [batch_size, 768]
        # The [CLS] token is the standard representation for the whole molecule.
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Move back to CPU and convert to numpy
        all_embeddings.append(batch_embeddings.cpu().numpy())

# 5. Concatenate and Save
# Stack all batches into one large matrix
final_features = np.vstack(all_embeddings)
x_tensor = torch.tensor(final_features, dtype=torch.float32)

print(f"\n>>> Final Tensor Shape: {x_tensor.shape}")
print(f"    (Should be [Num_Drugs, 768])")

torch.save(x_tensor, OUTPUT_FILE)
print(f"✅ Success! ChemBERTa embeddings saved to: {OUTPUT_FILE}")
