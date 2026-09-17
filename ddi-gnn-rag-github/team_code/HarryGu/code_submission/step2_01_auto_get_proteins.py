import pandas as pd
import requests
import time
from tqdm import tqdm
import os

# ================= 配置 =================
OUTPUT_FILE = 'nodes_protein.csv'
# =======================================

def find_files():
    """自动寻找 Drug-Target 和 PPI 文件"""
    dt_file = None
    dt_col = None
    ppi_file = None
    
    files = [f for f in os.listdir('.') if f.endswith('.csv') and 'node' not in f]
    
    print(">>> 正在扫描 CSV 文件...")
    
    for f in files:
        try:
            df = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df.columns]
            
            # 1. 寻找 Drug-Target 文件
            # 特征：同时包含 'drug' 和 ('protein' 或 'target' 或 'uniprot')
            if any('drug' in c for c in cols) and \
               any(x in c for c in cols for x in ['protein', 'target', 'uniprot']):
                dt_file = f
                # 找到那个代表蛋白质的列名
                for c in df.columns:
                    if any(x in c.lower() for x in ['protein', 'target', 'uniprot']) and 'drug' not in c.lower():
                        dt_col = c
                        break
                print(f"✅ 找到疑似 Drug-Target 文件: {f} (蛋白列: {dt_col})")
            
            # 2. 寻找 PPI 文件
            # 特征：包含 'protein' 两次，或者 'node_a', 'node_b'
            elif len([c for c in cols if 'protein' in c]) >= 2:
                ppi_file = f
                print(f"✅ 找到疑似 PPI 文件: {f}")
                
        except:
            pass
            
    return dt_file, dt_col, ppi_file

def fetch_sequences_batch(id_list, batch_size=50):
    base_url = "https://rest.uniprot.org/uniprotkb/accessions"
    results = {}
    for i in tqdm(range(0, len(id_list), batch_size), desc="下载序列中"):
        batch = id_list[i : i + batch_size]
        try:
            params = {'accessions': ','.join(batch), 'format': 'json', 'fields': 'accession,sequence'}
            r = requests.get(base_url, params=params)
            if r.status_code == 200:
                for entry in r.json().get('results', []):
                    results[entry['primaryAccession']] = entry['sequence']['value']
        except:
            pass
        time.sleep(0.5)
    return results

# --- 主流程 ---
dt_file, dt_col, ppi_file = find_files()

if not dt_file:
    print("\n❌ 严重错误：没找到任何像 '药物-靶点' 的文件！")
    print("   请确认你文件夹里有包含 'drug' 和 'uniprot/target' 列的 CSV 文件。")
    print("   (或者请你手动告诉我文件名)")
    exit()

print(f"\n>>> 将从 {dt_file} 的 '{dt_col}' 列提取蛋白质 ID...")
protein_ids = set()

# 读取 Drug-Target
df_dt = pd.read_csv(dt_file)
ids = df_dt[dt_col].dropna().astype(str).tolist()
protein_ids.update(ids)

# 读取 PPI (如果有)
if ppi_file:
    print(f">>> 将从 PPI 文件 {ppi_file} 补充 ID...")
    df_ppi = pd.read_csv(ppi_file)
    # 假设前两列是蛋白
    for c in df_ppi.columns[:2]:
        protein_ids.update(df_ppi[c].dropna().astype(str).tolist())

unique_ids = sorted(list(protein_ids))
print(f"✅ 总共需要下载 {len(unique_ids)} 个蛋白质序列。")

# 下载
seq_map = fetch_sequences_batch(unique_ids)
print(f"✅ 成功下载 {len(seq_map)} 条序列。")

# 保存
data = [{'protein_id': i, 'uniprot_id': uid, 'sequence': seq} 
        for i, (uid, seq) in enumerate(seq_map.items())]
df_final = pd.DataFrame(data)
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"🎉 蛋白主表已保存为: {OUTPUT_FILE}")
