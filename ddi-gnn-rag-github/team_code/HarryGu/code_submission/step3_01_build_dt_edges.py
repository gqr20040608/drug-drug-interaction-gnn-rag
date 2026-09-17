import pandas as pd
import torch
import os

# ================= 配置 =================
FILE_DRUG_NODES = 'nodes_drug_final.csv'       # 药物 ID 字典
FILE_PROT_NODES = 'nodes_protein.csv'          # 蛋白质 ID 字典
FILE_DT_EDGES = 'edges_drug_target_std.csv'    # 原始连边文件 (String ID)
OUTPUT_FILE = 'edge_index_dt.pt'               # 输出文件 (Integer ID)
# =======================================

print(">>> 正在加载节点映射字典...")

# 1. 加载药物映射 (DBxxxx -> 0, 1, 2...)
df_drug = pd.read_csv(FILE_DRUG_NODES)
# 优先找 drugbank_id，如果找不到找第一列
d_key = 'drugbank_id' if 'drugbank_id' in df_drug.columns else df_drug.columns[0] # 通常第一列就是ID
drug_map = dict(zip(df_drug[d_key], df_drug['drug_id']))
print(f"   药物字典: {len(drug_map)} 个 (Key: {d_key})")

# 2. 加载蛋白质映射 (Pxxxxx -> 0, 1, 2...)
df_prot = pd.read_csv(FILE_PROT_NODES)
# 优先找 uniprot_id
p_key = 'uniprot_id' if 'uniprot_id' in df_prot.columns else df_prot.columns[0]
prot_map = dict(zip(df_prot[p_key], df_prot['protein_id']))
print(f"   蛋白字典: {len(prot_map)} 个 (Key: {p_key})")

# 3. 读取原始连边文件
print(f"\n>>> 正在读取连边文件 {FILE_DT_EDGES}...")
if not os.path.exists(FILE_DT_EDGES):
    print(f"❌ 错误：找不到文件 {FILE_DT_EDGES}")
    exit()

df_edges = pd.read_csv(FILE_DT_EDGES)
print(f"   原始列名: {df_edges.columns.tolist()}")

# 自动识别列名
# 找含有 'drug' 的列做 source，含有 'protein'/'target'/'uniprot' 的列做 target
col_drug = None
col_prot = None

for c in df_edges.columns:
    c_lower = c.lower()
    if 'drug' in c_lower and not col_drug:
        col_drug = c
    elif ('protein' in c_lower or 'target' in c_lower or 'uniprot' in c_lower) and not col_prot:
        col_prot = c

if not col_drug or not col_prot:
    print("⚠️ 无法自动识别列名，尝试使用前两列...")
    col_drug = df_edges.columns[0]
    col_prot = df_edges.columns[1]

print(f"   映射方案: 药物列='{col_drug}' -> 蛋白列='{col_prot}'")

# 4. 执行映射 (String -> Int)
# map 函数：如果字典里找不到这个 ID，会变成 NaN
df_edges['src_int'] = df_edges[col_drug].map(drug_map)
df_edges['dst_int'] = df_edges[col_prot].map(prot_map)

# 5. 清洗无效连边
# (比如 CSV 里有的药，但在我们的 nodes 表里被删掉了)
before = len(df_edges)
df_clean = df_edges.dropna(subset=['src_int', 'dst_int'])
after = len(df_clean)
dropped = before - after

if dropped > 0:
    print(f"⚠️ 丢弃了 {dropped} 条连边 (因为 ID 在节点表中找不到)。")
else:
    print("✅ 所有连边 ID 匹配完美！")

# 6. 转换为 PyTorch Tensor 并保存
# 形状 [2, Num_Edges]
src = df_clean['src_int'].values.astype(int)
dst = df_clean['dst_int'].values.astype(int)
edge_index = torch.tensor([src, dst], dtype=torch.long)

print(f"\n>>> 最终药物-靶点连边形状: {edge_index.shape}")
torch.save(edge_index, OUTPUT_FILE)
print(f"✅ 成功！已保存为: {OUTPUT_FILE}")
