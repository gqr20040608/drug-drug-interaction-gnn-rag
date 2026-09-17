import pandas as pd
import torch
import os

# ================= 配置 =================
NODES_FILE = 'nodes_drug_final.csv' 
EDGES_FILE = 'pairs_all.csv'   # 你的连边文件
OUTPUT_FILE = 'edge_index.pt'
# =======================================

# 1. 读取节点表，建立 ID 映射字典
print(f">>> 读取节点表 {NODES_FILE}...")
df_nodes = pd.read_csv(NODES_FILE)

# 我们需要把 DB00xxx 映射为 0, 1, 2...
# 优先找 'drugbank_id' 列，如果没找到就用第一列
id_col = 'drugbank_id' if 'drugbank_id' in df_nodes.columns else df_nodes.columns[0]
print(f"   使用 '{id_col}' 列作为 ID 键值。")
id_map = dict(zip(df_nodes[id_col], df_nodes['drug_id']))

# 2. 读取连边文件
print(f">>> 读取连边文件 {EDGES_FILE}...")
df_edges = pd.read_csv(EDGES_FILE)

# 【核心修正】过滤数据！只保留 y=1 的行
if 'y' in df_edges.columns:
    print("✅ 发现 'y' 标签列，正在过滤正样本 (y=1)...")
    original_count = len(df_edges)
    # 只取 y == 1 的行
    df_edges = df_edges[df_edges['y'] == 1].copy()
    filtered_count = len(df_edges)
    print(f"   过滤前: {original_count} -> 过滤后: {filtered_count}")
    print(f"   (丢弃了 {original_count - filtered_count} 条负样本)")
else:
    print("⚠️ 警告：没找到 'y' 列！将使用所有行作为连边（如果数据含负样本，这会导致模型失效）。")

# 3. 映射 ID
# 你的文件前两列是 drug_a, drug_b
col_source = 'drug_a'
col_target = 'drug_b'

print(f"   正在映射 ID: {col_source} -> source, {col_target} -> target")
df_edges['source'] = df_edges[col_source].map(id_map)
df_edges['target'] = df_edges[col_target].map(id_map)

# 4. 丢弃匹配不上的行 (比如有些药不在 nodes 表里)
before_drop = len(df_edges)
df_edges.dropna(subset=['source', 'target'], inplace=True)
dropped = before_drop - len(df_edges)
if dropped > 0:
    print(f"⚠️ 另外丢弃了 {dropped} 条连边（因为 ID 在节点表中找不到）。")

# 5. 转换为 PyTorch Edge Index
src = df_edges['source'].values.astype(int)
dst = df_edges['target'].values.astype(int)

# 确保是无向图 (A-B 和 B-A 都要有) -- 可选，PyG 训练代码里通常会自动处理，但这里加上保险
# edge_index = torch.tensor([src, dst], dtype=torch.long) 
# 为了严谨，我们通常不需要手动双向，因为 GCNConv 默认处理有向，RandomLinkSplit 会处理无向。
# 这里保持原样即可：
edge_index = torch.tensor([src, dst], dtype=torch.long)

print(f"\n>>> 最终连边数量: {edge_index.shape[1]}")
torch.save(edge_index, OUTPUT_FILE)
print(f"✅ 修正版 edge_index.pt 已保存！")
