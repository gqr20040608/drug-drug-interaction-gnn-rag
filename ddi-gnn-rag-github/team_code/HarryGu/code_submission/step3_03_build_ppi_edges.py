import pandas as pd
import torch
import gzip
import os
from tqdm import tqdm

# ================= 配置 =================
FILE_MAPPING = 'string_to_internal_id.csv'       # 桥接表
FILE_STRING_NET = '9606.protein.links.v12.0.txt.gz' # STRING 原始网络
OUTPUT_FILE = 'edge_index_ppi.pt'                # 输出结果

# 【关键修改】提高阈值 (High Confidence)
# 400 = Medium (之前的设置)
# 700 = High (现在的设置，只保留高可信度连边)
SCORE_THRESHOLD = 700 
# =======================================

# 1. 加载映射字典
print(f">>> 正在读取映射表 {FILE_MAPPING} ...")
if not os.path.exists(FILE_MAPPING):
    print("❌ 错误：找不到桥接表！")
    exit()

df_map = pd.read_csv(FILE_MAPPING)
string_to_int = dict(zip(df_map['string_id'], df_map['protein_id']))
valid_string_ids = set(string_to_int.keys())

print(f"   有效的 STRING ID 数量: {len(valid_string_ids)}")

# 2. 扫描并清洗 STRING 网络
print(f">>> 正在扫描 STRING 网络 (Threshold >= {SCORE_THRESHOLD}, 去自环)...")

edges_src = []
edges_dst = []
count_total = 0
count_kept = 0
count_self_loop = 0

with gzip.open(FILE_STRING_NET, 'rt') as f:
    header = f.readline()
    
    for line in tqdm(f, desc="Filtering High-Conf Edges"):
        count_total += 1
        
        parts = line.strip().split()
        if len(parts) < 3: continue
        
        p1, p2, score = parts[0], parts[1], int(parts[2])
        
        # --- 清洗逻辑 ---
        
        # 1. 过滤低置信度
        if score < SCORE_THRESHOLD: 
            continue
            
        # 2. 过滤自环 (Self-loop)
        # 蛋白自己连自己对 GNN 消息传递通常没有额外贡献，反而增加计算量
        if p1 == p2:
            count_self_loop += 1
            continue
        
        # 3. 确保两个蛋白都在我们的名单里
        if p1 in valid_string_ids and p2 in valid_string_ids:
            id1 = string_to_int[p1]
            id2 = string_to_int[p2]
            
            # 添加无向边 (双向)
            edges_src.extend([id1, id2])
            edges_dst.extend([id2, id1])
            
            count_kept += 1

print(f"\n>>> 清洗完成！")
print(f"   原始扫描: {count_total} 条")
print(f"   剔除自环: {count_self_loop} 条")
print(f"   最终保留: {len(edges_src)} 条 (双向) | 阈值 > {SCORE_THRESHOLD}")

if len(edges_src) == 0:
    print("⚠️ 警告：没有提取到任何边！可能是阈值太高了。")
else:
    # 3. 转换为 Tensor 并去重
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # 再次去重 (防止原始文件里有重复行)
    # dim=1 表示按列去重，即去掉重复的 (src, dst) 对
    edge_index = torch.unique(edge_index, dim=1)
    
    print(f"   最终 PPI 连边形状 (去重后): {edge_index.shape}")
    
    torch.save(edge_index, OUTPUT_FILE)
    print(f"✅ 成功！高质量 PPI 连边已保存为: {OUTPUT_FILE}")
