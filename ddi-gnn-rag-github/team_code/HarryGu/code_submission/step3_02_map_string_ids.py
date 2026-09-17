import pandas as pd
import gzip
import os

# ================= 配置 =================
FILE_NODES = 'nodes_protein.csv'                  # 你的蛋白主表
FILE_ALIASES = '9606.protein.aliases.v12.0.txt.gz' # STRING 别名文件
OUTPUT_MAPPING = 'string_to_internal_id.csv'      # 输出的桥接表
# =======================================

# 1. 读取你自己的蛋白列表 (Target UniProt IDs)
print(f">>> 正在读取 {FILE_NODES} ...")
if not os.path.exists(FILE_NODES):
    print("❌ 错误：找不到蛋白主表！")
    exit()

df_nodes = pd.read_csv(FILE_NODES)
# 建立 UniProt -> Internal ID 的查找表
# 假设你的列名是 'uniprot_id' 和 'protein_id'
if 'uniprot_id' not in df_nodes.columns:
    print("❌ 错误：nodes_protein.csv 里没有 'uniprot_id' 列。")
    exit()

# 制作一个集合方便快速查找
target_uniprots = set(df_nodes['uniprot_id'].astype(str))
# 制作一个字典方便最后转换: UniProt -> Internal ID
uniprot_to_internal = dict(zip(df_nodes['uniprot_id'].astype(str), df_nodes['protein_id']))

print(f"   你的图中共有 {len(target_uniprots)} 个目标 UniProt ID。")

# 2. 扫描 STRING 别名文件，寻找匹配
print(f">>> 正在扫描别名文件 {FILE_ALIASES} (寻找匹配)...")
if not os.path.exists(FILE_ALIASES):
    print("❌ 错误：找不到别名文件，请先运行 curl 下载命令。")
    exit()

mapping_list = []
matched_uniprots = set()

# 使用 gzip 打开压缩文件
with gzip.open(FILE_ALIASES, 'rt') as f:
    # 跳过表头 (如果有的话，STRING alias 文件通常第一行就是数据，或者以 # 开头)
    for line in f:
        if line.startswith('#'): continue
        
        parts = line.strip().split('\t')
        # 格式通常是: string_protein_id, alias, source
        # 例如: 9606.ENSP00000000233, P00533, UniProt_AC
        
        if len(parts) < 3: continue
        
        string_id = parts[0]
        alias = parts[1]
        source = parts[2]
        
        # 我们只关心来源是 "UniProt_AC" (Accession) 的记录
        # 并且 alias 必须在我们的目标列表里
        if source == 'UniProt_AC' and alias in target_uniprots:
            internal_id = uniprot_to_internal[alias]
            mapping_list.append({
                'string_id': string_id,   # STRING 的 ID (9606.ENSP...)
                'uniprot_id': alias,      # 你的 UniProt ID
                'protein_id': internal_id # 你的数字 ID
            })
            matched_uniprots.add(alias)

# 3. 保存映射结果
df_map = pd.DataFrame(mapping_list)

# 去重：有时候 STRING 会把同一个 ENSP 对应到同一个 UniProt 多次，或者反过来
# 我们只需要 string_id -> protein_id 的唯一映射
df_map.drop_duplicates(subset=['string_id'], inplace=True)

print(f"\n>>> 扫描完成！")
print(f"   找到 {len(df_map)} 个有效的 STRING ID 映射。")
print(f"   覆盖了 {len(matched_uniprots)} / {len(target_uniprots)} 个你的 UniProt ID。")

missed = len(target_uniprots) - len(matched_uniprots)
if missed > 0:
    print(f"   ⚠️ 有 {missed} 个蛋白没在 STRING 里找到对应的 ENSP ID (将成为图中的孤立点)。")

df_map.to_csv(OUTPUT_MAPPING, index=False)
print(f"✅ 桥接表已保存为: {OUTPUT_MAPPING}")
print(df_map.head())
