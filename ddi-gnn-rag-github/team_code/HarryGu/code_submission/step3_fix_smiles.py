import pandas as pd
from rdkit import Chem
import os

INPUT_FILE = 'nodes_drug_final.csv'
OUTPUT_FILE = 'nodes_drug_final.csv' # 直接覆盖，因为我们要修复它

print(f">>> 读取 {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
original_len = len(df)

# 1. 定义标准化函数 (Canonicalize)
def get_canonical_smiles(smiles):
    if pd.isna(smiles) or str(smiles).strip() == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # 返回标准写法
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        return None

# 2. 清洗数据
print(">>> 正在剔除坏数据并标准化所有 SMILES...")
# 应用标准化函数
df['smiles'] = df['smiles'].apply(get_canonical_smiles)

# 3. 丢弃无效行 (即刚才变成 None 的行，比如 ID 2190)
df_clean = df.dropna(subset=['smiles']).copy()
dropped_count = original_len - len(df_clean)

if dropped_count > 0:
    print(f"✅ 已剔除 {dropped_count} 个无效药物 (包括那个 Parse Error 的)。")
else:
    print("✨ 没有发现无效药物。")

# 4. 【关键】重新生成 drug_id
# 因为删了一行，ID 可能会断开 (比如 2189 跳到 2191)。
# GNN 必须要求 ID 连续 (0, 1, 2...)，所以我们必须重置。
print(">>> 重新生成连续的 drug_id (0, 1, 2...)...")
df_clean.reset_index(drop=True, inplace=True)
df_clean['drug_id'] = df_clean.index  # 新的 ID

# 5. 保存
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"✅ 修复完成！文件已覆盖保存为: {OUTPUT_FILE}")
print(f"   最终有效药物数量: {len(df_clean)}")
