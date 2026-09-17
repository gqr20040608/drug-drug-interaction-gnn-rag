import pandas as pd
import os

# 1. 检查输入文件是否存在
input_file = 'nodes_drug_std.csv'
if not os.path.exists(input_file):
    print(f"❌ 错误：找不到 {input_file}！")
    print("   请确认你之前运行过 step1_clean_drugs.py，并且文件就在当前目录下。")
    exit()

print(f">>> 正在读取 {input_file}...")
df = pd.read_csv(input_file)

# 2. 修改 ID：把旧的 drug_id 改名为 drugbank_id 保留备份
if 'drug_id' in df.columns:
    df.rename(columns={'drug_id': 'drugbank_id'}, inplace=True)

# 3. 生成新 ID：使用 0, 1, 2... 的行号作为 drug_id
df.reset_index(drop=True, inplace=True)
df['drug_id'] = df.index 

# 4. 调整列顺序
cols = ['drug_id', 'smiles']
if 'drugbank_id' in df.columns:
    cols.append('drugbank_id')
# 把其他列也加上
for c in df.columns:
    if c not in cols:
        cols.append(c)

df_final = df[cols]

# 5. 保存最终文件
output_file = 'nodes_drug_final.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ 修复完成！文件已保存为: {output_file}")
print(f"   总行数: {len(df_final)}")
print("   前 5 行预览（确认 drug_id 是 0, 1, 2...）：")
print(df_final.head())
