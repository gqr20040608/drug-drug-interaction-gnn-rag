import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import os

# ================= 配置 =================
INPUT_FILE = 'nodes_drug_final.csv'
OUTPUT_FILE = 'drug_features_mordred.pt'
# =======================================

# 1. 加载数据
print(f">>> 正在读取 {INPUT_FILE}...")
if not os.path.exists(INPUT_FILE):
    print("❌ 错误：找不到文件。")
    exit()

df = pd.read_csv(INPUT_FILE)

# 按 drug_id 排序
if 'drug_id' in df.columns:
    df.sort_values('drug_id', inplace=True)
    print("✅ 已按 drug_id 排序。")

# 2. 准备分子对象
print(">>> 正在将 SMILES 转为 RDKit 分子对象...")
mols = [Chem.MolFromSmiles(s) for s in df['smiles']]

# 3. 初始化计算器 (只算 2D)
calc = Calculator(descriptors, ignore_3D=True)
print(f">>> Mordred 计算器已就绪，准备计算 {len(calc.descriptors)} 个描述符...")

# 4. 批量计算
# 【核心修正】nproc=1 表示强制使用单进程。
# 这解决了 Mac 上的 EOFError/BrokenPipeError 问题。
print(">>> 开始计算描述符 (nproc=1, 单核模式)...")
# quiet=False 会显示进度条
df_desc = calc.pandas(mols, nproc=1, quiet=False)

print(f"   原始特征形状: {df_desc.shape}")

# 5. 数据清洗
print(">>> 开始清洗特征矩阵...")

# 转为数字，错误变 NaN
df_desc = df_desc.apply(pd.to_numeric, errors='coerce')

# 删掉全空列
original_cols = df_desc.shape[1]
df_desc.dropna(axis=1, how='all', inplace=True)
print(f"   剔除了 {original_cols - df_desc.shape[1]} 个全空列。")

# 填充剩余 NaN
df_desc.fillna(0, inplace=True)

# 6. 保存
x_tensor = torch.tensor(df_desc.values.astype(np.float32))

print(f"\n>>> 最终 Mordred 矩阵形状: {x_tensor.shape}")
torch.save(x_tensor, OUTPUT_FILE)
print(f"✅ 成功！文件已保存为: {OUTPUT_FILE}")
