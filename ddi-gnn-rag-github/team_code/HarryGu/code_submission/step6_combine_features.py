import torch
import os

# ================= 配置 =================
FILE_BERT = 'drug_features.pt'          # ChemBERTa 特征
FILE_MORDRED = 'drug_features_mordred.pt' # Mordred 特征
OUTPUT_FILE = 'drug_features_combined.pt' # 输出文件
# =======================================

print(">>> 正在加载两套特征...")
if not os.path.exists(FILE_BERT) or not os.path.exists(FILE_MORDRED):
    print("❌ 错误：找不到特征文件，请确认 Step 2 和 Step 5 都跑完了。")
    exit()

x1 = torch.load(FILE_BERT)
x2 = torch.load(FILE_MORDRED)

print(f"   ChemBERTa 形状: {x1.shape}")
print(f"   Mordred   形状: {x2.shape}")

# 检查行数是否一致
if x1.shape[0] != x2.shape[0]:
    print(f"❌ 严重错误！行数不匹配 ({x1.shape[0]} vs {x2.shape[0]})")
    print("   这意味着两次特征提取时，药物 ID 的排序可能不一致，或者中间删改了数据。")
    exit()

# 归一化 Mordred 特征 (非常重要！)
# Mordred 的数值范围很大 (比如分子量几百，而 BERT 是小数)，直接拼会导致模型训练不稳定。
# 我们对 Mordred 做简单的 Z-Score 归一化。
print(">>> 正在归一化 Mordred 特征...")
mean = x2.mean(dim=0)
std = x2.std(dim=0)
# 防止除以 0 (有些列可能全是 0)
std[std == 0] = 1.0 
x2_norm = (x2 - mean) / std

# 拼接
print(">>> 正在拼接特征 (Concatenation)...")
x_combined = torch.cat([x1, x2_norm], dim=1)

print(f"\n>>> 最终超级特征形状: {x_combined.shape}")
print(f"    (应该是 [3256, {x1.shape[1] + x2.shape[1]}])")

torch.save(x_combined, OUTPUT_FILE)
print(f"✅ 成功！组合特征已保存为: {OUTPUT_FILE}")
