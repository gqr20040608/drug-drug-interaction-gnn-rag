import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = 'nodes_protein.csv'       # 刚刚清洗干净的文件
OUTPUT_FILE = 'protein_features.pt'    # 输出特征文件

# 选用 Meta 的 ESM-2 模型 (8M 参数版，速度快，维度 320)
# 如果你电脑配置很高，可以换更大的版本 (比如 esm2_t12_35M_UR50D)，但 8M 对 Baseline 足够了
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
BATCH_SIZE = 16  # 蛋白质序列较长，显存小的可以调成 8 或 4
# ===========================================

# 1. 设置设备 (Mac M2 推荐使用 mps)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f">>> 使用设备: {device}")

# 2. 读取数据
if not os.path.exists(INPUT_FILE):
    print(f"❌ 错误：找不到 {INPUT_FILE}")
    exit()

print(f">>> 正在读取 {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# 【关键】确保按 protein_id 排序 (0, 1, 2...)
# 必须严格对齐，否则后面构建图会乱套
if 'protein_id' in df.columns:
    df.sort_values('protein_id', inplace=True)
    print("✅ 已按 protein_id 排序。")
else:
    print("⚠️ 警告：没找到 protein_id，默认按行顺序处理。")

sequences = df['sequence'].tolist()
print(f"   待处理序列数量: {len(sequences)}")

# 3. 加载 ESM-2 模型
print(f">>> 正在加载模型 {MODEL_NAME} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("   请检查网络，首次运行需要连接 HuggingFace 下载模型。")
    exit()

# 4. 批量生成特征
print(f">>> 开始生成特征 (Output Dim: 320)...")
all_embeddings = []

for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
    batch_seqs = sequences[i : i + BATCH_SIZE]
    
    # Tokenize
    # max_length=1024: 截断超长序列 (大多数蛋白都在 1000 以内)
    inputs = tokenizer(batch_seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 提取特征：Mean Pooling (求平均)
        # 排除 padding 部分的影响
        hidden_states = outputs.last_hidden_state # [batch, len, dim]
        mask = inputs['attention_mask'].unsqueeze(-1) # [batch, len, 1]
        
        # 加权平均
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        all_embeddings.append(mean_embeddings.cpu().numpy())

# 5. 保存结果
if len(all_embeddings) > 0:
    final_features = np.vstack(all_embeddings)
    x_tensor = torch.tensor(final_features, dtype=torch.float32)

    print(f"\n>>> 最终蛋白质特征形状: {x_tensor.shape}")
    print(f"    (预期: [{len(df)}, 320])")

    torch.save(x_tensor, OUTPUT_FILE)
    print(f"✅ 成功！蛋白质特征已保存为: {OUTPUT_FILE}")
else:
    print("❌ 生成失败，特征列表为空。")
