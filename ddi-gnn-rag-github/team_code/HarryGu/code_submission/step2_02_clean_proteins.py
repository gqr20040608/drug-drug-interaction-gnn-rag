import pandas as pd
import os

# ================= 配置 =================
INPUT_FILE = 'nodes_protein.csv'
OUTPUT_FILE = 'nodes_protein.csv' # 清洗后直接覆盖，保证后续步骤用的是干净数据
# =======================================

# 定义氨基酸集合
# 20种标准氨基酸
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# 允许的模糊氨基酸 (ESM-2 可以处理这些，通常代表未定或特殊氨基酸)
# B(天冬氨酸/天冬酰胺), Z(谷氨酸/谷氨酰胺), X(未知), U(硒半胱氨酸), O(吡咯赖氨酸), J(亮氨酸/异亮氨酸)
AMBIGUOUS_AA = set("BZXUOJ")

def clean_sequence(seq):
    """
    清洗逻辑：
    1. 转大写
    2. 去除空格、制表符
    3. 检查非法字符
    4. 检查长度
    """
    if pd.isna(seq):
        return None, "Empty/NaN"
    
    # 1. 格式统一
    seq = str(seq).upper().strip()
    
    # 去除常见的非序列字符 (比如星号*代表终止，或连字符-)
    seq = seq.replace("*", "").replace("-", "").replace(" ", "")
    
    if len(seq) == 0:
        return None, "Empty After Clean"

    # 2. 长度检查 (Sanity Check)
    # ESM-2 理论上能处理很长的，但为了显存考虑，太长的(>10000)截断或丢弃
    # 太短的(<10)通常不是有效蛋白
    if len(seq) < 10:
        return None, "Too Short (<10)"
    if len(seq) > 20000:
        return None, "Too Long (>20000)"

    # 3. 字符合法性检查
    unique_chars = set(seq)
    
    # 检查是否包含非法字符 (即不在 Standard 也不在 Ambiguous 里的)
    invalid_chars = unique_chars - (STANDARD_AA | AMBIGUOUS_AA)
    
    if len(invalid_chars) > 0:
        # 比如出现了数字 '1', '9' 或者特殊符号
        return None, f"Invalid Chars: {invalid_chars}"
    
    # 区分“完美序列”和“含模糊字符序列”
    is_standard = unique_chars.issubset(STANDARD_AA)
    status = "Standard" if is_standard else "Ambiguous"
    
    return seq, status

# --- 主流程 ---
print(f">>> 正在读取 {INPUT_FILE} 进行序列清洗...")
if not os.path.exists(INPUT_FILE):
    print("❌ 错误：找不到文件。请先运行上一步下载序列。")
    exit()

df = pd.read_csv(INPUT_FILE)
original_count = len(df)

valid_rows = []
stats = {
    "Standard": 0,    # 纯标准氨基酸
    "Ambiguous": 0,   # 含 X, B, Z 等 (保留)
    "Dropped": 0      # 丢弃
}

drop_reasons = {}

print(">>> 开始逐行检查...")

for idx, row in df.iterrows():
    seq = row['sequence']
    uid = row.get('uniprot_id', f"Row_{idx}")
    
    cleaned_seq, status = clean_sequence(seq)
    
    if cleaned_seq is None:
        # 这一行要丢弃
        stats["Dropped"] += 1
        reason = status
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    else:
        # 这一行保留
        stats[status] += 1
        # 更新清洗后的序列
        row['sequence'] = cleaned_seq
        valid_rows.append(row)

# 生成新 DataFrame
df_clean = pd.DataFrame(valid_rows)

# 【关键】重新生成 protein_id
# 任何清洗步骤只要删了行，就必须重置 ID，否则 edge_index 会报错
if len(df_clean) < original_count:
    print(">>> 检测到行数变化，正在重置 protein_id (0, 1, 2...)...")
    df_clean.reset_index(drop=True, inplace=True)
    df_clean['protein_id'] = df_clean.index

# 保存
df_clean.to_csv(OUTPUT_FILE, index=False)

# --- 打印报告 ---
print("\n" + "="*30)
print("       蛋白质序列清洗报告")
print("="*30)
print(f"原始数量: {original_count}")
print(f"保留数量: {len(df_clean)}")
print(f"丢弃数量: {stats['Dropped']}")
print("-" * 30)
print(f"✅ 标准序列 (纯20种AA): {stats['Standard']}")
print(f"⚠️ 模糊序列 (含X/B/Z):  {stats['Ambiguous']} (ESM-2可处理，已保留)")
print("-" * 30)

if stats['Dropped'] > 0:
    print("丢弃原因统计:")
    for reason, count in drop_reasons.items():
        print(f"   - {reason}: {count} 条")
else:
    print("✨ 完美！没有发现垃圾序列。")

print("="*30)
print(f"清洗后的文件已保存为: {OUTPUT_FILE}")
