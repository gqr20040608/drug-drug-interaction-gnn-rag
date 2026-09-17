import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
FILE_DRUG_FEAT = 'drug_features_combined.pt'
FILE_PROT_FEAT = 'protein_features.pt'
FILE_DDI = 'edge_index.pt'
FILE_DT  = 'edge_index_dt.pt'
FILE_PPI = 'edge_index_ppi.pt'

MODEL_SAVE_PATH = 'model_final.pt'
FIG_ROC = 'roc_curve.png'
FIG_PR = 'pr_curve.png'

HIDDEN_CHANNELS = 64
EPOCHS = 100
LR = 0.001
# =======================================

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f">>> 使用设备: {device}")

# 1. 加载数据
print(">>> 正在加载数据...")
data = HeteroData()
data['drug'].x = torch.load(FILE_DRUG_FEAT)
data['protein'].x = torch.load(FILE_PROT_FEAT)
data['drug', 'interacts', 'drug'].edge_index = torch.load(FILE_DDI)
data['drug', 'targets', 'protein'].edge_index = torch.load(FILE_DT)
data['protein', 'ppi', 'protein'].edge_index = torch.load(FILE_PPI)

src, dst = data['drug', 'targets', 'protein'].edge_index
data['protein', 'rev_targets', 'drug'].edge_index = torch.stack([dst, src], dim=0)

# 2. 切分数据集
print(">>> 切分数据集...")
transform = RandomLinkSplit(
    num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False,
    edge_types=[('drug', 'interacts', 'drug')],
    rev_edge_types=[('drug', 'interacts', 'drug')],
)
train_data, val_data, test_data = transform(data)

# 3. 定义模型
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.gnn = GNN(hidden_channels, out_channels)
        self.model = to_hetero(self.gnn, metadata, aggr='sum')
        self.lin_drug = torch.nn.Linear(data['drug'].x.shape[1], hidden_channels)
        self.lin_prot = torch.nn.Linear(data['protein'].x.shape[1], hidden_channels)

    def encode(self, x_dict, edge_index_dict):
        x_dict['drug'] = self.lin_drug(x_dict['drug']).relu()
        x_dict['protein'] = self.lin_prot(x_dict['protein']).relu()
        return self.model(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index):
        z = z_dict['drug']
        src, dst = edge_label_index[0], edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)

model = HeteroGNN(HIDDEN_CHANNELS, 64, data.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# 4. 训练
def train():
    model.train()
    optimizer.zero_grad()
    x_dict = {k: v.to(device) for k, v in train_data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in train_data.edge_index_dict.items()}
    z_dict = model.encode(x_dict, edge_index_dict)
    
    pos_edge_index = train_data['drug', 'interacts', 'drug'].edge_label_index.to(device)
    pos_out = model.decode(z_dict, pos_edge_index)
    neg_edge_index = torch.randint(0, data['drug'].num_nodes, (2, pos_edge_index.size(1)), device=device)
    neg_out = model.decode(z_dict, neg_edge_index)
    
    loss = criterion(torch.cat([pos_out, neg_out]), torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]))
    loss.backward()
    optimizer.step()
    return loss.item()

print(f">>> 开始最终训练 ({EPOCHS} Epochs)...")
for epoch in range(1, EPOCHS + 1):
    loss = train()
    if epoch % 20 == 0:
        print(f"   Epoch {epoch}: Loss {loss:.4f}")

# 5. 保存
print(f">>> 保存模型权重到 {MODEL_SAVE_PATH} ...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# 6. 【修正版】评估与绘图
print(">>> 正在生成评估图表...")
model.eval()
with torch.no_grad():
    x_dict = {k: v.to(device) for k, v in test_data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in test_data.edge_index_dict.items()}
    
    # 获取 Embedding
    z_dict = model.encode(x_dict, edge_index_dict)
    
    # 核心修正：直接使用 test_data 里已经分配好的 edge_label_index 和 edge_label
    # 这里面已经包含了正样本和负样本
    edge_label_index = test_data['drug', 'interacts', 'drug'].edge_label_index.to(device)
    edge_label = test_data['drug', 'interacts', 'drug'].edge_label.cpu().numpy()
    
    # 预测
    out = model.decode(z_dict, edge_label_index)
    y_score = out.sigmoid().cpu().numpy()
    y_true = edge_label

# 画 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(FIG_ROC)
print(f"✅ ROC 曲线已保存: {FIG_ROC} (AUC={roc_auc:.4f})")

# 画 PR 曲线
precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig(FIG_PR)
print(f"✅ PR 曲线已保存: {FIG_PR} (AUC={pr_auc:.4f})")
