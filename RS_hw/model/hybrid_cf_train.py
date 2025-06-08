import itertools
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------
# 1. 数据加载 & 映射
# ---------------------------------------
def load_train(path):
    users, items, scores = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, _ = line.split('|')
                uid = int(uid)
            else:
                iid, sc = line.split()
                users.append(uid)
                items.append(int(iid))
                scores.append(float(sc))
    return pd.DataFrame({'user': users, 'item': items, 'score': scores})

raw_df = load_train('data/train.txt')
user_list = raw_df['user'].unique().tolist()
item_list = raw_df['item'].unique().tolist()
user2idx = {u:i for i,u in enumerate(user_list)}
item2idx = {i:j for j,i in enumerate(item_list)}
raw_df['u_idx'] = raw_df['user'].map(user2idx)
raw_df['i_idx'] = raw_df['item'].map(item2idx)

train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)

# ---------------------------------------
# 2. Dataset & DataLoader
# ---------------------------------------
class RatingDataset(Dataset):
    def __init__(self, df):
        self.u = torch.LongTensor(df['u_idx'].values)
        self.i = torch.LongTensor(df['i_idx'].values)
        self.r = torch.FloatTensor(df['score'].values)
    def __len__(self):
        return len(self.r)
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

# ---------------------------------------
# 3. Hybrid CF 模型
# ---------------------------------------
class HybridCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, u, i):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        x = torch.cat([u_e, i_e], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(1)

# ---------------------------------------
# 4. 训练 & 验证 函数
# ---------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i)
        loss = criterion(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for u, i, r in loader:
            u, i = u.to(device), i.to(device)
            pred = model(u, i).cpu().numpy()
            ys.extend(r.numpy())
            ps.extend(pred)
    return mean_squared_error(ys, ps, squared=False)

# ---------------------------------------
# 5. 网格搜索设置
# ---------------------------------------
param_grid = {
    'embed_dim': [16, 32, 64],
    'hidden_dim': [32, 64, 128],
    'dropout': [0.1, 0.2],
    'lr': [1e-3, 5e-4],
    'batch_size': [512, 1024]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_rmse = float('inf')
best_params = None
results = []

# 全局数据加载（后面每次只重构 DataLoader）
for embed_dim, hidden_dim, dropout, lr, batch_size in itertools.product(
        param_grid['embed_dim'],
        param_grid['hidden_dim'],
        param_grid['dropout'],
        param_grid['lr'],
        param_grid['batch_size']
    ):
    # DataLoader
    train_loader = DataLoader(RatingDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(RatingDataset(val_df  ), batch_size=batch_size, shuffle=False)

    # 模型、损失、优化器
    model = HybridCF(len(user_list), len(item_list),
                     embed_dim, hidden_dim, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练若干轮（这里取 10 轮，可根据情况增减）
    start = time.time()
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_rmse = evaluate(model, val_loader, device)
    duration = time.time() - start

    # 记录结果
    results.append({
        'embed_dim': embed_dim, 'hidden_dim': hidden_dim,
        'dropout': dropout, 'lr': lr, 'batch_size': batch_size,
        'val_rmse': val_rmse, 'time_s': duration
    })

    # 更新最优
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_params = results[-1]

    print(f"Tested {embed_dim},{hidden_dim},{dropout},{lr},{batch_size} → RMSE {val_rmse:.4f} ({duration:.1f}s)")

# ---------------------------------------
# 6. 输出最优结果
# ---------------------------------------
print("\n>> Best params:")
print(best_params)

# ---------------------------------------
# 7. 结果保存
# ---------------------------------------
df_res = pd.DataFrame(results)
df_res.to_csv('grid_search_results.csv', index=False)
