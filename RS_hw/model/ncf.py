import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_train(path):
    user_ids, item_ids, scores = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, _ = line.split('|')
                uid = int(uid)
            else:
                iid, sc = line.split()
                user_ids.append(uid)
                item_ids.append(int(iid))
                scores.append(float(sc))
    return pd.DataFrame({'user': user_ids, 'item': item_ids, 'score': scores})



class InteractionDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        # df 包含 ['user','item','score']，已映射索引
        self.users = torch.tensor(df['u_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['i_idx'].values, dtype=torch.long)
        self.scores = torch.tensor(df['score'].values, dtype=torch.float)
    def __len__(self):
        return len(self.scores)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.scores[idx]

# 2. 读取数据和映射
raw_df = load_train('data/train.txt')  # 重用之前定义的 load_train
user_list = raw_df['user'].unique().tolist()
item_list = raw_df['item'].unique().tolist()
user2idx = {u:i for i,u in enumerate(user_list)}
item2idx = {i:j for j,i in enumerate(item_list)}
raw_df['u_idx'] = raw_df['user'].map(user2idx)
raw_df['i_idx'] = raw_df['item'].map(item2idx)

# 3. 划分数据集
train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
print(f"train_df: {len(train_df)}, val_df: {len(val_df)}")
print("训练集用户数 =", train_df['user'].nunique())
print("验证集用户数=", val_df['user'].nunique())
# 构建 Dataset
train_ds = InteractionDataset(train_df, user2idx, item2idx)
val_ds   = InteractionDataset(val_df, user2idx, item2idx)
# DataLoader
batch_size = 1024
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# 4. 定义 PyTorch NCF 模型
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, mlp_layers=[64,32,16], dropout=0.2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        layers = []
        input_dim = embed_dim*2
        for units in mlp_layers:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = units
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, u, i):
        u_vec = self.user_emb(u)
        i_vec = self.item_emb(i)
        x = torch.cat([u_vec, i_vec], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out

# 5. 初始化模型、损失和优化器
device = torch.device('cpu')
model = NCF(num_users=len(user_list), num_items=len(item_list)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 训练与验证循环
best_rmse = float('inf')
patience, trials = 3, 0
for epoch in range(20):
    # 训练
    model.train()
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i)
        loss = criterion(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 验证
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i = u.to(device), i.to(device)
            pred = model(u, i).cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(r.numpy().tolist())
    rmse = mean_squared_error(ys, preds, squared=False)
    print(f"Epoch {epoch+1}, Validation RMSE: {rmse:.4f}")
    # 早停
    if rmse < best_rmse:
        best_rmse = rmse
        trials = 0
        torch.save(model.state_dict(), 'best_ncf_cpu.pth')
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# 加载最佳模型
state_dict = torch.load('best_ncf_cpu.pth', map_location='cpu')
model.load_state_dict(state_dict)


def load_test(path):
    user_ids, item_ids = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, _ = line.split('|')
                uid = int(uid)
            else:
                iid = int(line)
                user_ids.append(uid)
                item_ids.append(iid)
    return pd.DataFrame({'user': user_ids, 'item': item_ids})

# 7. 测试集预测
test_df = load_test('data/test.txt')
test_df['u_idx'] = test_df['user'].map(user2idx).fillna(-1).astype(int)
test_df['i_idx'] = test_df['item'].map(item2idx).fillna(-1).astype(int)
preds = []
model.eval()
with torch.no_grad():
    for _, row in test_df.iterrows():
        u, i = row['u_idx'], row['i_idx']
        if u >= 0 and i >= 0:
            input_u = torch.tensor([u], dtype=torch.long).to(device)
            input_i = torch.tensor([i], dtype=torch.long).to(device)
            score = model(input_u, input_i).item()
        else:
            score = raw_df['score'].mean()
        preds.append(score)
test_df['score'] = preds
test_df.to_csv('test_ncf_cpu_predictions.csv', index=False)  # 保存预测

with open('test_ncf_cpu_predictions.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        # 写入用户行，格式为 "<user>|<该用户预测评分的物品数量>"
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            # 写入物品及预测评分行保持浮点数格式
            f.write(f"{row['item']}  {row['score']}\n")