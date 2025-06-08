import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 数据加载 & 映射
def load_train(path):
    users, items, scores = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, _ = line.split('|'); uid = int(uid)
            else:
                iid, sc = line.split()
                users.append(uid)
                items.append(int(iid))
                scores.append(float(sc))
    return pd.DataFrame({'user': users, 'item': items, 'score': scores})

raw_df = load_train('data/train.txt')
user_list = raw_df['user'].unique().tolist()
item_list = raw_df['item'].unique().tolist()
user2idx = {u: i for i, u in enumerate(user_list)}
item2idx = {i: j for j, i in enumerate(item_list)}
raw_df['u_idx'] = raw_df['user'].map(user2idx)
raw_df['i_idx'] = raw_df['item'].map(item2idx)

# 2. Dataset & DataLoader
class RatingDataset(Dataset):
    def __init__(self, df):
        self.u = torch.LongTensor(df['u_idx'].values)
        self.i = torch.LongTensor(df['i_idx'].values)
        self.r = torch.FloatTensor(df['score'].values)
    def __len__(self): return len(self.r)
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_loader = DataLoader(RatingDataset(train_df), batch_size=1024, shuffle=True)
val_loader   = DataLoader(RatingDataset(val_df  ), batch_size=1024, shuffle=False)

# 3. Probabilistic Matrix Factorization (PMF)
class PMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, sigma2=0.1, lambda_u=0.1, lambda_v=0.1):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.emb_dim = emb_dim
        # 用户与物品隐向量
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        # 初始化正态分布
        nn.init.normal_(self.user_emb.weight, std=sigma2)
        nn.init.normal_(self.item_emb.weight, std=sigma2)
        # 超参数
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

    def forward(self, u, i):
        pu = self.user_emb(u)  # [B, emb_dim]
        qi = self.item_emb(i)  # [B, emb_dim]
        # 内积预测
        return (pu * qi).sum(dim=1)

    def regularization(self):
        # L2 正则项
        return self.lambda_u * torch.norm(self.user_emb.weight)**2 + \
               self.lambda_v * torch.norm(self.item_emb.weight)**2

# 4. 训练 & 验证

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PMF(len(user_list), len(item_list), emb_dim=32,
            sigma2=0.1, lambda_u=1e-4, lambda_v=1e-4).to(device)
opt = optim.Adam(model.parameters(), lr=0.005)
crit = nn.MSELoss()

best_rmse, patience, trials = float('inf'), 3, 0
for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i)
        # 均方误差 + MAP 正则化
        loss = crit(pred, r) + model.regularization() / (len(train_df))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i = u.to(device), i.to(device)
            p = model(u, i).cpu().tolist()
            ps.extend(p); ys.extend(r.tolist())
    rmse = mean_squared_error(ys, ps, squared=False)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse, trials = rmse, 0
        torch.save(model.state_dict(), 'best_pmf.pth')
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# 5. 测试集预测

def load_test(path):
    user_ids, item_ids = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, _ = line.split('|'); uid = int(uid)
            else:
                iid = int(line)
                user_ids.append(uid)
                item_ids.append(iid)
    return pd.DataFrame({'user': user_ids, 'item': item_ids})

test_df = load_test('data/test.txt')
test_df['u_idx'] = test_df['user'].map(user2idx).fillna(-1).astype(int)
test_df['i_idx'] = test_df['item'].map(item2idx).fillna(-1).astype(int)

# 加载最佳模型并预测
state = torch.load('best_pmf.pth', map_location=device)
model.load_state_dict(state)
model.eval()

preds = []
mean_score = raw_df['score'].mean()
for _, row in test_df.iterrows():
    u, i = row['u_idx'], row['i_idx']
    if u >= 0 and i >= 0:
        u_t = torch.LongTensor([u]).to(device)
        i_t = torch.LongTensor([i]).to(device)
        score = model(u_t, i_t).item()
    else:
        score = mean_score
    preds.append(score)

test_df['score'] = preds
test_df.to_csv('test_pmf_preds.csv', index=False)

with open('test_pmf_preds.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            f.write(f"{row['item']} {row['score']}\n")
