import numpy as np
import pandas as pd
import scipy.sparse as sp
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
            line=line.strip()
            if '|' in line:
                uid,_ = line.split('|'); uid=int(uid)
            else:
                iid, sc = line.split(); users.append(uid)
                items.append(int(iid)); scores.append(float(sc))
    return pd.DataFrame({'user':users,'item':items,'score':scores})

raw_df = load_train('data/train.txt')
user_list = raw_df['user'].unique().tolist()
item_list = raw_df['item'].unique().tolist()
user2idx = {u:i for i,u in enumerate(user_list)}
item2idx = {i:j for j,i in enumerate(item_list)}
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

# 3. LightGCN + Regression Head
class LightGCNReg(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, n_layers=3, device='cpu'):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.emb_dim, self.n_layers = emb_dim, n_layers
        self.device = device

        # 基础嵌入
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # 全局均值 μ（训练前计算并传入）
        self.mu = 0.0

        # 归一化邻接矩阵占位
        self.norm_adj = None

    def build_adj(self, interactions):
        # interactions: list of (u_idx, i_idx, r_ui)
        u_idx = [u for u, i, r in interactions]
        i_idx = [i for u, i, r in interactions]
        vals  = [r for u, i, r in interactions]  # 用真实评分作为边权

        R = sp.coo_matrix((vals, (u_idx, i_idx)),
                          shape=(self.n_users, self.n_items))
        upper = sp.hstack([sp.csr_matrix((self.n_users,self.n_users)), R])
        lower = sp.hstack([R.T, sp.csr_matrix((self.n_items,self.n_items))])
        A = sp.vstack([upper, lower]).tocoo()

        rowsum = np.array(A.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum>0)
        D = sp.diags(d_inv)
        normA = D.dot(A).dot(D).tocoo()

        idx = torch.LongTensor([normA.row, normA.col])
        vals = torch.FloatTensor(normA.data)
        self.norm_adj = torch.sparse_coo_tensor(
            idx, vals, torch.Size(normA.shape)
        ).to(self.device)

    def forward(self):
        # 初始嵌入
        all_emb = torch.cat([self.user_emb.weight,
                             self.item_emb.weight], dim=0)
        embs = [all_emb]

        # K 层图卷积
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        # 平均
        out = torch.stack(embs, dim=1).mean(dim=1)
        users, items = torch.split(out,
                                   [self.n_users, self.n_items], dim=0)
        return users, items

    def predict_batch(self, u, i):
        # u, i: LongTensor batch
        users, items = self.forward()
        emb_u = users[u]
        emb_i = items[i]
        bu   = self.user_bias(u).squeeze()
        bi   = self.item_bias(i).squeeze()
        dot  = (emb_u * emb_i).sum(dim=1)
        return self.mu + bu + bi + dot

# 4. 训练 & 评估
device = torch.device('cpu')
model = LightGCNReg(len(user_list), len(item_list),
                    emb_dim=32, n_layers=2, device=device).to(device)

# 全局均值 μ
model.mu = train_df['score'].mean()

# 构建邻接
model.build_adj(train_df[['u_idx','i_idx','score']].values.tolist())

opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
crit = nn.MSELoss()

best_rmse, trials, patience = float('inf'), 0, 3

for epoch in range(1, 21):
    # --- 训练 ---
    model.train()
    total_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model.predict_batch(u, i)
        loss = crit(pred, r)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # --- 验证 ---
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for u, i, r in val_loader:
            u,i = u.to(device), i.to(device)
            p = model.predict_batch(u, i).cpu().numpy()
            ps.extend(p.tolist()); ys.extend(r.numpy().tolist())
    rmse = mean_squared_error(ys, ps, squared=False)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse, trials = rmse, 0
        torch.save(model.state_dict(), 'best_lightgcn_reg.pth')
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# 5. 测试集预测

# 加载最佳模型
state = torch.load('best_lightgcn_reg.pth', map_location=device)
model.load_state_dict(state)

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

test_df = load_test('data/test.txt')
print(f"test_df: {len(test_df)}")
print("测试集用户数=", test_df['user'].nunique())
test_df['u_idx'] = test_df['user'].map(user2idx).fillna(-1).astype(int)
test_df['i_idx'] = test_df['item'].map(item2idx).fillna(-1).astype(int)

preds = []
model.eval()
with torch.no_grad():
    users_emb, items_emb = model()
    for _, row in test_df.iterrows():
        u, i = row['u_idx'], row['i_idx']
        if u >= 0 and i >= 0:
            score = (users_emb[u] * items_emb[i]).sum().item()
        else:
            score = raw_df['score'].mean()
        preds.append(score)

test_df['score'] = preds
test_df.to_csv('test_lightgcn_predictions_reg.csv', index=False)

with open('test_lightgcn_predictions_reg.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            f.write(f"{row['item']} {row['score']}\n")
