import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------
# 1. 数据加载与映射
# -------------------------
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

raw_df = load_train('data/train.txt')
user_list = raw_df['user'].unique().tolist()
item_list = raw_df['item'].unique().tolist()
user2idx = {u:i for i,u in enumerate(user_list)}
item2idx = {i:j for j,i in enumerate(item_list)}
raw_df['u_idx'] = raw_df['user'].map(user2idx)
raw_df['i_idx'] = raw_df['item'].map(item2idx)

# -------------------------
# 2. 数据集与 DataLoader
# -------------------------
class RatingDataset(Dataset):
    def __init__(self, df):
        self.u = torch.LongTensor(df['u_idx'].values)
        self.i = torch.LongTensor(df['i_idx'].values)
        self.r = torch.FloatTensor(df['score'].values)
    def __len__(self): return len(self.r)
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
print(f"train_df: {len(train_df)}, val_df: {len(val_df)}")
print("训练集用户数 =", train_df['user'].nunique())
print("验证集用户数=", val_df['user'].nunique())
train_loader = DataLoader(RatingDataset(train_df), batch_size=1024, shuffle=True)
val_loader   = DataLoader(RatingDataset(val_df  ), batch_size=1024, shuffle=False)

# -------------------------
# 3. LightGCN 模型定义
# -------------------------
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # 基础嵌入
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # 邻接（和归一化邻接）矩阵占位
        self.norm_adj = None

    def build_adj(self, interactions):
        """
        interactions: list of (u_idx, i_idx, rating)
        这里 rating 不参与图计算，只用作边存在标记
        """
        u_idx = [u for u, i, _ in interactions]
        i_idx = [i for u, i, _ in interactions]
        values = np.ones(len(u_idx), dtype=np.float32)

        R = sp.coo_matrix((values, (u_idx, i_idx)),
                          shape=(self.n_users, self.n_items))
        # 构造二部图 A
        upper = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), R])
        lower = sp.hstack([R.T, sp.csr_matrix((self.n_items, self.n_items))])
        A = sp.vstack([upper, lower]).tocoo()

        # 归一化: D^-1/2 A D^-1/2
        rowsum = np.array(A.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum>0)
        d_mat = sp.diags(d_inv_sqrt)
        normA = d_mat.dot(A).dot(d_mat).tocoo()

        # 转成 PyTorch 稀疏 tensor
        indices = torch.LongTensor([normA.row, normA.col])
        values = torch.FloatTensor(normA.data)
        self.norm_adj = torch.sparse.FloatTensor(indices, values, 
                                  torch.Size(normA.shape)).to(self.user_emb.weight.device)

    def forward(self):
        """
        单次图卷积：输出融合所有层后的用户和物品最终嵌入
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        # 多层平均
        embs = torch.stack(embs, dim=1).mean(dim=1)
        users, items = torch.split(embs, [self.n_users, self.n_items], dim=0)
        return users, items

# -------------------------
# 4. 训练与评估
# -------------------------
device = torch.device('cpu')
model = LightGCN(len(user_list), len(item_list), emb_dim=32, n_layers=3).to(device)
model.build_adj(train_df[['u_idx','i_idx','score']].values.tolist())

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

best_rmse, trials, patience = float('inf'), 0, 3

for epoch in range(1, 21):
    # ---- 训练 ----
    model.train()
    total_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        users_emb, items_emb = model()
        pred = (users_emb[u] * items_emb[i]).sum(dim=1)
        loss = criterion(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # ---- 验证 ----
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        users_emb, items_emb = model()
        for u, i, r in val_loader:
            emb_u = users_emb[u.to(device)]
            emb_i = items_emb[i.to(device)]
            pred = (emb_u * emb_i).sum(dim=1).cpu().numpy()
            ps.extend(pred.tolist())
            ys.extend(r.numpy().tolist())
    rmse = mean_squared_error(ys, ps, squared=False)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}")

    # 早停
    if rmse < best_rmse:
        best_rmse, trials = rmse, 0
        torch.save(model.state_dict(), 'best_lightgcn.pth')
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping.")
            break

# -------------------------
# 5. 测试集预测
# -------------------------
# 加载最佳模型
state = torch.load('best_lightgcn.pth', map_location=device)
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
test_df.to_csv('test_lightgcn_predictions.csv', index=False)

with open('test_lightgcn_predictions.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            f.write(f"{row['item']} {row['score']}\n")
