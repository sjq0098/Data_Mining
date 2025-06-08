import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------
# 1. 数据加载 & 映射
# ---------------------------------------
def load_train(path):
    users, items, scores = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if '|' in line:
                uid, _ = line.split('|'); uid = int(uid)
            else:
                iid, sc = line.split()
                users.append(uid); items.append(int(iid)); scores.append(float(sc))
    return pd.DataFrame({'user':users,'item':items,'score':scores})

def load_test(path):
    users, items = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if '|' in line:
                uid, _ = line.split('|'); uid = int(uid)
            else:
                iid = int(line)
                users.append(uid); items.append(iid)
    return pd.DataFrame({'user':users,'item':items})

# 读取
train_df = load_train('data/train.txt')
test_df  = load_test ('data/test.txt')

# 映射索引
user_list = train_df['user'].unique().tolist()
item_list = train_df['item'].unique().tolist()
user2idx = {u:i for i,u in enumerate(user_list)}
item2idx = {i:j for j,i in enumerate(item_list)}

train_df ['u_idx'] = train_df['user'].map(user2idx)
train_df ['i_idx'] = train_df['item'].map(item2idx)
test_df  ['u_idx'] = test_df['user'].map(user2idx).fillna(-1).astype(int)
test_df  ['i_idx'] = test_df['item'].map(item2idx).fillna(-1).astype(int)

# ---------------------------------------
# 2. Dataset & DataLoader
# ---------------------------------------
class RatingDataset(Dataset):
    def __init__(self, df, with_score=True):
        self.u = torch.LongTensor(df['u_idx'].values)
        self.i = torch.LongTensor(df['i_idx'].values)
        self.with_score = with_score
        if with_score:
            self.r = torch.FloatTensor(df['score'].values)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        if self.with_score:
            return self.u[idx], self.i[idx], self.r[idx]
        else:
            return self.u[idx], self.i[idx]

# 全量训练不再划分验证
train_loader = DataLoader(RatingDataset(train_df), batch_size=512, shuffle=True)

# ---------------------------------------
# 3. 模型定义（与之前相同）
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
# 4. 初始化模型 & 优化器
# ---------------------------------------
device = torch.device('cpu')
model = HybridCF(
    num_users = len(user_list),
    num_items = len(item_list),
    embed_dim = 16,     # grid search 最优
    hidden_dim= 128,    # grid search 最优
    dropout   = 0.2     # grid search 最优
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # lr=0.001

# ---------------------------------------
# 5. 全量训练
# ---------------------------------------
n_epochs = 20
for epoch in range(1, n_epochs+1):
    model.train()
    total_loss = 0
    for u, i, r in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i)
        loss = criterion(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} | RMSE: {np.sqrt(total_loss/len(train_loader)):.4f}")


# 保存最终模型
torch.save(model.state_dict(), 'hybrid_cf_final.pth')

# ---------------------------------------
# 6. 测试集预测
# ---------------------------------------
test_ds = RatingDataset(test_df, with_score=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

model.eval()
preds = []
with torch.no_grad():
    for u, i in test_loader:
        u, i = u.to(device), i.to(device)
        # 对映射失败（u=-1 or i=-1）直接返回全局均值
        mask = (u >= 0) & (i >= 0)
        batch_pred = torch.zeros(len(u), device=device)
        if mask.any():
            batch_pred[mask] = model(u[mask], i[mask])
        # 其他位置保留 0，后面再替换
        preds.extend(batch_pred.cpu().tolist())

# 用训练集全局均值替换映射失败的预测
global_mean = train_df['score'].mean()
preds = [ (p if (u>=0 and i>=0) else global_mean)
          for p,(u,i) in zip(preds, zip(test_df['u_idx'], test_df['i_idx'])) ]

# 写入 test_predictions.txt
with open('test_hybrid_cf_predictions.txt','w',encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            f.write(f"{row['item']}  {preds.pop(0):.4f}\n")

print("测试集预测完成，结果保存在 test_hybrid_cf_predictions.txt")

test_df.to_csv('test_hybrid_cf_predictions.csv', index=False)