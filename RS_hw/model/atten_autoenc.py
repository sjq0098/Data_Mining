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

# 2. 构建用户-物品矩阵及Dataset
class FullRatingDataset(Dataset):
    def __init__(self, df, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.mat = np.zeros((n_users, n_items), dtype=np.float32)
        for u, i, r in df[['u_idx', 'i_idx', 'score']].values:
            self.mat[int(u), int(i)] = r
        self.mask = (self.mat > 0).astype(np.float32)

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.mat[idx]),  # [n_items]
            torch.FloatTensor(self.mask[idx]), # [n_items]
            idx
        )

n_users, n_items = len(user_list), len(item_list)
dataset = FullRatingDataset(raw_df, n_users, n_items)
all_users = list(range(n_users))
train_users, val_users = train_test_split(all_users, test_size=0.2, random_state=42)
train_loader = DataLoader(
    torch.utils.data.Subset(dataset, train_users),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    torch.utils.data.Subset(dataset, val_users),
    batch_size=64, shuffle=False
)

# 3. Attention Autoencoder 模型
class AttnAutoRec(nn.Module):
    def __init__(self, n_items, emb_dim=64, n_heads=4, hidden_dim=256, n_layers=2, dropout=0.2):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        # Transformer 自注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items)
        )

    def forward(self, x):
        # x: [B, n_items]
        z = self.encoder(x)             # [B, emb_dim]
        z_seq = z.unsqueeze(1)          # [B, 1, emb_dim]
        z_attn = self.transformer(z_seq).squeeze(1)  # [B, emb_dim]
        out = self.decoder(z_attn)      # [B, n_items]
        return out

# 4. 训练 & 验证
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttnAutoRec(n_items, emb_dim=64, n_heads=4, hidden_dim=256, n_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss(reduction='sum')
best_rmse, patience, trials = float('inf'), 3, 0

for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for x, mask, _ in train_loader:
        x, mask = x.to(device), mask.to(device)
        pred = model(x)              # [B, n_items]
        # 仅对观察到的位置计算损失
        loss = criterion(pred * mask, x * mask) / (mask.sum() + 1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, mask, idx in val_loader:
            x, mask = x.to(device), mask.to(device)
            recon = model(x)
            # 收集已观测位置的预测与真实值
            for i, u_idx in enumerate(idx):
                obs = mask[i].nonzero(as_tuple=False).squeeze()
                ys.extend(x[i, obs].cpu().tolist())
                ps.extend(recon[i, obs].cpu().tolist())
    rmse = mean_squared_error(ys, ps, squared=False)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse, trials = rmse, 0
        torch.save(model.state_dict(), 'best_attn_autorec.pth')
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

# 加载最佳模型
state = torch.load('best_attn_autorec.pth', map_location=device)
model.load_state_dict(state)
model.eval()

# 重建全矩阵并预测
full_preds = np.zeros((n_users, n_items), dtype=np.float32)
for x, mask, idx in DataLoader(dataset, batch_size=256):
    out = model(x.to(device)).cpu().detach().numpy()
    for j, u in enumerate(idx):
        full_preds[u] = out[j]

preds = []
mean_score = raw_df['score'].mean()
for _, row in test_df.iterrows():
    u, i = row['u_idx'], row['i_idx']
    if u >= 0 and i >= 0:
        preds.append(full_preds[int(u), int(i)])
    else:
        preds.append(mean_score)

test_df['score'] = preds
test_df.to_csv('test_attn_autorec_preds.csv', index=False)


with open('test_attn_autorec_preds.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            f.write(f"{row['item']} {row['score']}\n")