import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 数据加载（同前）
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

# 2. 构建ID映射
raw_df = load_train('data/train.txt')
user_ids = raw_df['user'].unique()
item_ids = raw_df['item'].unique()
user2idx = {u:i for i,u in enumerate(user_ids)}
item2idx = {i:j for j,i in enumerate(item_ids)}
num_users, num_items = len(user_ids), len(item_ids)

# 3. Z-score 标准化
# 对 score 列进行标准化，记录均值和方差，用于反归一化
score_mean = raw_df['score'].mean()
score_std = raw_df['score'].std()
raw_df['score_z'] = (raw_df['score'] - score_mean) / score_std

# 4. 划分 train/val
train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)

def df_to_matrix(df, score_col):
    # 转为列表 (u_idx, i_idx, score)
    return [(user2idx[r['user']], item2idx[r['item']], r[score_col]) for _, r in df.iterrows()]
train_data = df_to_matrix(train_df, 'score_z')
val_data   = df_to_matrix(val_df, 'score_z')

# 5. 初始化参数
K = 20               # 隐因子数
epochs = 20
lr = 0.005
reg = 0.02

# 偏置项和隐向量初始化
b_u = np.zeros(num_users, dtype=np.float32)
b_i = np.zeros(num_items, dtype=np.float32)
P = np.random.normal(scale=0.1, size=(num_users, K)).astype(np.float32)
Q = np.random.normal(scale=0.1, size=(num_items, K)).astype(np.float32)

# 6. 训练 loop（基于 z-score）
for epoch in range(epochs):
    np.random.shuffle(train_data)
    for u, i, r_ui in train_data:
        pred = b_u[u] + b_i[i] + P[u].dot(Q[i])
        err = r_ui - pred
        # 梯度更新
        b_u[u] += lr * (err - reg * b_u[u])
        b_i[i] += lr * (err - reg * b_i[i])
        P[u]   += lr * (err * Q[i] - reg * P[u])
        Q[i]   += lr * (err * P[u] - reg * Q[i])
    # 验证集 RMSE（反归一化后计算）
    preds, ys = [], []
    for u, i, r_z in val_data:
        z_hat = b_u[u] + b_i[i] + P[u].dot(Q[i])
        # 反归一化
        y_hat = z_hat * score_std + score_mean
        y_true = r_z * score_std + score_mean
        preds.append(y_hat)
        ys.append(y_true)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    print(f"Epoch {epoch+1}/{epochs}, Validation RMSE: {rmse:.4f}")

# 7. 预测测试集并保存
# 加载测试集

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
results = []
for _, row in test_df.iterrows():
    u, i = row['user'], row['item']
    if u in user2idx and i in item2idx:
        u_idx, i_idx = user2idx[u], item2idx[i]
        z_hat = b_u[u_idx] + b_i[i_idx] + P[u_idx].dot(Q[i_idx])
    elif u in user2idx:
        z_hat = b_u[user2idx[u]]
    elif i in item2idx:
        z_hat = b_i[item2idx[i]]
    else:
        z_hat = 0.0
    # 反归一化
    results.append(z_hat * score_std + score_mean)

# 保存为 CSV
test_df['score'] = results
test_df.to_csv('test_predictions_mfz.csv', index=False)  # 保存为 CSV

with open('test_predictions_mfz.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        # 写入用户行，格式为 "<user>|<该用户预测评分的物品数量>"
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            # 写入物品及预测评分行保持浮点数格式
            f.write(f"{row['item']}  {row['score']}\n")
