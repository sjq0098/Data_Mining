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

# 3. 划分 train/val
train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
print(f"train_df: {len(train_df)}, val_df: {len(val_df)}")
num_users = train_df['user'].nunique()
print("训练集用户数 =", num_users)
print("验证集用户数=", val_df['user'].nunique())

def df_to_matrix(df):
    # 转为列表 (u_idx, i_idx, score)
    return [(user2idx[r['user']], item2idx[r['item']], r['score']) for _, r in df.iterrows()]
train_data = df_to_matrix(train_df)
val_data   = df_to_matrix(val_df)

# 4. 初始化参数
K = 20               # 隐因子数
epochs = 20
lr = 0.001
reg = 0.02

# 全局均值
mu = train_df['score'].mean()
# 偏置项
b_u = np.zeros(num_users, dtype=np.float32)
b_i = np.zeros(num_items, dtype=np.float32)
# 隐向量矩阵
P = np.random.normal(scale=0.1, size=(num_users, K)).astype(np.float32)
Q = np.random.normal(scale=0.1, size=(num_items, K)).astype(np.float32)

# 5. 训练 loop
for epoch in range(epochs):
    np.random.shuffle(train_data)
    for u, i, r_ui in train_data:
        # 预测
        pred = mu + b_u[u] + b_i[i] + P[u].dot(Q[i])
        err = r_ui - pred
        # 梯度更新
        b_u[u] += lr * (err - reg * b_u[u])
        b_i[i] += lr * (err - reg * b_i[i])
        P[u]   += lr * (err * Q[i] - reg * P[u])
        Q[i]   += lr * (err * P[u] - reg * Q[i])
    # 每轮打印验证集 RMSE
    preds = []
    ys = []
    for u, i, r in val_data:
        y_hat = mu + b_u[u] + b_i[i] + P[u].dot(Q[i])
        preds.append(y_hat)
        ys.append(r)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    print(f"Epoch {epoch+1}/{epochs}, Validation RMSE: {rmse:.4f}")

# 6. 预测测试集
# 读取 test.txt 并处理为 (user_idx, item_idx)
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

# 进行索引映射
results = []
for _, row in test_df.iterrows():
    u, i = row['user'], row['item']
    if u in user2idx and i in item2idx:
        u_idx, i_idx = user2idx[u], item2idx[i]
        pred = mu + b_u[u_idx] + b_i[i_idx] + P[u_idx].dot(Q[i_idx])
    elif u in user2idx:
        u_idx = user2idx[u]
        pred = mu + b_u[u_idx]
    elif i in item2idx:
        i_idx = item2idx[i]
        pred = mu + b_i[i_idx]
    else:
        pred = mu
    results.append(pred)

# 保存预测结果
test_df['score'] = results
test_df.to_csv('test_predictions_mf.csv', index=False)  # 保存为 CSV

with open('test_predictions_mf.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        # 写入用户行，格式为 "<user>|<该用户预测评分的物品数量>"
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            # 写入物品及预测评分行保持浮点数格式
            f.write(f"{row['item']}  {row['score']}\n")
