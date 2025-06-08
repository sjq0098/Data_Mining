import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error

# —— 读取原始 train.txt —— 
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

#读取测试集
def load_test(path):
    user_ids, item_ids = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                uid, n = line.split('|')
                uid = int(uid)
            else:
                iid = int(line)
                user_ids.append(uid)
                item_ids.append(iid)
    return pd.DataFrame({'user': user_ids, 'item': item_ids})



raw_df = load_train('data/train.txt')

# —— 按 8:2 拆分 ——  
train_df, val_df = train_test_split(
    raw_df,
    test_size=0.2,
    random_state=42,
    stratify=raw_df['user']  # 保证每个用户在两部分都有样本
)

print(f"train_df: {len(train_df)}, val_df: {len(val_df)}")
num_users = train_df['user'].nunique()
print("训练集用户数 =", num_users)
print("验证集用户数=", val_df['user'].nunique())



test_df = load_test('data/test.txt')
print(f"test_df: {len(test_df)}")
print("测试集用户数=", test_df['user'].nunique())

#构造评分矩阵
# 将 train 转为稀疏矩阵
train_matrix = train_df.pivot(index='user', columns='item', values='score').fillna(0)

#统计特征计算
user_mean = train_matrix.replace(0, pd.NA).mean(axis=1)
norm_matrix = train_matrix.sub(user_mean, axis=0).fillna(0)

#计算余弦相似度
user_sim = cosine_similarity(norm_matrix)
user_sim_df = pd.DataFrame(
    user_sim,
    index=norm_matrix.index,
    columns=norm_matrix.index
)

#预测验证集并计算 RMSE
def predict_user(u, i, k):
    # 若用户或物品不在训练集中：返回全局均值或用户均值
    if u not in user_sim_df.index or i not in train_matrix.columns:
        return user_mean.get(u, user_mean.mean())

    sims = user_sim_df.loc[u].drop(u)
    top_k = sims.nlargest(k)
    # 只保留这些用户对 i 有过评分的
    rated = train_matrix.loc[top_k.index, i] > 0
    top_k = top_k[rated]
    if top_k.abs().sum() == 0:
        return user_mean[u]
    numer = ((train_matrix.loc[top_k.index, i] - user_mean[top_k.index]) * top_k).sum()
    denom = top_k.abs().sum()
    return user_mean[u] + numer / denom

def evaluate(k):
    preds = []
    for _, row in val_df.iterrows():
        preds.append(predict_user(row['user'], row['item'], k))
    return np.sqrt(mean_squared_error(val_df['score'], preds))

# 在一组 K 值上搜索最优
candidates = [5, 10, 20, 40, 80]
results = {k: evaluate(k) for k in candidates}
best_k = min(results, key=results.get)
print("各 K 对应 RMSE:", results)
print("最佳 K =", best_k, "验证集 RMSE =", results[best_k])


test_df['pred'] = test_df.apply(
    lambda r: predict_user(r['user'], r['item'], best_k),
    axis=1
)
# 保存结果
test_df.to_csv('test_predictions.csv', index=False)
# 保存结果，按照指定格式写入到 test_predictions.txt
with open('test_predictions.txt', 'w', encoding='utf-8') as f:
    for user, group in test_df.groupby('user'):
        # 写入用户行，格式为 "<user>|<该用户预测评分的物品数量>"
        f.write(f"{user}|{len(group)}\n")
        for _, row in group.iterrows():
            # 写入物品及预测评分行，这里评分取整，你也可以保持浮点数格式
            f.write(f"{row['item']}  {round(row['pred'])}\n")




