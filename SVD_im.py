import numpy as np
from numpy.linalg import svd

# 1. 构造示例评分矩阵（5×6）
R = np.array([
    [5, 4, 0, 1, 0, 0],
    [4, 0, 0, 1, 1, 0],
    [1, 1, 0, 5, 4, 0],
    [0, 0, 5, 4, 0, 1],
    [0, 0, 4, 0, 0, 5],
], dtype=float)

# 2. 对 R 做 SVD
U, S, VT = svd(R, full_matrices=False)
# U: (5×5)->(5×6截断), S: (6,), VT: (6×6)

# 3. 截断到 k=2
k = 2
U_k = U[:, :k]            # (5×2)
S_k = np.diag(S[:k])      # (2×2)
V_k = VT[:k, :]           # (2×6)

# 4. 重构近似评分矩阵
R_hat = U_k @ S_k @ V_k   # (5×6)

# 5. 查看预测结果
print("原始 R：\n", R)
print("重构 R_hat：\n", np.round(R_hat, 2))

# 6. 给用户 0 推荐：选择原来为 0 的位置中预测值最高的两个电影
user_id = 0
preds = R_hat[user_id]
# 找出原始评分为 0 的索引
zeros = np.where(R[user_id] == 0)[0]
# 按预测评分排序
recommend_idx = zeros[np.argsort(preds[zeros])[::-1]]
print("推荐电影索引：", recommend_idx[:2])


from sklearn.decomposition import TruncatedSVD
svd_trunc = TruncatedSVD(n_components=2)


R_k = svd_trunc.fit_transform(R)            # 相当于 U_k @ S_k
VT_k = svd_trunc.components_                # 相当于 V_k

R_hat2 = R_k @ VT_k                         # 重构矩阵


# 5. 查看预测结果
print("原始 R：\n", R)
print("重构 R_hat2：\n", np.round(R_hat2, 2))

# 6. 给用户 0 推荐：选择原来为 0 的位置中预测值最高的两个电影
user_id = 0
preds = R_hat2[user_id]
# 找出原始评分为 0 的索引
zeros = np.where(R[user_id] == 0)[0]
# 按预测评分排序
recommend_idx = zeros[np.argsort(preds[zeros])[::-1]]
print("推荐电影索引：", recommend_idx[:2])




