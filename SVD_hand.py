import random
import math

# 1. 工具函数
def mat_mul(A, B):
    """矩阵乘法，A: m×n, B: n×p -> 返回 m×p"""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2
    C = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for t in range(n):
                C[i][j] += A[i][t] * B[t][j]
    return C

def transpose(A):
    """矩阵转置"""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def norm(v):
    """向量二范数"""
    return math.sqrt(sum(x*x for x in v))

def scalar_vec_mul(a, v):
    return [a*x for x in v]

def vec_sub(u, v):
    return [u[i]-v[i] for i in range(len(u))]

def outer(u, v):
    """u: m, v: n -> m×n"""
    return [[u_i * v_j for v_j in v] for u_i in u]

# 2. 幂迭代求一对奇异值/向量
def power_iteration(A, num_iters=50):
    """返回 (sigma, u, v)"""
    m, n = len(A), len(A[0])
    # 初始化 v
    v = [random.random() for _ in range(n)]
    # 归一化
    nv = norm(v)
    v = [x/nv for x in v]
    AT = transpose(A)
    for _ in range(num_iters):
        # u = A v
        u = [sum(A[i][j]*v[j] for j in range(n)) for i in range(m)]
        nu = norm(u)
        u = [x/nu for x in u]
        # v = A^T u
        v = [sum(AT[j][i]*u[i] for i in range(m)) for j in range(n)]
        nv = norm(v)
        v = [x/nv for x in v]
    sigma = sum(A[i][j]*v[j] for i in range(m) for j in range(n) if False)  # 占位
    # 更准确的 sigma = ||A v|| = norm(A v)
    Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(m)]
    sigma = norm(Av)
    return sigma, u, v

# 3. 挖去后迭代提取 k 对
def svd_manual(A, k=2):
    A_work = [row[:] for row in A]  # 深拷贝
    m, n = len(A), len(A[0])
    sigmas, Us, Vs = [], [], []
    for _ in range(k):
        sigma, u, v = power_iteration(A_work)
        sigmas.append(sigma)
        Us.append(u)
        Vs.append(v)
        # 挖去： A_work = A_work - sigma * u v^T
        UVT = outer(u, v)
        for i in range(m):
            for j in range(n):
                A_work[i][j] -= sigma * UVT[i][j]
    return sigmas, Us, Vs

# 4. 测试并重构
R = [
    [5,4,0,1,0,0],
    [4,0,0,1,1,0],
    [1,1,0,5,4,0],
    [0,0,5,4,0,1],
    [0,0,4,0,0,5],
]
k = 2
sigmas, Us, Vs = svd_manual(R, k)

# 构造 U_k Σ_k V_k^T
m, n = len(R), len(R[0])
# 重构矩阵 R_hat
R_hat = [[0.0]*n for _ in range(m)]
for idx in range(k):
    σ = sigmas[idx]
    u = Us[idx]
    v = Vs[idx]
    for i in range(m):
        for j in range(n):
            R_hat[i][j] += σ * u[i] * v[j]

# 打印结果
print("奇异值：", [round(s,2) for s in sigmas])
print("重构 R_hat：")
for row in R_hat:
    print([round(x,2) for x in row])

# 推荐示例：为用户 0 推荐
user0 = R_hat[0]
# 原始未评分位置
zeros = [j for j,val in enumerate(R[0]) if val==0]
# 按预测评分降序
zeros_sorted = sorted(zeros, key=lambda j: user0[j], reverse=True)
print("推荐电影索引：", zeros_sorted[:2])
