import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#######################################
# 数据生成：模拟多个簇
#######################################
def generate_data(n_samples=400, centers=4, spread=1.0, random_state=42):
    np.random.seed(random_state)
    X = []
    for _ in range(centers):
        center = np.random.uniform(-10, 10, size=2)
        cluster = center + spread * np.random.randn(n_samples // centers, 2)
        X.append(cluster)
    return np.vstack(X)

X = generate_data(n_samples=4000, centers=7, spread=0.5)

#######################################
# BFR 算法实现（适用于大规模数据）
#######################################
# BFR 中：用 DS 表示已分配的数据（使用充分统计量：N, SUM, SUMSQ）
# 利用 KMeans 库对初始块数据聚类，然后对后续块数据用 Mahalanobis 距离归类

def compute_DS_summary(points):
    N = len(points)
    SUM = np.sum(points, axis=0)
    SUMSQ = np.sum(points**2, axis=0)
    return {'N': N, 'SUM': SUM, 'SUMSQ': SUMSQ}

def update_DS(summary, point):
    summary['N'] += 1
    summary['SUM'] += point
    summary['SUMSQ'] += point**2

def mahalanobis_distance(point, summary):
    N = summary['N']
    centroid = summary['SUM'] / N
    variance = summary['SUMSQ'] / N - centroid**2
    std = np.sqrt(np.where(variance > 0, variance, 1e-6))
    return np.linalg.norm((point - centroid) / std)

def bfr_clustering(X, k, threshold=2.0, chunk_size=50):
    # 随机打乱数据，并分成若干块（模拟大规模数据或流式数据）
    idx = np.random.permutation(len(X))
    X_shuffled = X[idx]
    chunks = [X_shuffled[i:i+chunk_size] for i in range(0, len(X), chunk_size)]
    
    # 利用 KMeans 库对第一块数据初始化 DS
    initial_chunk = chunks[0]
    kmeans_init = KMeans(n_clusters=k, random_state=42).fit(initial_chunk)
    init_labels = kmeans_init.labels_
    DS_summaries = []
    DS_points = {}  # 用于存储每个 DS 簇的点，便于后续可视化
    for j in range(k):
        pts = initial_chunk[init_labels == j]
        summary = compute_DS_summary(pts)
        DS_summaries.append(summary)
        DS_points[j] = pts.copy()
    
    RS_points = []  # RS：未归类的孤立点
    
    # 依次处理后续数据块
    for chunk in chunks[1:]:
        for point in chunk:
            distances = [mahalanobis_distance(point, s) for s in DS_summaries]
            min_distance = min(distances)
            cluster_idx = distances.index(min_distance)
            if min_distance < threshold:
                # 如果距离小于阈值，则更新 DS 统计量
                update_DS(DS_summaries[cluster_idx], point)
                DS_points[cluster_idx] = np.vstack([DS_points[cluster_idx], point])
            else:
                RS_points.append(point)
    
    RS_points = np.array(RS_points) if len(RS_points) > 0 else np.empty((0, X.shape[1]))
    return DS_points, RS_points, DS_summaries

# 运行 BFR 算法
DS_points, RS_points, DS_summaries = bfr_clustering(X, k=4, threshold=2.0, chunk_size=50)

# 可视化 BFR 结果：DS 簇用不同颜色，RS 用黑色标记
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'purple']
for j in DS_points:
    plt.scatter(DS_points[j][:,0], DS_points[j][:,1], color=colors[j % len(colors)], label=f'DS Cluster {j}')
if RS_points.shape[0] > 0:
    plt.scatter(RS_points[:,0], RS_points[:,1], color='black', marker='x', label='RS (Outliers)')
plt.title('BFR Clustering Result')
plt.legend()
plt.show()

#######################################
# CURE 算法实现（适用于非球形簇）
#######################################
# CURE 算法思路：
# 1. 对大规模数据可进行抽样，利用 KMeans 库对抽样数据做初步聚类
# 2. 对每个簇选取 r 个代表点，并向簇中心收缩 alpha 比例
# 3. 对所有数据，根据代表点距离重新分配簇

def cure_clustering(X, k, r=5, alpha=0.2, sample_size=None):
    # 如有需要，对数据抽样
    if sample_size is not None and sample_size < len(X):
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    # 利用 KMeans 库对抽样数据进行初步聚类
    kmeans_sample = KMeans(n_clusters=k, random_state=42).fit(X_sample)
    labels_sample = kmeans_sample.labels_
    centroids = kmeans_sample.cluster_centers_
    
    # 为每个簇选取 r 个代表点，并进行向中心收缩
    rep_points = {}
    for j in range(k):
        cluster_points = X_sample[labels_sample == j]
        if len(cluster_points) == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        # 选取距离中心较远的点作为第一个代表点
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        first_idx = np.argmax(distances)
        reps = [cluster_points[first_idx]]
        # 贪心地依次选择距离当前已选代表点最远的点
        while len(reps) < r and len(cluster_points) > len(reps):
            dists = np.array([min(np.linalg.norm(p - rep) for rep in reps) for p in cluster_points])
            next_idx = np.argmax(dists)
            reps.append(cluster_points[next_idx])
        reps = np.array(reps)
        # 向中心收缩：使代表点更接近簇中心，减少噪声影响
        reps_shrunk = reps + alpha * (centroid - reps)
        rep_points[j] = reps_shrunk
    
    # 对所有数据，根据各簇代表点的最小距离进行分配
    final_labels = np.zeros(len(X), dtype=int)
    for i, point in enumerate(X):
        distances = []
        for j in range(k):
            if j in rep_points:
                d = np.min(np.linalg.norm(rep_points[j] - point, axis=1))
            else:
                d = np.inf
            distances.append(d)
        final_labels[i] = np.argmin(distances)
    return final_labels, rep_points

# 运行 CURE 算法
cure_labels, cure_rep_points = cure_clustering(X, k=4, r=5, alpha=0.2, sample_size=300)

# 可视化 CURE 结果：用不同颜色显示不同簇，同时标记代表点
plt.figure(figsize=(8,6))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=cure_labels, palette='deep', legend='full')
for j in cure_rep_points:
    plt.scatter(cure_rep_points[j][:,0], cure_rep_points[j][:,1],
                color='black', marker='X', s=100, label=f'Rep Points {j}')
plt.title('CURE Clustering Result')
plt.legend()
plt.show()
