import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

#生成数据
def generate_data(n_samples=200, centers=4, spread=1.0, random_state=3307, 
                 noise_level=0.1, cluster_sizes=None, distribution='gaussian',
                 cluster_shapes='circular', rotation_angles=None):
    """
    生成用于聚类分析的数据集
    
    参数:
    n_samples: 总样本数
    centers: 簇的数量
    spread: 簇的扩散程度
    random_state: 随机种子
    noise_level: 噪声水平（0-1之间）
    cluster_sizes: 每个簇的样本数量列表，如果为None则均匀分布
    distribution: 数据分布类型 ('gaussian', 'uniform', 'exponential')
    cluster_shapes: 簇的形状 ('circular', 'elliptical', 'line')
    rotation_angles: 每个簇的旋转角度列表
    """
    np.random.seed(random_state)
    X = []
    
    # 如果没有指定簇大小，则均匀分布
    if cluster_sizes is None:
        cluster_sizes = [n_samples // centers] * centers
        # 处理不能整除的情况
        remaining = n_samples % centers
        for i in range(remaining):
            cluster_sizes[i] += 1
    
    # 如果没有指定旋转角度，则随机生成
    if rotation_angles is None:
        rotation_angles = np.random.uniform(0, 2*np.pi, centers)
    
    for i in range(centers):
        # 生成簇中心
        center = np.random.uniform(-10, 10, size=2)
        
        # 根据分布类型生成数据
        if distribution == 'gaussian':
            points = np.random.randn(cluster_sizes[i], 2)
        elif distribution == 'uniform':
            points = np.random.uniform(-1, 1, (cluster_sizes[i], 2))
        elif distribution == 'exponential':
            points = np.random.exponential(1, (cluster_sizes[i], 2))
        
        # 根据簇形状调整数据
        if cluster_shapes == 'circular':
            # 将点归一化到单位圆
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        elif cluster_shapes == 'elliptical':
            # 创建椭圆形分布
            points[:, 0] *= 2
        elif cluster_shapes == 'line':
            # 创建线性分布
            points[:, 1] = 0.1 * points[:, 0] + np.random.randn(cluster_sizes[i]) * 0.1
        
        # 应用旋转
        rotation_matrix = np.array([
            [np.cos(rotation_angles[i]), -np.sin(rotation_angles[i])],
            [np.sin(rotation_angles[i]), np.cos(rotation_angles[i])]
        ])
        points = points @ rotation_matrix.T
        
        # 添加噪声
        noise = np.random.randn(cluster_sizes[i], 2) * noise_level
        
        # 将点移动到指定中心并应用扩散
        cluster = center + spread * points + noise
        X.append(cluster)
    
    return np.vstack(X)

#手搓Kmeans
def kmeans(X,k,max_items=100,tol=1e-4):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_items):
        distances=cdist(X,centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros((k, X.shape[1]))  # 预先创建一个新的质心数组
        counts = np.zeros(k)  # 记录每个簇的样本数量

        # 遍历所有数据点，计算每个簇的均值
        for i in range(len(X)):
            cluster_idx = labels[i]  # 当前点所属的簇
            new_centroids[cluster_idx] += X[i]  # 逐步累加坐标
            counts[cluster_idx] += 1  # 该簇点数加1

        # 计算均值，避免除零错误
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]  # 计算均值

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels,centroids

#层次聚类
def hierarchical_clustering(X, k):
    clusters = {i: [i] for i in range(len(X))}
    distances = cdist(X, X)
    np.fill_diagonal(distances,np.inf)

    while len(clusters)>k:
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        clusters[i].extend(clusters[j])
        del clusters[j]
        for idx in clusters:
            if idx != i:
                distances[i, idx] = distances[idx, i] = np.min(cdist(X[clusters[i]], X[clusters[idx]]))
        distances[j, :] = distances[:, j] = np.inf
    
    labels = np.zeros(len(X))
    for cluster_id, points in enumerate(clusters.values()):
        labels[points] = cluster_id

    return labels

# 生成不同大小的簇
X1 = generate_data(n_samples=300, centers=3, cluster_sizes=[100, 150, 50])

# 生成带有噪声的椭圆形簇
X2 = generate_data(n_samples=200, centers=4, noise_level=0.2, cluster_shapes='elliptical')

# 生成线性簇
X3 = generate_data(n_samples=200, centers=3, cluster_shapes='line', distribution='uniform')

kmeans_labels, kmeans_centroids = kmeans(X1, 4)
hierarchical_labels = hierarchical_clustering(X1, 4)

# 树状图
distance_matrix = linkage(X1, method='ward')

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means 可视化
sns.scatterplot(x=X1[:, 0], y=X1[:, 1], hue=kmeans_labels, palette='viridis', ax=axes[0])
axes[0].scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='x', s=100)
axes[0].set_title('K-Means Clustering')

# 层次聚类可视化
sns.scatterplot(x=X1[:, 0], y=X1[:, 1], hue=hierarchical_labels, palette='coolwarm', ax=axes[1])
axes[1].set_title('Hierarchical Clustering')

# 树状图
dendrogram(distance_matrix, truncate_mode='level', p=5, ax=axes[2])
axes[2].set_title('Hierarchical Dendrogram')

plt.show()

