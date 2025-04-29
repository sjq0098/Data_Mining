import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering

# 生成一个非凸形状的数据集（环形数据）
X, y_true = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

# 使用 scikit-learn 的 SpectralClustering 进行聚类
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = sc.fit_predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title('Spectral Clustering Result')
plt.show()
