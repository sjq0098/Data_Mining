import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def generate_complex_graph(
    n_nodes: int = 20,
    n_communities: int = 3,
    intra_community_prob: float = 0.7,
    inter_community_prob: float = 0.1,
    random_state: Optional[int] = None
) -> nx.DiGraph:
    """
    生成一个具有社区结构的复杂有向图
    
    参数:
    n_nodes: 节点总数
    n_communities: 社区数量
    intra_community_prob: 社区内部边的生成概率
    inter_community_prob: 社区之间边的生成概率
    random_state: 随机种子
    
    返回:
    nx.DiGraph: 生成的有向图
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 创建空图
    G = nx.DiGraph()
    
    # 为每个节点分配社区
    nodes = list(range(n_nodes))
    community_size = n_nodes // n_communities
    communities = []
    
    for i in range(n_communities):
        start_idx = i * community_size
        end_idx = start_idx + community_size if i < n_communities - 1 else n_nodes
        communities.append(nodes[start_idx:end_idx])
    
    # 添加节点
    G.add_nodes_from(nodes)
    
    # 生成社区内部的边
    for community in communities:
        for i in community:
            for j in community:
                if i != j and np.random.random() < intra_community_prob:
                    # 随机决定边的方向
                    if np.random.random() < 0.5:
                        G.add_edge(i, j)
                    else:
                        G.add_edge(j, i)
    
    # 生成社区之间的边
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i != j:
                for node1 in comm1:
                    for node2 in comm2:
                        if np.random.random() < inter_community_prob:
                            # 随机决定边的方向
                            if np.random.random() < 0.5:
                                G.add_edge(node1, node2)
                            else:
                                G.add_edge(node2, node1)
    
    return G

def generate_random_graph(
    n_nodes: int = 20,
    edge_prob: float = 0.3,
    random_state: Optional[int] = None
) -> nx.DiGraph:
    """
    生成一个随机有向图
    
    参数:
    n_nodes: 节点数量
    edge_prob: 边的生成概率
    random_state: 随机种子
    
    返回:
    nx.DiGraph: 生成的有向图
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and np.random.random() < edge_prob:
                G.add_edge(i, j)
    
    return G

def hits_algorithm(G, max_iter=100, tolerance=1e-6):
    """
    实现HITS算法
    返回: hub_scores, authority_scores
    """
    # 初始化hub和authority分数
    nodes = list(G.nodes())
    n = len(nodes)
    hub_scores = np.ones(n)
    authority_scores = np.ones(n)
    
    # 创建邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).toarray()
    
    for _ in range(max_iter):
        # 保存旧的分数用于收敛检查
        old_hub_scores = hub_scores.copy()
        old_authority_scores = authority_scores.copy()
        
        # 更新authority分数
        authority_scores = adj_matrix.T @ hub_scores
        authority_scores = authority_scores / np.linalg.norm(authority_scores)
        
        # 更新hub分数
        hub_scores = adj_matrix @ authority_scores
        hub_scores = hub_scores / np.linalg.norm(hub_scores)
        
        # 检查收敛性
        hub_diff = np.linalg.norm(hub_scores - old_hub_scores)
        auth_diff = np.linalg.norm(authority_scores - old_authority_scores)
        
        if hub_diff < tolerance and auth_diff < tolerance:
            break
    
    # 将分数转换为字典
    hub_dict = dict(zip(nodes, hub_scores))
    auth_dict = dict(zip(nodes, authority_scores))
    
    return hub_dict, auth_dict

def visualize_graph(G, hub_scores, auth_scores):
    """可视化图及其HITS分数"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_size=1000,
                                 node_color=list(hub_scores.values()),
                                 cmap='YlOrRd')
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # 添加标签
    labels = {node: f'{node}\nHub: {hub_scores[node]:.3f}\nAuth: {auth_scores[node]:.3f}'
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('HITS Algorithm Results')
    plt.colorbar(nodes, label='Hub Score')
    plt.axis('off')
    plt.savefig('hits_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 生成两种不同类型的测试图
    print("生成具有社区结构的图...")
    G1 = generate_complex_graph(n_nodes=20, n_communities=3, random_state=42)
    print(f"节点数: {G1.number_of_nodes()}, 边数: {G1.number_of_edges()}")
    
    print("\n生成随机图...")
    G2 = generate_random_graph(n_nodes=20, edge_prob=0.3, random_state=42)
    print(f"节点数: {G2.number_of_nodes()}, 边数: {G2.number_of_edges()}")
    
    # 对两个图分别运行HITS算法
    for i, G in enumerate([G1, G2], 1):
        print(f"\n图 {i} 的HITS算法结果:")
        print("节点\tHub分数\tAuthority分数")
        print("-" * 40)
        
        hub_scores, auth_scores = hits_algorithm(G)
        
        for node in G.nodes():
            print(f"{node}\t{hub_scores[node]:.4f}\t{auth_scores[node]:.4f}")
        
        # 可视化结果
        visualize_graph(G, hub_scores, auth_scores)
        print(f"\n图 {i} 的可视化结果已保存为 'hits_visualization_{i}.png'")

if __name__ == "__main__":
    main() 