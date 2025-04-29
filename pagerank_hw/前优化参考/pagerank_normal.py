import numpy as np

def read_graph(file_path):
    edges=[]
    nodes_set=set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src, dst = int(parts[0]), int(parts[1])
            edges.append((src, dst))
            nodes_set.update([src, dst])
    return edges, nodes_set

def build_transition_matrix(edges, nodes_set):
    nodes = sorted(list(nodes_set))
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)

    M = np.zeros((N, N))
    outdeg = np.zeros(N)  # 出度
    indeg = np.zeros(N)   # 入度
    sum_outdeg = 0

    # 统计出度和入度
    for src, dst in edges:
        j = node_to_index[src]
        i = node_to_index[dst]
        outdeg[j] += 1
        indeg[i] += 1
        sum_outdeg += 1

    # 构建转移矩阵
    for src, dst in edges:
        j = node_to_index[src]
        i = node_to_index[dst]
        if outdeg[j] > 0:
            M[i, j] = 1.0 / outdeg[j]

    # 处理 dead-ends
    for j in range(N):
        if outdeg[j] == 0:
            M[:, j] = 1.0 / N

    # 统计孤立节点（既没有入度也没有出度的节点）
    isolated_nodes = sum(1 for i in range(N) if indeg[i] == 0 and outdeg[i] == 0)
    print(f"孤立节点数量: {isolated_nodes}")
    
    return M, node_to_index, nodes, sum_outdeg, isolated_nodes

def pagerank(M, damping=0.85, tol=1e-6, max_iter=100):

    N = M.shape[0]
    rank = np.ones(N) / N
    teleport = np.ones(N) / N
    
    for iteration in range(max_iter):
        new_rank = damping * (M @ rank) + (1 - damping) * teleport
        if np.linalg.norm(new_rank - rank, 1) < tol:
            print(f"迭代收敛于第 {iteration} 次")
            break
        rank = new_rank
    return rank

def main():
    file_path = 'Data.txt'
    edges, nodes_set = read_graph(file_path)
    
    M, node_to_index, nodes,sum_outdeg,unique_node = build_transition_matrix(edges, nodes_set)
    N = len(nodes)
    Average_outdegree = sum_outdeg / N if N > 0 else 0
    print(f"unique node: {unique_node}")
    print(f"Average outdegree: {Average_outdegree:.2f}")
    print(f"Total nodes: {N}, Total edges: {len(edges)}")
    
    pr = pagerank(M, damping=0.85)
    
    inv_idx = {idx: node for node, idx in node_to_index.items()}
    
    top = np.argsort(-pr)[:100]
    with open('Res_normal.txt', 'w') as f:
        for i in top:
            f.write(f"{inv_idx[i]} {pr[i]:.8f}\n")
    print("Top-100 PageRank results written to Res_normal.txt.")
if __name__ == '__main__':
    main()



