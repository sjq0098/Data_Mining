def pagerank(graph, d=0.85, max_iter=100, tol=1e-6):
    N = len(graph)
    # 初始化每个节点的 PageRank 值
    pr = {node: 1.0 / N for node in graph}
    
    for iteration in range(max_iter):
        # 预先初始化 pr_new 字典，对每个节点都设置基本的随机跳转部分
        pr_new = {node: (1-d) / N for node in graph}
        
        # 遍历所有节点及其出链
        for src, out_nodes in graph.items():
            if out_nodes:
                share = pr[src] / len(out_nodes)
                for dest in out_nodes:
                    pr_new[dest] += d * share
            else:
                # 如果 src 没有出链，将其 PageRank 均分给所有节点（dangling node处理）
                for dest in graph:
                    pr_new[dest] += d * pr[src] / N
        
        # 判断收敛性
        diff = sum(abs(pr_new[node] - pr[node]) for node in graph)
        pr = pr_new
        if diff < tol:
            print(f"Converged after {iteration+1} iterations.")
            break
    return pr

# 测试图数据

# 图1：原先的简单示例图
graph1 = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A'],
    'D': ['C']
}

# 图2：环形和分支混合图
graph2 = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['A', 'D', 'E'],
    'D': ['E'],
    'E': ['F'],
    'F': ['D']
}

# 图3：含孤立节点和无出链节点的图
graph3 = {
    'A': ['B'],
    'B': ['C'],
    'C': [],       # C 无出链
    'D': ['C', 'A'],
    'E': ['D'],
    'F': []        # F 为孤立出链节点
}

# 图4：较大规模图（示例）
graph4 = {
    'A': ['B', 'C', 'D'],
    'B': ['C', 'E'],
    'C': ['D', 'F'],
    'D': ['C', 'G'],
    'E': ['F', 'H'],
    'F': ['D', 'I'],
    'G': ['J'],
    'H': ['I'],
    'I': ['G'],
    'J': ['A', 'I']
}

# 定义一个字典方便批量测试
graphs = {
    "Graph1": graph1,
    "Graph2": graph2,
    "Graph3": graph3,
    "Graph4": graph4
}

for name, g in graphs.items():
    print("___", name, "___")
    print("Graph structure:", g)
    pr = pagerank(g)
    print("PageRank result (sorted):")
    sorted_pr = sorted(pr.items(), key=lambda item: item[1], reverse=True)
    for node, score in sorted_pr:
        print(node, score)

