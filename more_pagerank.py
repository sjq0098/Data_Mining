# 定义之前的算法实现

def pagerank(graph, d=0.85, max_iter=100, tol=1e-6):
    N = len(graph)
    pr = {node: 1.0 / N for node in graph}
    for iteration in range(max_iter):
        pr_new = {node: (1-d)/N for node in graph}
        for node, out_nodes in graph.items():
            if out_nodes:
                share = pr[node] / len(out_nodes)
                for dest in out_nodes:
                    pr_new[dest] += d * share
            else:
                for dest in graph:
                    pr_new[dest] += d * pr[node] / N
        diff = sum(abs(pr_new[node] - pr[node]) for node in graph)
        pr = pr_new
        if diff < tol:
            print(f"PageRank converged after {iteration+1} iterations.")
            break
    return pr

def topic_sensitive_pagerank(graph, v, d=0.85, max_iter=100, tol=1e-6):
    N = len(graph)
    pr = {node: 1.0 / N for node in graph}
    for iteration in range(max_iter):
        pr_new = {node: (1-d) * v.get(node, 0) for node in graph}
        for node, out_nodes in graph.items():
            if out_nodes:
                share = pr[node] / len(out_nodes)
                for dest in out_nodes:
                    pr_new[dest] += d * share
            else:
                for dest in graph:
                    pr_new[dest] += d * pr[node] / N
        diff = sum(abs(pr_new[node] - pr[node]) for node in graph)
        pr = pr_new
        if diff < tol:
            print(f"Topic-sensitive PageRank converged after {iteration+1} iterations.")
            break
    return pr

def trustrank(graph, trusted_nodes, d=0.85, max_iter=100, tol=1e-6):
    N = len(graph)
    # 构造个性化向量：可信节点分配较高权重，其余节点分配较低权重
    v = {}
    trusted_prob = 0.9
    non_trusted_prob = 0.1
    num_trusted = len(trusted_nodes)
    num_non_trusted = N - num_trusted
    for node in graph:
        if node in trusted_nodes:
            v[node] = trusted_prob / num_trusted
        else:
            v[node] = non_trusted_prob / num_non_trusted if num_non_trusted else 0
    return topic_sensitive_pagerank(graph, v, d, max_iter, tol)

def hits(graph, max_iter=100, tol=1e-6):
    nodes = list(graph.keys())
    authority = {node: 1.0 for node in nodes}
    hub = {node: 1.0 for node in nodes}
    for iteration in range(max_iter):
        new_authority = {node: 0.0 for node in nodes}
        for node in nodes:
            # 累加所有指向 node 的 hub 值
            for src, out_nodes in graph.items():
                if node in out_nodes:
                    new_authority[node] += hub[src]
        new_hub = {node: 0.0 for node in nodes}
        for node in nodes:
            # 累加 node 指向的所有 authority 值
            for dest in graph[node]:
                new_hub[node] += new_authority[dest]
        norm_a = sum(val**2 for val in new_authority.values()) ** 0.5
        norm_h = sum(val**2 for val in new_hub.values()) ** 0.5
        for node in nodes:
            new_authority[node] = new_authority[node] / (norm_a if norm_a else 1)
            new_hub[node] = new_hub[node] / (norm_h if norm_h else 1)
        diff = sum(abs(new_authority[node]-authority[node]) for node in nodes) + \
               sum(abs(new_hub[node]-hub[node]) for node in nodes)
        authority, hub = new_authority, new_hub
        if diff < tol:
            print(f"HITS converged after {iteration+1} iterations.")
            break
    return authority, hub

# -------------------------
# 下面提供多个测试图数据

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

# 定义一个列表方便批量测试
graphs = {
    "Graph1": graph1,
    "Graph2": graph2,
    "Graph3": graph3,
    "Graph4": graph4
}

# 将结果输出到文件中
with open("pagerank_log.txt", "w", encoding="utf-8") as f:
    for name, g in graphs.items():
        f.write(f"\n------ {name} ------\n")
        f.write("Graph structure: " + str(g) + "\n")
        
        # PageRank 测试
        pr_result = pagerank(g)
        # 排序：按照 PageRank 值从高到低排序
        sorted_pr = sorted(pr_result.items(), key=lambda item: item[1], reverse=True)
        f.write("PageRank (sorted):\n")
        for node, score in sorted_pr:
            f.write(f"{node}: {score}\n")
        
        # 主题敏感 PageRank 测试：
        # 假设对 A, B, C 三个节点感兴趣，构造个性化向量
        v = {node: 0.4 if node in ['A', 'B', 'C'] else 0.05 for node in g}
        total = sum(v.values())
        v = {node: val/total for node, val in v.items()}
        tspr_result = topic_sensitive_pagerank(g, v)
        sorted_tspr = sorted(tspr_result.items(), key=lambda item: item[1], reverse=True)
        f.write("Topic-sensitive PageRank (sorted):\n")
        for node, score in sorted_tspr:
            f.write(f"{node}: {score}\n")
        
        # TrustRank 测试：
        # 假定 A 和 B 为可信种子
        tr_result = trustrank(g, trusted_nodes=['A', 'B'])
        sorted_tr = sorted(tr_result.items(), key=lambda item: item[1], reverse=True)
        f.write("TrustRank (sorted):\n")
        for node, score in sorted_tr:
            f.write(f"{node}: {score}\n")
        
        # HITS 测试
        authority, hub = hits(g)
        sorted_authority = sorted(authority.items(), key=lambda item: item[1], reverse=True)
        sorted_hub = sorted(hub.items(), key=lambda item: item[1], reverse=True)
        f.write("HITS Authority (sorted):\n")
        for node, score in sorted_authority:
            f.write(f"{node}: {score}\n")
        f.write("HITS Hub (sorted):\n")
        for node, score in sorted_hub:
            f.write(f"{node}: {score}\n")
        
        f.write("\n\n")
        
print("所有结果已写入 pagerank_log.txt 文件。")