import networkx as nx

def load_graph(file_path):
    """加载图数据：假设每行包含两个整数，表示有向边。"""
    # 若数据为有向边关系，建议使用 nx.DiGraph() 来构造有向图
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            G.add_edge(src, dst)
    return G

def compute_and_save_pagerank(G, output_file, top_k=100, alpha=0.85, tol=1e-6):
    """
    计算 PageRank 并将排名前 top_k 的节点及其 PageRank 值写入输出文件。
    
    参数：
      G: 网络图对象
      output_file: 输出文件路径
      top_k: 输出的 Top 节点数量
      alpha: 阻尼系数，通常为 0.85
      tol: 收敛容差
    """
    # 这里使用 nx.pagerank，适用于当前版本的 NetworkX，它内部会自动选择合适的计算方法
    pageranks = nx.pagerank(G, alpha=alpha, tol=tol)
    
    # 按 PageRank 值降序排序
    sorted_pr = sorted(pageranks.items(), key=lambda item: item[1], reverse=True)
    
    # 将结果写入输出文件
    with open(output_file, "w") as f:
        for node, score in sorted_pr[:top_k]:
            f.write(f"{node} {score:.8f}\n")
            
    print(f"已将 Top-{top_k} 结果写入 {output_file}")

def main():
    data_file = "Data.txt"
    output_file = "res_networkx.txt"
    
    # 加载图数据
    G = load_graph(data_file)
    
    # 计算 PageRank 并保存结果
    compute_and_save_pagerank(G, output_file, top_k=100)

if __name__ == "__main__":
    main()





        



