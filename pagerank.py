def pagerank(links, d=0.85, tol=1e-6, max_iter=100):
    """
    links: 字典，表示网页之间的链接关系
           键为网页标识，值为该网页指向的其他网页列表
    d: 阻尼因子，默认为 0.85
    tol: 收敛阈值
    max_iter: 最大迭代次数
    """
    N = len(links)
    # 初始化所有网页的 PageRank 值相等
    pr = {page: 1.0 / N for page in links}
    
    for _ in range(max_iter):
        pr_new = {}
        for page in links:
            # 基础得分：(1 - d) / N
            rank = (1 - d) / N
            # 累加所有链接到当前页面的贡献
            for other_page, out_links in links.items():
                if page in out_links:
                    rank += d * pr[other_page] / len(out_links)
            pr_new[page] = rank
        
        # 判断是否收敛
        diff = sum(abs(pr_new[p] - pr[p]) for p in links)
        if diff < tol:
            break
        pr = pr_new

    return pr

# 示例：构建一个简单的网页链接图
links = {
    'A': ['B', 'C'],   # 网页 A 指向 B、C
    'B': ['A'],        # 网页 B 指向 A
    'C': ['A', 'B']    # 网页 C 指向 A、B
}

# 计算 PageRank 值
result = pagerank(links)
print(result)
