#!/usr/bin/env python3
import math
import os

# 参数设置
beta = 0.85         # 随机游走时跟随超链接的概率，其余概率均匀跳转
epsilon = 1e-6      # 收敛阈值
max_iter = 100      # 最大迭代次数
block_size = 1000   # 条带宽度，即每个条带覆盖的节点数；根据内存情况调整

outdeg = {}         # 记录每个节点的出度
nodes = set()       # 存储所有出现过的节点

with open("Data.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u = int(parts[0])
        v = int(parts[1])
        nodes.add(u)
        nodes.add(v)
        outdeg[u] = outdeg.get(u, 0) + 1

# 确定节点总数 N
N = max(nodes) + 1  # 假定节点编号从 0 开始且连续
print('Total nodes N:', N)


def create_stripe_files(data_file, N, block_size):
    num_stripes = math.ceil(N / block_size)
    # 打开 num_stripes 个输出文件写入条带数据
    stripe_files = [open(f"stripe_{i}.txt", "w") for i in range(num_stripes)]
    
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            stripe_id = v // block_size
            stripe_files[stripe_id].write(f"{u} {v}\n")
    
    for sf in stripe_files:
        sf.close()
    print(f"预处理完成，生成 {num_stripes} 个条带文件.")

create_stripe_files("Data.txt", N, block_size)


r = [1.0 / N for _ in range(N)]

def stripe_based_pagerank(r, outdeg, N, block_size, beta, epsilon, max_iter):
    num_stripes = math.ceil(N / block_size)
    for iter_num in range(max_iter):
        # 初始化 r_new: 每个节点先赋值基础分 (1-beta)/N
        r_new = [(1 - beta) / N for _ in range(N)]
        
        # 针对每个条带文件更新对应区间内节点的 PageRank
        for stripe_id in range(num_stripes):
            block_start = stripe_id * block_size
            block_end = min((stripe_id + 1) * block_size, N)
            stripe_file = f"stripe_{stripe_id}.txt"
            if not os.path.exists(stripe_file):
                continue
            with open(stripe_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    u = int(parts[0])
                    v = int(parts[1])
                    # 条带预处理保证 v 落在当前条带对应区间内，此处可直接更新
                    if u in outdeg and outdeg[u] > 0:
                        r_new[v] += beta * (r[u] / outdeg[u])
            print(f"条带 {stripe_id} 更新完成")
        
        # 处理死节点：没有出链的节点贡献“流失”部分累加到所有节点上
        leaked = 0.0
        for i in range(N):
            if i not in outdeg or outdeg[i] == 0:
                leaked += beta * r[i]
        leaked_share = leaked / N
        r_new = [val + leaked_share for val in r_new]
        
        # 计算更新前后 PageRank 的 L1 差异，判断是否收敛
        diff = sum(abs(r_new[i] - r[i]) for i in range(N))
        print(f"Iteration {iter_num}: diff = {diff}")
        r = r_new
        if diff < epsilon:
            break

    return r

r_final = stripe_based_pagerank(r, outdeg, N, block_size, beta, epsilon, max_iter)


nodes_rank = [(i, r_final[i]) for i in range(N)]
nodes_rank.sort(key=lambda x: x[1], reverse=True)
top100 = nodes_rank[:100]

with open("res_stripe.txt", "w") as out:
    for node, score in top100:
        out.write(f"{node}\t{score}\n")
print("条带更新：Top 100 PageRank 节点已输出到 Res.txt")
