#!/usr/bin/env python3
import math
import os
import heapq

# 参数设置
beta = 0.85         # 随机游走时跟随超链接的概率，其余概率均匀跳转
epsilon = 1e-6      # 收敛阈值
max_iter = 100      # 最大迭代次数
block_size = 1000   # 条带宽度，根据内存和磁盘 I/O 调整

data_file = "Data.txt"

# 1. 读取边并分区到条带文件（按目标节点 v）
outdeg = {}         # 全局出度映射：node -> count
nodes = set()
with open(data_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        u, v = map(int, parts)
        nodes.add(u); nodes.add(v)
        outdeg[u] = outdeg.get(u, 0) + 1

N = max(nodes) + 1
num_stripes = math.ceil(N / block_size)

def create_edge_stripes():
    stripe_fs = [open(f"stripe_{i}.txt", "w") for i in range(num_stripes)]
    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = map(int, parts)
            sid = v // block_size
            stripe_fs[sid].write(f"{u} {v}\n")
    for sf in stripe_fs:
        sf.close()

# 2. 初始化 PageRank：将 r 分块写入磁盘
def write_initial_r():
    for sid in range(num_stripes):
        start = sid * block_size
        end = min((sid + 1) * block_size, N)
        with open(f"r_stripe_{sid}.txt", "w") as rf:
            for _ in range(start, end):
                rf.write(f"{1.0/N}\n")

# 3. 外存化 PageRank 迭代

def external_stripe_pagerank():
    for iter_num in range(max_iter):
        # 计算流失
        leaked = 0.0
        for sid in range(num_stripes):
            start = sid * block_size
            with open(f"r_stripe_{sid}.txt") as rf:
                for local_idx, line in enumerate(rf):
                    i = start + local_idx
                    val = float(line)
                    if outdeg.get(i, 0) == 0:
                        leaked += beta * val
        leaked_share = leaked / N

        diff = 0.0
        for vid in range(num_stripes):
            v_start = vid * block_size
            v_end = min((vid + 1) * block_size, N)
            size_v = v_end - v_start
            r_new = [(1 - beta) / N + leaked_share for _ in range(size_v)]

            for uid in range(num_stripes):
                u_start = uid * block_size
                u_end = min((uid + 1) * block_size, N)
                with open(f"r_stripe_{uid}.txt") as uf:
                    r_block = [float(x) for x in uf]
                stripe_file = f"stripe_{vid}.txt"
                if not os.path.exists(stripe_file):
                    continue
                with open(stripe_file) as ef:
                    for line in ef:
                        u, v = map(int, line.split())
                        if u_start <= u < u_end:
                            out = outdeg.get(u, 0)
                            if out:
                                r_new[v - v_start] += beta * (r_block[u - u_start] / out)

            new_file = f"r_new_stripe_{vid}.txt"
            with open(new_file, "w") as wf:
                for val in r_new:
                    wf.write(f"{val}\n")
            with open(f"r_stripe_{vid}.txt") as oldf, open(new_file) as newf:
                for old_line, new_line in zip(oldf, newf):
                    diff += abs(float(new_line) - float(old_line))

        for sid in range(num_stripes):
            os.replace(f"r_new_stripe_{sid}.txt", f"r_stripe_{sid}.txt")
        if diff < epsilon:
            break

# 4. 合并各分块并流式输出 Top100
def merge_top100(output_file="res_stripe_ext.txt", k=100):
    # 先对每个条带排序并写入临时文件
    sorted_files = []
    for sid in range(num_stripes):
        start = sid * block_size
        scores = []
        with open(f"r_stripe_{sid}.txt") as rf:
            for idx, line in enumerate(rf):
                scores.append((float(line), start + idx))
        scores.sort(reverse=True)
        tmp = f"sorted_{sid}.txt"
        with open(tmp, "w") as sf:
            for score, nid in scores:
                sf.write(f"{nid}\t{score}\n")
        sorted_files.append(tmp)

    # k 路归并写 Top k
    files = [open(f, "r") for f in sorted_files]
    heap = []
    for sid, f in enumerate(files):
        line = f.readline().strip()
        if not line: continue
        nid, score = line.split("\t")
        heapq.heappush(heap, (-float(score), sid, int(nid)))

    with open(output_file, "w") as out:
        for _ in range(k):
            if not heap: break
            neg, sid, nid = heapq.heappop(heap)
            out.write(f"{nid}\t{-neg}\n")
            line = files[sid].readline().strip()
            if line:
                nid2, score2 = line.split("\t")
                heapq.heappush(heap, (-float(score2), sid, int(nid2)))

    for f in files:
        f.close()
    for tmp in sorted_files:
        os.remove(tmp)

# 主流程
if __name__ == '__main__':
    create_edge_stripes()
    write_initial_r()
    external_stripe_pagerank()
    merge_top100()
    print("External-stripe PageRank top100 wrote to res_stripe_ext.txt")
