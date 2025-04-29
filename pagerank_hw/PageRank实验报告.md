# <center>pagerank优化实验报告</center>

<center>申健强 &nbsp;&nbsp;&nbsp;是忻言&nbsp;&nbsp;&nbsp;仇科文  </center>
<center> 2313119&nbsp;&nbsp; 2311848 &nbsp; &nbsp;2312237</center>

## 一、实验目的
在给定网页链接数据集(Data.txt)上实现PageRank算法，通过块矩阵与稀疏矩阵技术优化内存使用，满足最大内存80MB、运行时间60秒的性能约束，输出Top 100节点及其PageRank分数。

### 具体要求：算法与输出

- 程序需读取*Data.txt*，计算 PageRank，并将 Top 100 节点按以下格式输出到*Res.txt*：`NodeID   Score` 
- 需处理**dead-ends 和 spider-traps**
- 尽可能优化内存使用，**块矩阵和稀疏矩阵优化为强制要求**
- 程序需迭代至收敛

## 二、实验原理阐述

PageRank是基于图的链接分析的经典算法，它通过分析网页间的链接关系来确定每个网页的重要性，并将结果用于搜索结果排序。

PageRank 算法的基本想法是在有向图上定义一个随机游走模型，即一阶马尔可夫链，描述随机游走者沿着有向图随机访问各个结点的行为。在一定条件下，极限情况访问每个结点的概率将趋于稳定，即计算收敛。
一般而言，PageRank算法的迭代公式如下：

\[
PR^{(t+1)}(p_i) \;=\;\sum_{p_j \,\to\, p_i} \frac{PR^{(t)}(p_j)}{L(p_j)}
\]

但对于现实情况而言，对于一张表示链接关系的有向图而言存在`dead-ends` 和 `spider-traps`现象：

- **Dead-Ends**:
**定义**：出度为0的节点（无外链的网页）  
**数学表现**：转移矩阵对应列全为0  
**危害**：导致PageRank分数持续泄漏，最终收敛到0向量  
```math
\sum_{i=1}^N PR(p_i) < 1
```

- **Spider-Traps**
**定义**：形成闭环的节点组（自循环或小范围循环）  
**典型结构**：A→B→C→A  
**危害**：PageRank分数在闭环内无限累积  
```math
\lim_{k→∞} PR_{closed\_group} = 1
```

于是我们需要对这样的情况进行处理，我们的处理方法如下：
#### 1. Teleport机制（随机跳转）  
**核心公式修正**：
```math
PR(p_i) = \frac{1-\beta}{N} + \beta\sum_{p_j∈M(p_i)}\frac{PR(p_j)}{L(p_j)}
```
**参数说明**：
- β：跟随链接的概率（通常取0.85）
- 1-β：随机跳转概率
- N：总节点数

#### 2. 数学本质
将原始转移矩阵修正为：
```math
M' = \beta M + \frac{(1-\beta)}{N}E
```
其中E是全1矩阵，保证：
- 矩阵随机性（列和为1）
- 不可约性（任意状态可达）
- 非周期性（保证收敛）

---


## 三、实验设计和实现与优化
### 数据集描述
#### 1. 数据格式
- 文件：Data.txt
- 行格式：`FromNodeID ToNodeID`
#### 2.数据规模
| 统计项 | 数值 |
|--------|------|
| 总节点数 | 9500 |
| 总边数  | 150000 |
| 最大节点编号 | 10000 |
| 平均出度数量 | 15.79 |

*在本次实验中可以不构建索引完成实验，所以统计量节点数最终起到的作用不大,但对于我们验证使用内存的规模时意义巨大*

### 实验设计
（这里讲述最终方案是如何做的尤其是以下代码详解）

代码的核心步骤是基于流式处理条带文件的PageRank
```algorithm
Algorithm: stripe_based_pagerank(r, outdeg, N, block_size, β, ε, max_iter)
Input:
    r[0..N–1]         – 初始 PageRank 向量  
    outdeg            – 节点出度映射  
    N                 – 总节点数  
    block_size        – 条带宽度  
    β (beta)          – damping factor  
    ε (epsilon)       – 收敛阈值  
    max_iter          – 最大迭代次数  
Output:
    r                 – 收敛后的 PageRank 向量

1.  num_stripes ← ceil(N / block_size)  
2.  for iter_num ← 1 to max_iter do  
3.      // ① 初始化新向量  
4.      for i ← 0 to N–1 do  
5.          r_new[i] ← (1 – β) / N  
6.  
7.      // ② 按条带流式更新  
8.      for stripe_id ← 0 to num_stripes–1 do  
9.          stripe_file ← "stripe_" ∥ stripe_id ∥ ".txt"  
10.         if file_exists(stripe_file) then  
11.             for each line in stripe_file do  
12.                 parse u, v from line  
13.                 if outdeg[u] > 0 then  
14.                     r_new[v] ← r_new[v] + β * (r[u] / outdeg[u])  
15.  
16.     // ③ 处理 dead-ends 泄漏  
17.     leaked ← 0  
18.     for i ← 0 to N–1 do  
19.         if outdeg[i] = 0 then  
20.             leaked ← leaked + β * r[i]  
21.     leaked_share ← leaked / N  
22.     for i ← 0 to N–1 do  
23.         r_new[i] ← r_new[i] + leaked_share  
24.  
25.     // ④ 收敛判断  
26.     diff ← 0  
27.     for i ← 0 to N–1 do  
28.         diff ← diff + |r_new[i] – r[i]|  
29.     if diff < ε then  
30.         break  
31.     r ← r_new  
32.  
33. return r
```

```python
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
    print("External-stripe PageRank top100 wrote to Res.txt")

```
### 实验结果及分析
| 平均内存使用 | 平均运行时间 | 优化方向 |
|------------|------------|---------|
| 14.96 MB | 1.17 s | 稀疏+外部条带+泄漏处理 |


## 四、优化历程与有效优化组件总结

（用表格描述这个过程）

(普通normal，700MB)->(加入稀疏矩阵70MB,方案已经放弃)->(外部条块化矩阵优化，40MB量级，有两个)->(稀疏存储+流处理,30MB)->(将输入数据当作矩阵作稀疏处理,13MB)->(建索引发现回到30MB时代)->(条块优化解决IO问题和应对极端数据量情况，14MB)/(分块处理，14MB但比较慢)

（列出在优化过程中有效的优化组件）



(调用库Networkx的实现，140MB)

## 五、实验心得与改进
