import numpy as np
import os
import pickle
from collections import defaultdict

class StripePageRank:
    def __init__(self, data_file, stripe_size=1000, alpha=0.85, max_iter=100, tol=1e-6):
        self.data_file = data_file
        self.stripe_size = stripe_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.stripe_dir = "matrix_stripes"
        self.node_map = {}
        self.reverse_map = {}
        
    def prepare_directory(self):
        """准备存储矩阵条带的目录"""
        if os.path.exists(self.stripe_dir):
            # 清理旧的矩阵条带
            for f in os.listdir(self.stripe_dir):
                os.remove(os.path.join(self.stripe_dir, f))
        else:
            os.makedirs(self.stripe_dir)
            
    def read_and_map_nodes(self):
        """第一遍扫描：读取并映射节点"""
        nodes = set()
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                nodes.add(src)
                nodes.add(dst)
        
        nodes = sorted(list(nodes))
        self.N = len(nodes)
        self.node_map = {node: idx for idx, node in enumerate(nodes)}
        self.reverse_map = {idx: node for node, idx in self.node_map.items()}
        
        # 计算需要的条带数
        self.num_stripes = (self.N + self.stripe_size - 1) // self.stripe_size
        print(f"总节点数: {self.N}")
        print(f"条带大小: {self.stripe_size}")
        print(f"总条带数: {self.num_stripes}")
        
    def create_stripes(self):
        """第二遍扫描：创建矩阵条带并存储到硬盘"""
        # 初始化出度计数
        out_degrees = np.zeros(self.N)
        
        # 第一遍：计算出度
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                src_idx = self.node_map[src]
                out_degrees[src_idx] += 1
        
        # 创建临时的条带数据结构
        stripes = defaultdict(list)
        
        # 第二遍：构建条带
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                src_idx = self.node_map[src]
                dst_idx = self.node_map[dst]
                
                if out_degrees[src_idx] > 0:
                    # 计算目标节点所在的条带
                    stripe_idx = dst_idx // self.stripe_size
                    
                    # 将边的信息添加到对应的条带中
                    # 存储格式：(行在条带中的位置，列，值)
                    stripes[stripe_idx].append((
                        dst_idx % self.stripe_size,
                        src_idx,
                        1.0 / out_degrees[src_idx]
                    ))
        
        # 将条带写入文件
        for i in range(self.num_stripes):
            if stripes[i]:  # 如果条带非空
                stripe_data = {
                    'entries': stripes[i],
                    'size': min(self.stripe_size, self.N - i * self.stripe_size)
                }
                with open(os.path.join(self.stripe_dir, f'stripe_{i}.pkl'), 'wb') as f:
                    pickle.dump(stripe_data, f)
        
        # 保存出度信息，用于处理dead ends
        with open(os.path.join(self.stripe_dir, 'out_degrees.pkl'), 'wb') as f:
            pickle.dump(out_degrees, f)
            
        return out_degrees
    
    def stripe_vector_multiply(self, v, stripe_idx):
        """计算一个条带与向量的乘积"""
        result = np.zeros(min(self.stripe_size, self.N - stripe_idx * self.stripe_size))
        
        stripe_file = os.path.join(self.stripe_dir, f'stripe_{stripe_idx}.pkl')
        if os.path.exists(stripe_file):
            with open(stripe_file, 'rb') as f:
                stripe_data = pickle.load(f)
            
            # 计算条带的贡献
            for row, col, val in stripe_data['entries']:
                result[row] += val * v[col]
        
        return result
    
    def compute_pagerank(self):
        """计算PageRank值"""
        print("准备目录...")
        self.prepare_directory()
        
        print("读取并映射节点...")
        self.read_and_map_nodes()
        
        print("创建矩阵条带...")
        out_degrees = self.create_stripes()
        
        # 初始化PageRank向量
        r = np.ones(self.N) / self.N
        
        # 找出dead ends
        dead_ends = (out_degrees == 0)
        
        print("开始迭代计算PageRank...")
        for iteration in range(self.max_iter):
            new_r = np.zeros(self.N)
            
            # 计算dead ends的贡献
            dead_ends_contribution = np.sum(r[dead_ends]) / self.N
            
            # 分条带计算矩阵向量乘法
            for i in range(self.num_stripes):
                start_idx = i * self.stripe_size
                end_idx = min((i + 1) * self.stripe_size, self.N)
                new_r[start_idx:end_idx] = self.stripe_vector_multiply(r, i)
            
            # 应用PageRank公式
            new_r = self.alpha * (new_r + dead_ends_contribution) + (1 - self.alpha) / self.N
            
            # 计算收敛误差
            diff = np.sum(np.abs(new_r - r))
            print(f"迭代 {iteration + 1}: 误差 = {diff:.6e}")
            
            if diff < self.tol:
                print(f"已收敛，总迭代次数: {iteration + 1}")
                break
                
            r = new_r.copy()
        
        return r
    
    def save_results(self, scores, output_file, top_k=100):
        """保存top-k的PageRank结果"""
        top_indices = np.argsort(-scores)[:top_k]
        with open(output_file, 'w') as f:
            for idx in top_indices:
                original_id = self.reverse_map[idx]
                f.write(f"{original_id} {scores[idx]:.8f}\n")
        print(f"已将Top-{top_k}结果写入{output_file}")

def main():
    data_file = "Data.txt"
    output_file = "Res_opti.txt"
    stripe_size = 950  # 可以根据实际情况调整
    
    pr = StripePageRank(data_file, stripe_size=stripe_size)
    scores = pr.compute_pagerank()
    pr.save_results(scores, output_file)

if __name__ == "__main__":
    main() 