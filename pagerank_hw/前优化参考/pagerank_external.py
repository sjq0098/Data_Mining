import numpy as np
import os
import pickle
from collections import defaultdict

class ExternalPageRank:
    def __init__(self, data_file, block_size=100, alpha=0.85, max_iter=100, tol=1e-6):
        self.data_file = data_file
        self.block_size = block_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.matrix_dir = "matrix_blocks"
        self.node_map = {}
        self.reverse_map = {}
        
    def prepare_directory(self):
        """准备存储矩阵块的目录"""
        if os.path.exists(self.matrix_dir):
            # 清理旧的矩阵块
            for f in os.listdir(self.matrix_dir):
                os.remove(os.path.join(self.matrix_dir, f))
        else:
            os.makedirs(self.matrix_dir)
            
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
        
        # 计算需要的块数
        self.num_blocks = (self.N + self.block_size - 1) // self.block_size
        print(f"总节点数: {self.N}")
        print(f"块大小: {self.block_size}")
        print(f"总块数: {self.num_blocks}")
        
    def create_matrix_blocks(self):
        """第二遍扫描：创建矩阵块并存储到硬盘"""
        # 初始化出度计数
        out_degrees = np.zeros(self.N)
        
        # 第一遍：计算出度
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                src_idx = self.node_map[src]
                out_degrees[src_idx] += 1
        
        # 创建临时的块数据结构
        blocks = defaultdict(lambda: defaultdict(list))
        
        # 第二遍：构建块
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                src_idx = self.node_map[src]
                dst_idx = self.node_map[dst]
                
                if out_degrees[src_idx] > 0:
                    # 计算源节点和目标节点所在的块
                    src_block = src_idx // self.block_size
                    dst_block = dst_idx // self.block_size
                    
                    # 将边的信息添加到对应的块中
                    # 存储格式：(行在块中的位置，列在块中的位置，值)
                    blocks[dst_block][src_block].append((
                        dst_idx % self.block_size,
                        src_idx % self.block_size,
                        1.0 / out_degrees[src_idx]
                    ))
        
        # 将块写入文件
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if blocks[i][j]:  # 如果块非空
                    block_data = {
                        'entries': blocks[i][j],
                        'shape': (
                            min(self.block_size, self.N - i * self.block_size),
                            min(self.block_size, self.N - j * self.block_size)
                        )
                    }
                    with open(os.path.join(self.matrix_dir, f'block_{i}_{j}.pkl'), 'wb') as f:
                        pickle.dump(block_data, f)
        
        # 保存出度信息，用于处理dead ends
        with open(os.path.join(self.matrix_dir, 'out_degrees.pkl'), 'wb') as f:
            pickle.dump(out_degrees, f)
            
        return out_degrees
    
    def block_vector_multiply(self, v, block_row):
        """计算一个块行与向量的乘积"""
        result = np.zeros(min(self.block_size, self.N - block_row * self.block_size))
        
        # 对该行的每个块进行计算
        for j in range(self.num_blocks):
            block_file = os.path.join(self.matrix_dir, f'block_{block_row}_{j}.pkl')
            if os.path.exists(block_file):
                with open(block_file, 'rb') as f:
                    block_data = pickle.load(f)
                
                # 获取这个块对应的向量部分
                v_start = j * self.block_size
                v_end = min((j + 1) * self.block_size, self.N)
                v_block = v[v_start:v_end]
                
                # 计算这个块的贡献
                for row, col, val in block_data['entries']:
                    result[row] += val * v_block[col]
        
        return result
    
    def compute_pagerank(self):
        """计算PageRank值"""
        print("准备目录...")
        self.prepare_directory()
        
        print("读取并映射节点...")
        self.read_and_map_nodes()
        
        print("创建矩阵块...")
        out_degrees = self.create_matrix_blocks()
        
        # 初始化PageRank向量
        r = np.ones(self.N) / self.N
        
        # 找出dead ends
        dead_ends = (out_degrees == 0)
        
        print("开始迭代计算PageRank...")
        for iteration in range(self.max_iter):
            new_r = np.zeros(self.N)
            
            # 计算dead ends的贡献
            dead_ends_contribution = np.sum(r[dead_ends]) / self.N
            
            # 分块计算矩阵向量乘法
            for i in range(self.num_blocks):
                start_idx = i * self.block_size
                end_idx = min((i + 1) * self.block_size, self.N)
                new_r[start_idx:end_idx] = self.block_vector_multiply(r, i)
            
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
    block_size = 100  # 可以根据实际情况调整
    
    pr = ExternalPageRank(data_file, block_size=block_size)
    scores = pr.compute_pagerank()
    pr.save_results(scores, output_file)

if __name__ == "__main__":
    main() 