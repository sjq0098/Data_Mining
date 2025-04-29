import numpy as np
from scipy import sparse
import time

class PageRank:
    def __init__(self, data_file, alpha=0.85, max_iter=100, tol=1e-6):
        self.data_file = data_file
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0  # 总节点数
        self.edges = []  # 边列表
        self.node_map = {}  # 节点ID到索引的映射
        self.reverse_map = {}  # 索引到节点ID的映射
        
    def read_data(self):
        """读取数据并构建节点映射"""
        nodes = set()
        self.edges = []
        
        # 第一遍扫描：获取所有唯一节点
        with open(self.data_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                nodes.add(src)
                nodes.add(dst)
                self.edges.append((src, dst))
        
        # 构建节点映射
        nodes = sorted(list(nodes))
        self.N = len(nodes)
        self.node_map = {node: idx for idx, node in enumerate(nodes)}
        self.reverse_map = {idx: node for node, idx in self.node_map.items()}
        
    def build_sparse_matrix(self):
        """构建稀疏转移矩阵"""
        rows = []
        cols = []
        data = []
        out_degrees = np.zeros(self.N)
        
        # 计算出度
        for src, dst in self.edges:
            src_idx = self.node_map[src]
            out_degrees[src_idx] += 1
        
        # 构建稀疏矩阵元素
        for src, dst in self.edges:
            src_idx = self.node_map[src]
            dst_idx = self.node_map[dst]
            if out_degrees[src_idx] > 0:
                rows.append(dst_idx)
                cols.append(src_idx)
                data.append(1.0 / out_degrees[src_idx])
        
        # 创建CSR格式的稀疏矩阵
        M = sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        return M, out_degrees
    
    def block_stripe_multiply(self, M, v, block_size):
        """使用block-stripe优化的矩阵向量乘法"""
        result = np.zeros(self.N)
        for i in range(0, self.N, block_size):
            end = min(i + block_size, self.N)
            # 获取矩阵的一个block
            M_block = M[i:end, :]
            # 计算这个block的结果
            result[i:end] = M_block.dot(v)
        return result
    
    def compute(self, block_size=1000):
        """计算PageRank值"""
        print("开始读取数据...")
        self.read_data()
        print(f"总节点数: {self.N}")
        print(f"总边数: {len(self.edges)}")
        
        print("构建稀疏转移矩阵...")
        M, out_degrees = self.build_sparse_matrix()
        
        # 初始化PageRank向量
        r = np.ones(self.N) / self.N
        
        # 找出dead ends（出度为0的节点）
        dead_ends = (out_degrees == 0)
        
        print("开始迭代计算PageRank...")
        for iteration in range(self.max_iter):
            start_time = time.time()
            
            # 计算dead ends的贡献
            dead_ends_contribution = np.sum(r[dead_ends]) / self.N
            
            # 使用block-stripe优化的矩阵乘法
            new_r = self.block_stripe_multiply(M, r, block_size)
            
            # 应用PageRank公式
            new_r = self.alpha * (new_r + dead_ends_contribution) + (1 - self.alpha) / self.N
            
            # 计算收敛误差
            diff = np.sum(np.abs(new_r - r))
            iteration_time = time.time() - start_time
            
            print(f"迭代 {iteration + 1}: 误差 = {diff:.6e}, 用时 = {iteration_time:.2f}秒")
            
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
    # 设置参数
    data_file = "Data.txt"
    output_file = "Res_opti.txt"
    alpha = 0.85
    max_iter = 100
    tol = 1e-6
    block_size = 1000
    
    # 创建PageRank实例并计算
    pr = PageRank(data_file, alpha, max_iter, tol)
    scores = pr.compute(block_size)
    
    # 保存结果
    pr.save_results(scores, output_file)

if __name__ == "__main__":
    main() 