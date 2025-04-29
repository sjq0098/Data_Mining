import numpy as np
import os
import pickle
from collections import defaultdict

class StreamingSparsePageRank:
    def __init__(self, data_file, block_size=100, buffer_size=10000, alpha=0.85, max_iter=100, tol=1e-6):
        self.data_file = data_file
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.N = 0
        self.block_dir = "matrix_blocks"
        
    def prepare_directory(self):
        if os.path.exists(self.block_dir):
            for f in os.listdir(self.block_dir):
                os.remove(os.path.join(self.block_dir, f))
        else:
            os.makedirs(self.block_dir)
            
    def stream_count_nodes(self):
        """流式统计节点数"""
        nodes = set()
        with open(self.data_file, 'r') as f:
            while True:
                lines = f.readlines(self.buffer_size)
                if not lines:
                    break
                for line in lines:
                    src, dst = map(int, line.strip().split())
                    nodes.add(src)
                    nodes.add(dst)
        
        # 创建节点映射
        node_list = sorted(list(nodes))
        self.node_map = {node: idx for idx, node in enumerate(node_list)}
        self.reverse_map = {idx: node for idx, node in enumerate(node_list)}
        self.N = len(nodes)
        self.num_blocks = (self.N + self.block_size - 1) // self.block_size
        print(f"总节点数: {self.N}")
        print(f"块大小: {self.block_size}")
        print(f"总块数: {self.num_blocks}")
        
    def stream_compute_outdegrees(self):
        """流式计算出度"""
        out_degrees = np.zeros(self.N, dtype=np.float64)  # 使用float64确保精度
        with open(self.data_file, 'r') as f:
            while True:
                lines = f.readlines(self.buffer_size)
                if not lines:
                    break
                for line in lines:
                    src, dst = map(int, line.strip().split())
                    src_idx = self.node_map[src]
                    out_degrees[src_idx] += 1
        return out_degrees
        
    def create_blocks(self):
        """流式创建矩阵块，使用block-stripe方法"""
        # 流式计算出度
        out_degrees = self.stream_compute_outdegrees()
        
        # 创建临时块存储
        blocks = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        edges_count = 0
        
        # 流式处理边
        with open(self.data_file, 'r') as f:
            while True:
                lines = f.readlines(self.buffer_size)
                if not lines:
                    break
                    
                for line in lines:
                    src, dst = map(int, line.strip().split())
                    src_idx = self.node_map[src]
                    dst_idx = self.node_map[dst]
                    
                    if out_degrees[src_idx] > 0:
                        # 计算块索引
                        row_block = dst_idx // self.block_size
                        col_block = src_idx // self.block_size
                        
                        # 计算块内局部坐标
                        local_row = dst_idx % self.block_size
                        local_col = src_idx % self.block_size
                        
                        # 累加相同位置的值
                        blocks[row_block][col_block][(local_row, local_col)] += 1.0 / out_degrees[src_idx]
                        
                        edges_count += 1
                        if edges_count % 50000 == 0:
                            # 定期保存并清理块
                            self._save_and_clear_blocks(blocks)
        
        # 保存剩余的块
        self._save_and_clear_blocks(blocks)
        
        # 保存出度信息
        with open(os.path.join(self.block_dir, 'out_degrees.pkl'), 'wb') as f:
            pickle.dump(out_degrees, f)
            
        return out_degrees
    
    def _save_and_clear_blocks(self, blocks):
        """保存并清理块"""
        for row_block in list(blocks.keys()):
            for col_block in list(blocks[row_block].keys()):
                if blocks[row_block][col_block]:
                    # 将字典转换为列表格式
                    entries = []
                    for (row, col), val in blocks[row_block][col_block].items():
                        entries.append((row, col, val))
                    
                    # 保存块数据
                    block_file = os.path.join(self.block_dir, f'block_{row_block}_{col_block}.pkl')
                    
                    # 如果文件已存在，合并数据
                    if os.path.exists(block_file):
                        with open(block_file, 'rb') as f:
                            existing_entries = pickle.load(f)
                        # 合并现有数据和新数据
                        entry_dict = defaultdict(float)
                        for row, col, val in existing_entries:
                            entry_dict[(row, col)] += val
                        for row, col, val in entries:
                            entry_dict[(row, col)] += val
                        entries = [(row, col, val) for (row, col), val in entry_dict.items()]
                    
                    with open(block_file, 'wb') as f:
                        pickle.dump(entries, f)
                    
                    # 清理内存
                    blocks[row_block][col_block].clear()
            blocks[row_block].clear()
        blocks.clear()
    
    def block_vector_multiply(self, v, row_block):
        """计算块与向量的乘积"""
        block_height = min(self.block_size, self.N - row_block * self.block_size)
        result = np.zeros(block_height, dtype=np.float64)  # 使用float64确保精度
        
        # 对该行的所有块进行计算
        for col_block in range(self.num_blocks):
            block_file = os.path.join(self.block_dir, f'block_{row_block}_{col_block}.pkl')
            if os.path.exists(block_file):
                with open(block_file, 'rb') as f:
                    entries = pickle.load(f)
                
                # 获取对应的向量片段
                start_col = col_block * self.block_size
                end_col = min((col_block + 1) * self.block_size, self.N)
                v_block = v[start_col:end_col]
                
                # 计算块的贡献
                for row, col, val in entries:
                    if col < len(v_block):  # 确保不越界
                        result[row] += val * v_block[col]
        
        return result
    
    def compute_pagerank(self):
        """计算PageRank值"""
        print("准备目录...")
        self.prepare_directory()
        
        print("流式统计节点...")
        self.stream_count_nodes()
        
        print("创建矩阵块...")
        out_degrees = self.create_blocks()
        
        # 初始化PageRank向量
        r = np.ones(self.N, dtype=np.float64) / self.N  # 使用float64确保精度
        
        # 找出dead ends
        dead_ends = (out_degrees == 0)
        
        print("开始迭代计算PageRank...")
        for iteration in range(self.max_iter):
            new_r = np.zeros(self.N, dtype=np.float64)  # 使用float64确保精度
            
            # 计算dead ends的贡献
            dead_ends_contribution = np.sum(r[dead_ends]) / self.N
            
            # 分块计算矩阵向量乘法
            for i in range(self.num_blocks):
                start_idx = i * self.block_size
                end_idx = min((i + 1) * self.block_size, self.N)
                new_r[start_idx:end_idx] = self.block_vector_multiply(r, i)
            
            # 应用PageRank公式
            new_r = self.alpha * (new_r + dead_ends_contribution) + (1 - self.alpha) / self.N
            
            # 归一化
            new_r = new_r / np.sum(new_r)
            
            # 计算收敛误差
            diff = np.sum(np.abs(new_r - r))
            print(f"迭代 {iteration + 1}: 误差 = {diff:.6e}")
            
            if diff < self.tol:
                print(f"已收敛，总迭代次数: {iteration + 1}")
                break
                
            r = new_r
        
        return r
    
    def save_results(self, scores, output_file, top_k=100):
        """保存top-k的PageRank结果"""
        # 使用部分排序减少内存使用
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        with open(output_file, 'w') as f:
            for idx in top_indices:
                f.write(f"{self.reverse_map[idx]} {scores[idx]:.8f}\n")
        print(f"已将Top-{top_k}结果写入{output_file}")

def main():
    data_file = "Data.txt"
    output_file = "Res_opti.txt"
    block_size = 100  # 块大小
    buffer_size = 10000  # 读取缓冲区大小
    
    pr = StreamingSparsePageRank(data_file, block_size=block_size, buffer_size=buffer_size)
    scores = pr.compute_pagerank()
    pr.save_results(scores, output_file)

if __name__ == "__main__":
    main()