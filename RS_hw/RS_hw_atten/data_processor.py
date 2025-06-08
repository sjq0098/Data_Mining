import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from model import sparse_mx_to_torch_sparse_tensor

class RatingDataset(Dataset):
    """评分数据集"""
    def __init__(self, user_item_pairs, ratings=None):
        """
        参数:
        user_item_pairs: 用户-物品对的列表 [(user_id, item_id), ...]
        ratings: 评分列表，如果为None则为测试集
        """
        self.user_item_pairs = user_item_pairs
        self.ratings = ratings
        self.is_test = ratings is None
        
    def __len__(self):
        return len(self.user_item_pairs)
        
    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        
        if self.is_test:
            return torch.LongTensor([user]), torch.LongTensor([item])
        else:
            rating = self.ratings[idx]
            return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([rating])


class DataProcessor:
    """处理训练和测试数据"""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.user_item_matrix = None
        self.user_map = {}  # 用户ID到索引的映射
        self.item_map = {}  # 物品ID到索引的映射
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
    def process_data(self, train_ratio=0.8, batch_size=1024):
        """处理数据并返回数据加载器"""
        print("读取训练数据...")
        train_users, train_items, train_ratings = self._read_train_data(self.train_path)
        
        print("读取测试数据...")
        test_users, test_items = self._read_test_data(self.test_path)
        
        print(f"统计信息: {len(set(train_users))} 用户, {len(set(train_items))} 物品, {len(train_ratings)} 评分")
        print(f"测试数据: {len(set(test_users))} 用户, {len(test_items)} 测试物品")
        
        print("创建用户-物品图...")
        # 创建用户和物品的映射
        self._create_mappings(train_users, train_items, test_users, test_items)
        
        # 将原始ID映射到索引
        train_users_idx = [self.user_map[u] for u in train_users]
        train_items_idx = [self.item_map[i] for i in train_items]
        test_users_idx = [self.user_map[u] for u in test_users]
        test_items_idx = [self.item_map[i] for i in test_items]
        
        # 创建用户-物品交互矩阵
        self.user_item_matrix = self._create_user_item_matrix(
            train_users_idx, train_items_idx, train_ratings
        )
        
        # 创建邻接矩阵
        adj_matrix = self._create_adj_matrix()
        
        # 划分训练集和验证集
        n_train = len(train_users_idx)
        indices = np.random.permutation(n_train)
        split = int(train_ratio * n_train)
        train_idx, valid_idx = indices[:split], indices[split:]
        
        train_u = [train_users_idx[i] for i in train_idx]
        train_i = [train_items_idx[i] for i in train_idx]
        train_r = [train_ratings[i] for i in train_idx]
        
        valid_u = [train_users_idx[i] for i in valid_idx]
        valid_i = [train_items_idx[i] for i in valid_idx]
        valid_r = [train_ratings[i] for i in valid_idx]
        
        # 创建数据集
        train_dataset = RatingDataset(list(zip(train_u, train_i)), train_r)
        valid_dataset = RatingDataset(list(zip(valid_u, valid_i)), valid_r)
        test_dataset = RatingDataset(list(zip(test_users_idx, test_items_idx)))
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        self.train_data = train_loader
        self.valid_data = valid_loader
        self.test_data = test_loader
        
        return train_loader, valid_loader, test_loader, adj_matrix
    
    def _read_train_data(self, filepath):
        """读取训练数据"""
        users, items, ratings = [], [], []
        current_user = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    # 用户行
                    parts = line.split('|')
                    current_user = int(parts[0])
                else:
                    # 评分行
                    parts = line.split()
                    if len(parts) >= 2:
                        item_id, rating = int(parts[0]), float(parts[1])
                        users.append(current_user)
                        items.append(item_id)
                        ratings.append(rating)
        
        return users, items, ratings
    
    def _read_test_data(self, filepath):
        """读取测试数据"""
        users, items = [], []
        current_user = None
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    # 用户行
                    parts = line.split('|')
                    current_user = int(parts[0])
                else:
                    # 物品行
                    if line:
                        item_id = int(line.strip())
                        users.append(current_user)
                        items.append(item_id)
        
        return users, items
    
    def _create_mappings(self, train_users, train_items, test_users, test_items):
        """创建用户和物品ID到索引的映射"""
        unique_users = set(train_users + test_users)
        unique_items = set(train_items + test_items)
        
        for i, user_id in enumerate(unique_users):
            self.user_map[user_id] = i
            
        for i, item_id in enumerate(unique_items):
            self.item_map[item_id] = i
            
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
    
    def _create_user_item_matrix(self, users, items, ratings):
        """创建用户-物品评分矩阵"""
        matrix = sp.csr_matrix((ratings, (users, items)), 
                                shape=(self.n_users, self.n_items))
        return matrix
    
    def _create_adj_matrix(self):
        """创建邻接矩阵 [A_ui; A_iu]"""
        # 用户-物品评分矩阵
        user_item_matrix = self.user_item_matrix
        
        # 创建二部图邻接矩阵 [0, R; R^T, 0]
        adj = sp.vstack([
            sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), user_item_matrix]),
            sp.hstack([user_item_matrix.T, sp.csr_matrix((self.n_items, self.n_items))])
        ])
        
        # 转换为PyTorch稀疏张量
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        
        return adj_tensor
    
    def get_test_reference(self):
        """获取测试集的用户和物品ID用于生成结果"""
        test_users = []
        test_items = []
        
        for users, items in self.test_data:
            batch_users = users.squeeze().tolist()
            batch_items = items.squeeze().tolist()
            
            # 处理单条记录的情况
            if isinstance(batch_users, int):
                batch_users = [batch_users]
                batch_items = [batch_items]
                
            for u_idx, i_idx in zip(batch_users, batch_items):
                # 映射回原始ID
                u_id = list(self.user_map.keys())[list(self.user_map.values()).index(u_idx)]
                i_id = list(self.item_map.keys())[list(self.item_map.values()).index(i_idx)]
                test_users.append(u_id)
                test_items.append(i_id)
                
        return test_users, test_items
