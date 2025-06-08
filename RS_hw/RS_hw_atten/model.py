import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder_mu = nn.Linear(input_dim, hidden_dim)
        self.encoder_logvar = nn.Linear(input_dim, hidden_dim)
        
        # 解码器
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class GumberlSampler(nn.Module):
    def __init__(self, temperature=1.0):
        super(GumberlSampler, self).__init__()
        self.temperature = temperature
        
    def forward(self, logits):
        # Gumbel-Softmax 重参数化技巧
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y_soft = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        return y_soft
        
        
class BernoulliSampler(nn.Module):
    def __init__(self):
        super(BernoulliSampler, self).__init__()
        
    def forward(self, probs):
        # 训练时使用连续松弛
        if self.training:
            noise = torch.rand_like(probs)
            samples = torch.sigmoid((torch.log(probs + 1e-10) - torch.log(1 - probs + 1e-10) + torch.log(noise + 1e-10) - torch.log(1 - noise + 1e-10)))
            return samples
        # 测试时进行离散采样
        else:
            return (probs > 0.5).float()


class GraphConvolution(nn.Module):
    def __init__(self, adj_matrix):
        super(GraphConvolution, self).__init__()
        # 规范化邻接矩阵
        self.register_buffer('adj', self._normalize_adj(adj_matrix))
        
    def _normalize_adj(self, adj):
        """对邻接矩阵进行归一化处理"""
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum + 1e-10, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # A' = D^(-0.5) * A * D^(-0.5)
        return torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
    def forward(self, x):
        # 图卷积操作: H = A' * X
        return torch.sparse.mm(self.adj, x)


class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, adj_matrix, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # 用户和物品的嵌入矩阵
        self.user_embedding = nn.Embedding(user_num, embedding_dim)
        self.item_embedding = nn.Embedding(item_num, embedding_dim)
        
        # 初始化嵌入
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # 图卷积层
        self.gcn = GraphConvolution(adj_matrix)
        
        # VAE层，每层GCN输出都会有一个对应的VAE
        self.vaes = nn.ModuleList([
            VAE(embedding_dim, embedding_dim) for _ in range(n_layers + 1)
        ])
        
        # 层级注意力机制
        self.layer_attention = nn.Linear((n_layers + 1) * embedding_dim, n_layers + 1)
        
    def forward(self, users=None, items=None):
        # 初始嵌入
        user_embs = self.user_embedding.weight
        item_embs = self.item_embedding.weight
        
        # 所有节点的嵌入矩阵
        all_embs = torch.cat([user_embs, item_embs], dim=0)
        
        # 保存每一层的嵌入
        embs_list = [all_embs]
        vae_losses = []
        
        # 图卷积传播
        emb_0 = all_embs
        for i in range(self.n_layers):
            all_embs = self.gcn(all_embs)  # 图卷积
            embs_list.append(all_embs)
            
            # 对当前层嵌入使用VAE
            recon, mu, logvar = self.vaes[i](all_embs)
            vae_loss = self.vae_loss(recon, all_embs, mu, logvar)
            vae_losses.append(vae_loss)
        
        # 最终嵌入使用VAE
        recon, mu, logvar = self.vaes[-1](all_embs)
        vae_loss = self.vae_loss(recon, all_embs, mu, logvar)
        vae_losses.append(vae_loss)
        
        # 计算层级注意力
        embs_tensor = torch.stack(embs_list, dim=1)  # [n_nodes, n_layers+1, embedding_dim]
        embs_flat = embs_tensor.reshape(embs_tensor.shape[0], -1)  # [n_nodes, (n_layers+1)*embedding_dim]
        
        # 注意力分数
        attention_logits = self.layer_attention(embs_flat)
        attention_scores = F.softmax(attention_logits, dim=1)  # [n_nodes, n_layers+1]
        
        # 根据注意力分数聚合各层嵌入
        attention_scores = attention_scores.unsqueeze(-1)  # [n_nodes, n_layers+1, 1]
        weighted_embs = embs_tensor * attention_scores
        final_embs = weighted_embs.sum(dim=1)  # [n_nodes, embedding_dim]
        
        # 分离用户和物品的嵌入
        user_final_embs = final_embs[:self.user_num]
        item_final_embs = final_embs[self.user_num:]
        
        # 如果输入了用户和物品索引，计算它们之间的评分
        if users is not None and items is not None:
            # 获取指定用户和物品的嵌入
            user_emb = user_final_embs[users.squeeze()]
            item_emb = item_final_embs[items.squeeze()]
            
            # 计算用户-物品评分预测（内积）
            scores = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
            
            return scores, torch.stack(vae_losses).mean()
        
        # 否则返回所有用户和物品的嵌入
        return user_final_embs, item_final_embs, torch.stack(vae_losses).mean()
    
    def vae_loss(self, recon_x, x, mu, logvar):
        # 重构误差
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def get_all_embeddings(self):
        """获取所有用户和物品的最终嵌入"""
        user_embs, item_embs, _ = self.forward()
        return user_embs, item_embs


class GraphEnhancedVAE(nn.Module):
    """完整的图增强VAE模型"""
    def __init__(self, user_num, item_num, adj_matrix, embedding_dim=64, n_layers=3):
        super(GraphEnhancedVAE, self).__init__()
        
        # LightGCN 核心
        self.lightgcn = LightGCN(user_num, item_num, adj_matrix, embedding_dim, n_layers)
        
        # 评分预测层 (MLP)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, users, items):
        # 通过LightGCN获取评分和VAE损失
        scores, vae_loss = self.lightgcn(users, items)
        
        # 缩放评分到20-100范围
        # 方法1：简单线性缩放
        ratings = 40.0 + scores * 30.0
        
        # 确保评分在20-100范围内
        ratings = torch.clamp(ratings, 20.0, 100.0)
        
        return ratings, vae_loss
        
    def predict(self, users, items):
        """预测用户对物品的评分"""
        with torch.no_grad():
            ratings, _ = self.forward(users, items)
        return ratings

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape) 