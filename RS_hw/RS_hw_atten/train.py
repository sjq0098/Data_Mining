import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import argparse
import math

from model import GraphEnhancedVAE
from data_processor import DataProcessor


def train(args):
    # 设置随机种子保证可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 准备数据
    processor = DataProcessor(
        train_path=os.path.join(args.data_path, 'train.txt'),
        test_path=os.path.join(args.data_path, 'test.txt')
    )
    
    train_loader, valid_loader, test_loader, adj_matrix = processor.process_data(
        train_ratio=args.train_ratio,
        batch_size=args.batch_size
    )
    
    # 创建模型
    model = GraphEnhancedVAE(
        user_num=processor.n_users,
        item_num=processor.n_items,
        adj_matrix=adj_matrix,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers
    )
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 训练模型
    best_rmse = float('inf')
    best_epoch = 0
    early_stop_count = 0
    
    print("开始训练...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_vae_loss = 0.0
        train_rmse = 0.0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc=f"训练 Epoch {epoch}"):
            users, items, ratings = batch
            
            # 前向传播
            pred_ratings, vae_loss = model(users, items)
            
            # 确保维度匹配
            if pred_ratings.dim() != ratings.dim():
                if pred_ratings.dim() > ratings.dim():
                    pred_ratings = pred_ratings.squeeze()
                else:
                    ratings = ratings.reshape(pred_ratings.shape)
                    
            # 计算损失
            mse_loss = criterion(pred_ratings, ratings)
            loss = mse_loss + args.vae_weight * vae_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += mse_loss.item() * len(ratings)
            train_vae_loss += vae_loss.item() * len(ratings)
            train_rmse += torch.sum((pred_ratings - ratings) ** 2).item()
            train_samples += len(ratings)
        
        train_loss /= train_samples
        train_vae_loss /= train_samples
        train_rmse = math.sqrt(train_rmse / train_samples)
        
        # 验证
        model.eval()
        valid_loss = 0.0
        valid_rmse = 0.0
        valid_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"验证 Epoch {epoch}"):
                users, items, ratings = batch
                
                # 前向传播
                pred_ratings, _ = model(users, items)
                
                # 确保维度匹配
                if pred_ratings.dim() != ratings.dim():
                    if pred_ratings.dim() > ratings.dim():
                        pred_ratings = pred_ratings.squeeze()
                    else:
                        ratings = ratings.reshape(pred_ratings.shape)
                
                # 计算损失
                mse_loss = criterion(pred_ratings, ratings)
                
                # 累计损失
                valid_loss += mse_loss.item() * len(ratings)
                valid_rmse += torch.sum((pred_ratings - ratings) ** 2).item()
                valid_samples += len(ratings)
        
        valid_loss /= valid_samples
        valid_rmse = math.sqrt(valid_rmse / valid_samples)
        
        # 打印信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} - {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - VAE Loss: {train_vae_loss:.4f} - RMSE: {train_rmse:.4f} - "
              f"Valid Loss: {valid_loss:.4f} - RMSE: {valid_rmse:.4f}")
        
        # 保存最佳模型
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_epoch = epoch
            early_stop_count = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"已保存最佳模型 (RMSE: {best_rmse:.4f})")
        else:
            early_stop_count += 1
            if early_stop_count >= args.patience:
                print(f"早停 - {args.patience} 轮未改进")
                break
    
    print(f"训练完成! 最佳验证 RMSE: {best_rmse:.4f} at epoch {best_epoch}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(args.save_path))
    
    return model, processor


def predict(model, processor, args):
    print("开始预测测试集...")
    model.eval()
    
    # 获取测试数据
    test_loader = processor.test_data
    test_users, test_items = processor.get_test_reference()
    
    # 预测评分
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测测试集"):
            users, items = batch
            
            # 前向传播
            pred_ratings = model.predict(users, items)
            
            # 将预测结果转换为Python标量
            batch_predictions = pred_ratings.cpu().numpy().tolist()
            all_predictions.extend(batch_predictions)
    
    # 确保预测值在评分范围内 (20-100)并取整
    all_predictions = [min(max(int(round(pred)), 20), 100) for pred in all_predictions]
    
    # 生成结果文件
    with open(args.result_path, 'w') as f:
        current_user = None
        user_predictions = []
        
        for i, (user, item, pred) in enumerate(zip(test_users, test_items, all_predictions)):
            if current_user is None:
                current_user = user
                user_predictions = [(item, pred)]
            elif user != current_user:
                # 写入上一个用户的结果
                f.write(f"{current_user}|{len(user_predictions)}\n")
                for item_id, score in user_predictions:
                    f.write(f"{item_id}  {score}  \n")
                
                # 开始新用户
                current_user = user
                user_predictions = [(item, pred)]
            else:
                user_predictions.append((item, pred))
        
        # 写入最后一个用户的结果
        if current_user is not None:
            f.write(f"{current_user}|{len(user_predictions)}\n")
            for item_id, score in user_predictions:
                f.write(f"{item_id}  {score}  \n")
    
    print(f"结果已保存到 {args.result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练推荐系统模型')
    parser.add_argument('--data_path', type=str, default='./data', help='数据目录')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='GCN层数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--vae_weight', type=float, default=0.1, help='VAE损失权重')
    parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--patience', type=int, default=10, help='早停轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_path', type=str, default='./best_model.pt', help='模型保存路径')
    parser.add_argument('--result_path', type=str, default='./result.txt', help='结果保存路径')
    
    args = parser.parse_args()
    
    # 训练模型
    model, processor = train(args)
    
    # 预测测试集
    predict(model, processor, args) 