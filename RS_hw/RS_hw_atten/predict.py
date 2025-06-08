import os
import torch
import argparse
from tqdm import tqdm

from model import GraphEnhancedVAE
from data_processor import DataProcessor
from train import predict


def main(args):
    # 准备数据
    processor = DataProcessor(
        train_path=os.path.join(args.data_path, 'train.txt'),
        test_path=os.path.join(args.data_path, 'test.txt')
    )
    
    _, _, _, adj_matrix = processor.process_data(
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
    
    # 加载模型
    model.load_state_dict(torch.load(args.model_path))
    
    # 进行预测
    predict(model, processor, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预测推荐系统模型')
    parser.add_argument('--data_path', type=str, default='./data', help='数据目录')
    parser.add_argument('--model_path', type=str, default='./best_model.pt', help='模型路径')
    parser.add_argument('--result_path', type=str, default='./result.txt', help='结果保存路径')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='GCN层数')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    
    args = parser.parse_args()
    
    main(args) 