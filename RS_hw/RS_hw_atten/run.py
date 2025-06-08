import argparse
import os
from train import train, predict


def main():
    parser = argparse.ArgumentParser(description='图增强VAE推荐系统')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_path', type=str, default='./data', help='数据目录')
    train_parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    train_parser.add_argument('--n_layers', type=int, default=3, help='GCN层数')
    train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    train_parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    train_parser.add_argument('--vae_weight', type=float, default=0.1, help='VAE损失权重')
    train_parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    train_parser.add_argument('--patience', type=int, default=10, help='早停轮数')
    train_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    train_parser.add_argument('--save_path', type=str, default='./best_model.pt', help='模型保存路径')
    train_parser.add_argument('--result_path', type=str, default='./result.txt', help='结果保存路径')
    train_parser.add_argument('--predict', action='store_true', help='训练后进行预测')
    
    # 预测模式
    predict_parser = subparsers.add_parser('predict', help='预测评分')
    predict_parser.add_argument('--data_path', type=str, default='./data', help='数据目录')
    predict_parser.add_argument('--model_path', type=str, default='./best_model.pt', help='模型路径')
    predict_parser.add_argument('--result_path', type=str, default='./result.txt', help='结果保存路径')
    predict_parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    predict_parser.add_argument('--n_layers', type=int, default=3, help='GCN层数')
    predict_parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    predict_parser.add_argument('--batch_size', type=int, default=1024, help='批大小')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        model, processor = train(args)
        
        # 如果需要，进行预测
        if args.predict:
            predict(model, processor, args)
    
    elif args.mode == 'predict':
        # 预测模式
        from predict import main as predict_main
        predict_main(args)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 