import pandas as pd
import numpy as np

def load_train(path):
    user_ids, item_ids, scores = [], [], []
    current_user = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                # 用户行，格式为 "<user id>|<评分项数>"
                current_user = int(line.split('|')[0])
            else:
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    item_id = int(parts[0])
                    score = float(parts[1])
                    user_ids.append(current_user)
                    item_ids.append(item_id)
                    scores.append(score)
    return pd.DataFrame({'user': user_ids, 'item': item_ids, 'score': scores})

def load_test(path):
    # 测试集格式与训练集类似：用户行后跟该用户所有的评分记录（这里只取用户和物品信息）
    user_ids, item_ids = [], []
    current_user = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                current_user = int(line.split('|')[0])
            else:
                if not line:
                    continue
                parts = line.split()
                if parts:
                    item_id = int(parts[0])
                    user_ids.append(current_user)
                    item_ids.append(item_id)
    return pd.DataFrame({'user': user_ids, 'item': item_ids})

def main():
    train_path = 'data/train.txt'
    test_path = 'data/test.txt'
    
    train_df = load_train(train_path)
    test_df = load_test(test_path)
    
    # 用户统计
    train_users = set(train_df['user'].unique())
    test_users = set(test_df['user'].unique())
    
    unseen_users = test_users - train_users
    print("测试集中在训练集中完全未出现的用户数:", len(unseen_users))
    
    train_user_counts = train_df['user'].value_counts()
    sparse_threshold = 5  # 阈值可以根据具体情况调整
    sparse_users = set(train_user_counts[train_user_counts < sparse_threshold].index)
    sparse_test_users = sparse_users & test_users
    print("测试集中训练数据中评分数低于 {} 的用户数:".format(sparse_threshold), len(sparse_test_users))
    
    total_test_users = len(test_users)
    print("测试集总用户数:", total_test_users)
    print("完全未出现用户占比: {:.2f}%".format(len(unseen_users) / total_test_users * 100))
    print("稀疏用户占比: {:.2f}%".format(len(sparse_test_users) / total_test_users * 100))
    
    # 物品统计
    train_items = set(train_df['item'].unique())
    test_items = set(test_df['item'].unique())
    
    unseen_items = test_items - train_items
    print("\n测试集中在训练集中完全未出现的物品数:", len(unseen_items))
    
    train_item_counts = train_df['item'].value_counts()
    sparse_items = set(train_item_counts[train_item_counts < sparse_threshold].index)
    sparse_test_items = sparse_items & test_items
    print("测试集中训练数据中被评分数低于 {} 的物品数:".format(sparse_threshold), len(sparse_test_items))
    
    total_test_items = len(test_items)
    print("测试集总物品数:", total_test_items)
    print("完全未出现物品占比: {:.2f}%".format(len(unseen_items) / total_test_items * 100))
    print("稀疏物品占比: {:.2f}%".format(len(sparse_test_items) / total_test_items * 100))

if __name__ == '__main__':
    main()