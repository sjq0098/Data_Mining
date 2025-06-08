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
                # 每行格式为 "<item id> <score>"
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

def main():
    train_path = 'data/train.txt'
    df = load_train(train_path)
    
    print("数据集统计:")
    total_ratings = len(df)
    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    print("总评分数:", total_ratings)
    print("用户数:", num_users)
    print("物品数:", num_items)
    print("每个用户平均评分数: {:.2f}".format(total_ratings / num_users))
    print("每个物品平均被评分数: {:.2f}".format(total_ratings / num_items))
    
    print("\n评分分布:")
    print(df['score'].describe())

if __name__ == '__main__':
    main()