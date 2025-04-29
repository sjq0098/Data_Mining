def apriori(transactions, min_support):
    # 第一步：计算单个项的支持度
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            if item not in item_counts:
                item_counts[item] = 1
            else:
                item_counts[item] += 1

    # 第二步：筛选出频繁1项集
    num_transactions = len(transactions)
    frequent_itemsets = {item: count / num_transactions for item, count in item_counts.items() if count / num_transactions >= min_support}
    all_frequent_itemsets = [frequent_itemsets]
    frequent_itemsets_list = [set([item]) for item in frequent_itemsets.keys()]

    # 第三步：生成候选项集并计算支持度，直到找不到频繁项集为止
    k = 2
    while True:
        candidate_itemsets = {}
        # 根据前一轮的频繁项集生成候选项集
        for i in range(len(frequent_itemsets_list)):
            for j in range(i + 1, len(frequent_itemsets_list)):
                union_set = frequent_itemsets_list[i].union(frequent_itemsets_list[j])
                if len(union_set) == k:
                    itemset = tuple(sorted(union_set))
                    if itemset not in candidate_itemsets:
                        candidate_itemsets[itemset] = 0
        for transaction in transactions:
            for itemset in candidate_itemsets.keys():
                if set(itemset).issubset(set(transaction)):
                    candidate_itemsets[itemset] += 1
        frequent_itemsets = {itemset: count / num_transactions for itemset, count in candidate_itemsets.items() if count / num_transactions >= min_support}

        if not frequent_itemsets:
            break

        all_frequent_itemsets.append(frequent_itemsets)
        frequent_itemsets_list = [set(itemset) for itemset in frequent_itemsets.keys()]
        k += 1

    return all_frequent_itemsets

# 测试数据
transactions = [['milk', 'beer', 'diaper'],
                ['milk', 'diaper', 'coke'],
                ['beer', 'diaper', 'coke']]
min_support = 0.5

# 运行Apriori算法
result = apriori(transactions, min_support)
print("Frequent Itemsets:", result)