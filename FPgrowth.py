class FPNode:
    def __init__(self, item, count, parent):

        self.item = item

        self.count = count

        self.parent = parent

        self.children = {}



def build_fp_tree(transactions, min_support):

    # 计算项的频繁度

    item_counts = {}

    for transaction in transactions:

        for item in transaction:

            if item not in item_counts:

                item_counts[item] = 1

            else:

                item_counts[item] += 1



    # 筛选频繁项集

    item_counts = {item: count for item, count in item_counts.items() if count / len(transactions) >= min_support}

    frequent_items = set(item_counts.keys())



    # 构建FP树

    root = FPNode(None, 1, None)

    for transaction in transactions:

        filtered_transaction = [item for item in transaction if item in frequent_items]

        filtered_transaction.sort(key=lambda x: item_counts[x], reverse=True)

        current_node = root

        for item in filtered_transaction:

            if item not in current_node.children:

                current_node.children[item] = FPNode(item, 1, current_node)

            else:

                current_node.children[item].count += 1

            current_node = current_node.children[item]



    return root



# 测试数据

transactions = [['milk', 'beer', 'diaper'],

                ['milk', 'diaper', 'coke'],

                ['beer', 'diaper', 'coke']]

min_support = 0.5



# 构建FP树

fp_tree = build_fp_tree(transactions, min_support)



# 打印FP树结构

def print_fp_tree(node, level=0):

    if node is not None:

        print('  ' * level + f"{node.item}: {node.count}")

        for child in node.children.values():

            print_fp_tree(child, level + 1)



print_fp_tree(fp_tree)
