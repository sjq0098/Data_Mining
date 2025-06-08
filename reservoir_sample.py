import random

def reservoir_sample(stream, k):
    """
    对可迭代对象 stream 做大小为 k 的蓄水池抽样，返回列表 R。
    """
    # 初始化：直接取前 k 个元素
    R = []
    for i, x in enumerate(stream):
        if i < k:
            R.append(x)
        else:
            # 生成 [0, i] 范围内的随机整数
            j = random.randint(0, i)
            # 以 k/(i+1) 的概率（即 j<k）替换
            if j < k:
                R[j] = x
    return R

# 示例：从 1~100 的整数流中随机抽取 5 个
if __name__ == "__main__":
    stream = range(1, 101)
    k = 5
    sample = reservoir_sample(stream, k)
    print(f"随机抽取的 {k} 个样本：{sample}")
