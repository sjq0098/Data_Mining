import json
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
with open("pagerank_imageGen/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取数据
filenames = [entry["filename"] for entry in data]
memory_usage = [entry["average_memory_usage_MB"] for entry in data]
runtime = [entry["average_runtime_s"] for entry in data]
optimization_directions = [entry["optimization_direction"] for entry in data]

# 创建散点图
plt.figure(figsize=(10, 6))

# 使用不同的颜色和标记表示不同的优化方向
unique_directions = list(set(optimization_directions))
markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(unique_directions)))

for i, direction in enumerate(unique_directions):
    x = [
        runtime[j]
        for j in range(len(runtime))
        if optimization_directions[j] == direction
    ]
    y = [
        memory_usage[j]
        for j in range(len(memory_usage))
        if optimization_directions[j] == direction
    ]
    plt.scatter(
        x, y, label=direction, marker=markers[i % len(markers)], color=colors[i]
    )

# 设置对数坐标
plt.xscale("log")
plt.yscale("log")

# 添加标签和标题
plt.xlabel("Average Runtime (s)")
plt.ylabel("Average Memory Usage (MB)")
plt.title("PageRank Optimization Analysis")
plt.legend(title="Optimization Direction")
plt.grid(True, which="both", ls="--", linewidth=0.5)

# plt.show()
plt.savefig("pagerank_optimization_analysis.png")
