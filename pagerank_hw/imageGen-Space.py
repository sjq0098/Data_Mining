import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm

# 读取数据
with open("pagerank_imageGen/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 按内存使用大小降序排列
sorted_data = sorted(data, key=lambda x: x["average_memory_usage_MB"], reverse=True)

# 提取数据
filenames = [entry["filename"] for entry in sorted_data]
memory_usage = [entry["average_memory_usage_MB"] for entry in sorted_data]
runtime = [entry["average_runtime_s"] for entry in sorted_data]

# 提取优化方向
optimization_directions = [entry["optimization_direction"] for entry in sorted_data]

# 计算颜色映射
norm = LogNorm(vmin=min(runtime), vmax=max(runtime))
colors = plt.get_cmap("viridis")(norm(runtime))

# 创建柱状图
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(optimization_directions, memory_usage, color=colors)

# 在每个柱状图的右端标注其占用空间大小
for i, v in enumerate(memory_usage):
    ax.text(v, i, f"{v:.2f} MB", va="center", ha="left")

# 添加标签和标题
ax.set_xlabel("Average Memory Usage (MB)")
ax.set_title("PageRank Implementations by Memory Usage")
ax.invert_yaxis()  # 反转Y轴以使最大值在顶部
ax.grid(axis="x", linestyle="--", linewidth=0.5)

# 设置对数坐标
ax.set_xscale("log")

plt.xlim(4, 1500)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Average Runtime (s)")

# 保存图形
plt.savefig("pagerank_memory_usage_bar_chart_with_runtime.png", bbox_inches="tight")
