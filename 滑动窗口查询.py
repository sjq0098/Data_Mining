from collections import deque

class SlidingWindow:
    def __init__(self, size: int):
        self.size = size
        self.window = deque()
        self.sum = 0

    def add(self, value: float) -> None:
        """向滑动窗口中添加一个新值，并更新窗口内的和"""
        self.window.append(value)
        self.sum += value
        if len(self.window) > self.size:
            removed = self.window.popleft()
            self.sum -= removed

    def get_sum(self) -> float:
        """返回当前滑动窗口内的和"""
        return self.sum

    def get_average(self) -> float:
        """返回当前滑动窗口内的平均值"""
        if not self.window:
            return 0.0
        return self.sum / len(self.window)


# 示例：计算流数据 [1, 3, 5, 7, 9] 的滑动窗口大小为 3 的和与平均值
stream = [1, 3, 5, 7, 9]
window_size = 3
sw = SlidingWindow(window_size)

results = []
for x in stream:
    sw.add(x)
    results.append({
        "新加入元素": x,
        "当前窗口内容": list(sw.window),
        "窗口和": sw.get_sum(),
        "窗口平均值": round(sw.get_average(), 2)
    })

import pandas as pd
df = pd.DataFrame(results)
print(df)
