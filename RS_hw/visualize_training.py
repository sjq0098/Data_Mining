import matplotlib.pyplot as plt

# 这里用上面输出的训练日志数据
epochs = list(range(1, 21))
train_loss = [
    4374.3474, 604.9691, 350.8752, 317.3099, 299.7588,
    290.8417, 284.8022, 280.8689, 278.6053, 276.3088,
    275.1372, 274.2147, 273.3200, 272.4768, 272.1731,
    271.8004, 270.5879, 270.7347, 270.3051, 270.2447
]
rmse = [
    66.1388, 24.5961, 18.7317, 17.8132, 17.3135,
    17.0541, 16.8761, 16.7591, 16.6915, 16.6225,
    16.5873, 16.5594, 16.5324, 16.5069, 16.4977,
    16.4864, 16.4496, 16.4540, 16.4410, 16.4391
]

plt.figure(figsize=(10, 4))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss over Epochs')

# 绘制 RMSE 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, rmse, marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE over Epochs')

plt.tight_layout()
plt.show()