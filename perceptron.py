import numpy as np
import matplotlib.pyplot as plt

#支持中文图标
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# Define a Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # +1 for bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, X):
        """Predict the class label for input samples X."""
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(linear_output >= 0, 1, 0)

    def fit(self, X, y):
        """Train the perceptron using the training data X and labels y."""
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                update = self.learning_rate * (target - output)
                self.weights[1:] += update * xi
                self.weights[0] += update  # bias update

# Generate a simple linearly separable dataset
np.random.seed(42)
num_samples = 100

# Positive class
X_pos = np.random.randn(num_samples // 2, 2) + np.array([2, 2])
y_pos = np.ones(num_samples // 2)

# Negative class
X_neg = np.random.randn(num_samples // 2, 2) + np.array([-2, -2])
y_neg = np.zeros(num_samples // 2)

# Combine data
X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

# Shuffle the data
indices = np.random.permutation(num_samples)
X, y = X[indices], y[indices]

# Train-test split
split_index = int(0.8 * num_samples)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize and train the Perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.fit(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"valid auc: {accuracy * 100:.2f}%")

# Plot the data and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="类 0", marker='o')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="类 1", marker='x')

# Decision boundary: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
w1, w2 = perceptron.weights[1], perceptron.weights[2]
b = perceptron.weights[0]
x_vals = np.array([X_test[:, 0].min() - 1, X_test[:, 0].max() + 1])
y_vals = -(w1 * x_vals + b) / w2

plt.plot(x_vals, y_vals, 'k--', label="决策边界")
plt.legend()
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title("感知机决策边界及测试集样本")
plt.grid(True)
plt.show()
