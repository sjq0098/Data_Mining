import numpy as np
import matplotlib.pyplot as plt

#支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Define activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Define a Multi-Layer Perceptron (MLP) class
class MLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=1000):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, Y, A2):
        # Binary cross-entropy loss
        m = Y.shape[0]
        loss = -(1/m) * np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
        return loss

    def backward(self, X, Y):
        m = X.shape[0]
        # Compute gradients
        dZ2 = self.A2 - Y.reshape(-1, 1)                  # (m, 1)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)              # (hidden_size, 1)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # (1, 1)
        
        dA1 = np.dot(dZ2, self.W2.T)                      # (m, hidden_size)
        dZ1 = dA1 * sigmoid_derivative(self.A1)           # (m, hidden_size)
        dW1 = (1/m) * np.dot(X.T, dZ1)                    # (input_size, hidden_size)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # (1, hidden_size)

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y):
        # Training loop
        for epoch in range(self.epochs):
            A2 = self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        # Predict labels
        A2 = self.forward(X)
        return np.where(A2 >= 0.5, 1, 0).flatten()

# Generate a simple linearly separable dataset (same as before)
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

# Initialize and train the MLP
mlp = MLP(input_size=2, hidden_size=5, learning_rate=0.1, epochs=10000)
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"测试集准确率: {accuracy * 100:.2f}%")

# Plot the data and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="类 0", marker='o')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="类 1", marker='x')

# Create a meshgrid to plot decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
preds_grid = mlp.predict(grid).reshape(xx.shape)

# Plot decision boundary contour
plt.contourf(xx, yy, preds_grid, alpha=0.2, levels=[-0.1, 0.5, 1.1], colors=['blue', 'orange'])
plt.legend()
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title("MLP 决策边界及测试集样本")
plt.grid(True)
plt.show()
