#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>  // for std::setprecision

// ----------------------------------------------
// Helper functions: random number, Sigmoid and its derivative
// ----------------------------------------------
static std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));

double uniform_rand(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

// ----------------------------------------------
// Simple matrix operations using std::vector<vector<double>>
// ----------------------------------------------
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Create a rows×cols zero matrix
Matrix zeros(int rows, int cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

// Create a rows×cols matrix with uniform random values in [minVal, maxVal)
Matrix random_matrix(int rows, int cols, double minVal, double maxVal) {
    Matrix M(rows, Vector(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i][j] = uniform_rand(minVal, maxVal);
        }
    }
    return M;
}

// Matrix multiplication: A(m×k) * B(k×n) = C(m×n)
Matrix mat_mul(const Matrix &A, const Matrix &B) {
    int m = A.size();
    int k = A[0].size();
    int n = B[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int t = 0; t < k; t++) {
                sum += A[i][t] * B[t][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

// Add a bias vector (length n) to each row of an m×n matrix
Matrix add_bias_rowwise(const Matrix &mat, const Vector &bias) {
    int m = mat.size();
    int n = mat[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = mat[i][j] + bias[j];
        }
    }
    return C;
}

// Transpose of a matrix A(m×n) -> A^T(n×m)
Matrix transpose(const Matrix &A) {
    int m = A.size();
    int n = A[0].size();
    Matrix T = zeros(n, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

// Apply sigmoid elementwise to matrix A
Matrix apply_sigmoid(const Matrix &A) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = sigmoid(A[i][j]);
        }
    }
    return C;
}

// Apply sigmoid derivative (param A already = sigmoid(Z)) elementwise
Matrix apply_sigmoid_derivative(const Matrix &A) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = sigmoid_derivative(A[i][j]);
        }
    }
    return C;
}

// Elementwise subtraction: A(m×n) - B(m×n)
Matrix mat_sub(const Matrix &A, const Matrix &B) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// Multiply matrix A by scalar alpha
Matrix mat_scalar_mul(const Matrix &A, double alpha) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            C[i][j] = alpha * A[i][j];
        }
    }
    return C;
}

// Hadamard (elementwise) product: A(m×n) ⊙ B(m×n)
Matrix hadamard(const Matrix &A, const Matrix &B) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            C[i][j] = A[i][j] * B[i][j];
        }
    }
    return C;
}

// Convert a length-m vector into an m×1 matrix
Matrix vector_to_matrix(const Vector &v) {
    int m = v.size();
    Matrix M = zeros(m, 1);
    for (int i = 0; i < m; i++){
        M[i][0] = v[i];
    }
    return M;
}

// Convert an m×1 matrix into a length-m vector
Vector matrix_to_vector(const Matrix &M) {
    int m = M.size();
    Vector v(m);
    for (int i = 0; i < m; i++){
        v[i] = M[i][0];
    }
    return v;
}

// Compute binary cross-entropy loss for Y(m×1), A2(m×1)
double compute_binary_cross_entropy(const Matrix &Y, const Matrix &A2) {
    int m = Y.size();
    double loss = 0.0;
    const double eps = 1e-8;
    for (int i = 0; i < m; i++){
        double y_true = Y[i][0];
        double y_pred = A2[i][0];
        loss += - (y_true * std::log(y_pred + eps) + (1.0 - y_true) * std::log(1.0 - y_pred + eps));
    }
    return loss / m;
}

// Threshold A(m×n) by 0.5 → element ≥0.5 becomes 1.0, else 0.0
Matrix threshold_05(const Matrix &A) {
    int m = A.size();
    int n = A[0].size();
    Matrix C = zeros(m, n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            C[i][j] = (A[i][j] >= 0.5 ? 1.0 : 0.0);
        }
    }
    return C;
}

// Compute accuracy between Y_true(m×1) and Y_pred(m×1)
double accuracy(const Matrix &Y_true, const Matrix &Y_pred) {
    int m = Y_true.size();
    int correct = 0;
    for (int i = 0; i < m; i++){
        if (std::fabs(Y_true[i][0] - Y_pred[i][0]) < 1e-9) {
            correct++;
        }
    }
    return static_cast<double>(correct) / m;
}

// ----------------------------------------------
// MLP class with one hidden layer
// ----------------------------------------------
class MLP {
public:
    // Constructor:
    //   in_size: input dimension (for XOR, 2)
    //   hidden_size: number of hidden neurons (e.g. 5)
    //   lr: learning rate
    //   epochs: number of training epochs
    MLP(int in_size, int hidden_size, double lr = 0.1, int epochs = 5000)
        : input_size(in_size), hidden_size(hidden_size),
          learning_rate(lr), epochs(epochs)
    {
        // Randomly initialize weights and biases
        W1 = random_matrix(input_size, hidden_size, -0.5, 0.5);  // (in_size × hidden_size)
        b1 = Vector(hidden_size, 0.0);                           // (hidden_size)

        W2 = random_matrix(hidden_size, 1, -0.5, 0.5);           // (hidden_size × 1)
        b2 = Vector(1, 0.0);                                     // (1)
    }

    // Forward pass: input X(m×in_size) → returns A2(m×1)
    Matrix forward(const Matrix &X) {
        int m = X.size();

        // Z1 = X·W1 + b1 → (m×hidden_size)
        Matrix Z1 = add_bias_rowwise(mat_mul(X, W1), b1);
        A1 = apply_sigmoid(Z1);  // (m×hidden_size)

        // Z2 = A1·W2 + b2 → (m×1)
        Matrix Z2 = add_bias_rowwise(mat_mul(A1, W2), b2);
        A2 = apply_sigmoid(Z2);  // (m×1)

        return A2;
    }

    // Backward pass & update parameters; X(m×in_size), Y(m×1)
    void backward(const Matrix &X, const Matrix &Y) {
        int m = X.size();

        // 1) Output-layer error: delta2 = A2 - Y  (m×1)
        Matrix delta2 = mat_sub(A2, Y);

        // 2) dW2 = (1/m) * A1^T · delta2  → (hidden_size×1)
        Matrix A1_T = transpose(A1);  // (hidden_size×m)
        Matrix dW2 = mat_scalar_mul(mat_mul(A1_T, delta2), 1.0 / m);

        // 3) db2 = (1/m) * sum(delta2)  → length 1
        double sum_delta2 = 0.0;
        for (int i = 0; i < m; i++){
            sum_delta2 += delta2[i][0];
        }
        double db2_scalar = sum_delta2 / m;
        Vector db2_vec(1, db2_scalar);

        // 4) Hidden-layer error: delta1 = (delta2·W2^T) ⊙ sigmoid'(A1)
        Matrix W2_T = transpose(W2);                      // (1×hidden_size)
        Matrix delta2_W2T = mat_mul(delta2, W2_T);         // (m×hidden_size)
        Matrix dA1 = apply_sigmoid_derivative(A1);         // (m×hidden_size)
        Matrix delta1 = hadamard(delta2_W2T, dA1);          // (m×hidden_size)

        // 5) dW1 = (1/m) * X^T · delta1  → (in_size×hidden_size)
        Matrix X_T = transpose(X);                         // (in_size×m)
        Matrix dW1 = mat_scalar_mul(mat_mul(X_T, delta1), 1.0 / m);

        // 6) db1 = (1/m) * sum(delta1, axis=0) → length hidden_size
        Vector db1_vec(hidden_size, 0.0);
        for (int j = 0; j < hidden_size; j++) {
            double sum_col = 0.0;
            for (int i = 0; i < m; i++){
                sum_col += delta1[i][j];
            }
            db1_vec[j] = sum_col / m;
        }

        // 7) Update weights and biases:
        //    W2 ← W2 - lr * dW2
        W2 = mat_sub(W2, mat_scalar_mul(dW2, learning_rate));
        //    b2[0] ← b2[0] - lr * db2
        b2[0] -= learning_rate * db2_vec[0];

        //    W1 ← W1 - lr * dW1
        W1 = mat_sub(W1, mat_scalar_mul(dW1, learning_rate));
        //    b1[j] ← b1[j] - lr * db1_vec[j]
        for (int j = 0; j < hidden_size; j++) {
            b1[j] -= learning_rate * db1_vec[j];
        }
    }

    // Train on X(m×in_size) with labels y (length m)
    void fit(const Matrix &X, const Vector &y) {
        int m = X.size();
        // Convert y vector to an m×1 matrix Y_mat
        Matrix Y_mat = zeros(m, 1);
        for (int i = 0; i < m; i++) {
            Y_mat[i][0] = y[i];
        }

        for (int epoch = 1; epoch <= epochs; epoch++) {
            // 1) forward pass
            Matrix A2_pred = forward(X);

            // 2) print loss every 10% or at epoch 1
            if (epoch == 1 || epoch % (epochs / 10) == 0) {
                double loss = compute_binary_cross_entropy(Y_mat, A2_pred);
                std::cout << "Epoch " << std::setw(5) << epoch
                          << " / " << epochs
                          << "   Loss = " << std::fixed << std::setprecision(6)
                          << loss << "\n";
            }

            // 3) backward & update
            backward(X, Y_mat);
        }
    }

    // Predict on X(m×in_size), return a binary matrix (m×1)
    Matrix predict(const Matrix &X) {
        Matrix A2_pred = forward(X);          // (m×1) of probabilities
        Matrix Y_pred = threshold_05(A2_pred);// threshold at 0.5
        return Y_pred;
    }

private:
    int input_size;
    int hidden_size;
    double learning_rate;
    int epochs;

    Matrix W1;  // (input_size × hidden_size)
    Vector b1;  // (hidden_size)
    Matrix W2;  // (hidden_size × 1)
    Vector b2;  // (1)

    // Saved during forward pass
    Matrix A1;  // (m×hidden_size)
    Matrix A2;  // (m×1)
};

// ----------------------------------------------
// main(): create XOR dataset, train & test MLP
// ----------------------------------------------
int main() {
    // 1. XOR dataset: 4 samples
    Matrix X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    Vector y = {0.0, 1.0, 1.0, 0.0};

    // 2. Initialize MLP
    int input_dim = 2;
    int hidden_neurons = 5;   // you can change this
    double lr = 0.5;          // learning rate
    int epochs = 10000;       // number of training epochs
    MLP model(input_dim, hidden_neurons, lr, epochs);

    // 3. Train
    std::cout << ">> Starting to train MLP on XOR dataset...\n";
    model.fit(X, y);
    std::cout << ">> Training completed.\n\n";

    // 4. Predict on the same XOR points
    Matrix Y_pred = model.predict(X);  // (4×1)
    std::cout << "Predictions on XOR inputs:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Input (" << X[i][0] << ", " << X[i][1] << ")"
                  << "  Label = " << y[i]
                  << "  Pred = " << Y_pred[i][0] << "\n";
    }

    // 5. Compute accuracy
    Matrix Y_true_mat = zeros(4, 1);
    for (int i = 0; i < 4; i++) {
        Y_true_mat[i][0] = y[i];
    }
    double acc = accuracy(Y_true_mat, Y_pred);
    std::cout << "\nAccuracy on XOR: " << std::fixed << std::setprecision(2)
              << (acc * 100.0) << "%\n";

    return 0;
}
