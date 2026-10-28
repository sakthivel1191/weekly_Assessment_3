import numpy as np
import matplotlib.pyplot as plt
import torch

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    # Clip values to avoid overflow in exp()
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # Subtract the maximum for numerical stability
    x = np.array(x, dtype=float)
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
def initialize_weights(seed=7):
    np.random.seed(seed)
    W1 = np.random.randn(2, 4) * 0.7
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.7
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

def forward_pass(X, W1, b1, W2, b2):
    hidden_linear = X @ W1 + b1
    hidden_output = relu(hidden_linear)
    output_linear = hidden_output @ W2 + b2
    output = sigmoid(output_linear)
    return hidden_linear, hidden_output, output

def train_xor(learning_rate=0.5, epochs=5000, seed=7):
    W1, b1, W2, b2 = initialize_weights(seed)
    losses = []
    for epoch in range(epochs):
        z1, a1, predictions = forward_pass(
            X, W1, b1, W2, b2
        )
        loss = cross_entropy(y, predictions)
        losses.append(loss)
        dz2 = (predictions - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    _, _, predictions = forward_pass(X, W1, b1, W2, b2)
    return predictions, losses, (W1, b1, W2, b2)

print("ReLU:", relu(np.array([-2, 0, 3])))
print("Sigmoid:", sigmoid(np.array([0.0])))
print("Softmax:", np.round(softmax([2.0, 1.0, 0.1]), 3))
print("MSE:", mse(np.array([1, 0]), np.array([0.8, 0.2])))

predictions, losses, weights = train_xor(
    learning_rate=0.5,
    epochs=5000
)
print("\nXOR predictions:")
print(np.round(predictions.flatten(), 3))
print("\nPredicted classes:")
print((predictions.flatten() >= 0.5).astype(int))
print("\nFinal loss:", round(losses[-1], 5))

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("XOR Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/task1_loss.png")
plt.show()
plt.figure(figsize=(8, 4))

for lr in [0.01, 0.5, 5.0]:
    pred, lr_losses, _ = train_xor(
        learning_rate=lr,
        epochs=5000
    )
    lr_losses = np.array(lr_losses)
    lr_losses[~np.isfinite(lr_losses)] = np.nan
    plt.plot(lr_losses, label=f"Learning rate = {lr}")

plt.title("Effect of Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.ylim(0, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/learning_rates.png")
plt.show()

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor(2.0)
target = torch.tensor(10.0)
loss = (w * x + b - target) ** 2
loss.backward()
print("\nPyTorch autograd:")
print("w.grad =", w.grad.item())
print("b.grad =", b.grad.item())

