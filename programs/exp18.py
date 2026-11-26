"""Experiment 18: Two-Layer Neural Network with Backpropagation.
Tasks:
1. Initialise weights for an input–hidden–output network in NumPy.
2. Write the forward pass with sigmoid activations.
3. Derive and code the backward pass using chain rule.
4. Train with gradient descent and plot loss against epochs.
5. Plot non-linear decision boundary; verify gradients numerically for one weight.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

np.random.seed(42)

print("=== Experiment 18: Two-Layer NN with Backpropagation ===")

# Generate two-moons dataset
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
y = y.reshape(-1, 1)

# Architecture: 2 inputs -> 4 hidden units -> 1 output
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

# Training loop
lr = 0.5
epochs = 800
losses = []

for ep in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    # BCE loss
    loss = -np.mean(y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))
    losses.append(loss)
    
    # Backward pass
    dz2 = (a2 - y) / len(X)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_deriv(a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Gradient descent update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(f"Trained for {epochs} epochs. Final Loss: {losses[-1]:.4f}")

# Numerical gradient check for W1[0, 0]
eps = 1e-5
W1_plus = W1.copy()
W1_plus[0, 0] += eps
a2_plus = sigmoid(sigmoid(X @ W1_plus + b1) @ W2 + b2)
loss_plus = -np.mean(y * np.log(a2_plus + 1e-9) + (1 - y) * np.log(1 - a2_plus + 1e-9))

W1_minus = W1.copy()
W1_minus[0, 0] -= eps
a2_minus = sigmoid(sigmoid(X @ W1_minus + b1) @ W2 + b2)
loss_minus = -np.mean(y * np.log(a2_minus + 1e-9) + (1 - y) * np.log(1 - a2_minus + 1e-9))

num_grad = (loss_plus - loss_minus) / (2 * eps)
print(f"Gradient Check W1[0,0] -> Analytical: {dW1[0, 0]:.6f} | Numerical: {num_grad:.6f} | Diff: {abs(dW1[0,0]-num_grad):.2e}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(losses, 'b-', lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].set_title('Training Loss Curve')
axes[0].grid(True, alpha=0.3)

# 2D decision boundary mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_pts = np.c_[xx.ravel(), yy.ravel()]
grid_pred = sigmoid(sigmoid(grid_pts @ W1 + b1) @ W2 + b2).reshape(xx.shape)

axes[1].contourf(xx, yy, grid_pred, levels=20, cmap='RdBu', alpha=0.6)
axes[1].scatter(X[y[:, 0]==1, 0], X[y[:, 0]==1, 1], color='blue', edgecolors='k', label='Class 1')
axes[1].scatter(X[y[:, 0]==0, 0], X[y[:, 0]==0, 1], color='red', edgecolors='k', label='Class 0')
axes[1].set_title('Learned Non-Linear Decision Boundary')
axes[1].legend()

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp18_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
