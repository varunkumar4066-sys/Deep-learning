"""Experiment 17: Perceptron and Forward Propagation.
Tasks:
1. Implement weighted sum, bias and step activation in NumPy.
2. Train the perceptron on loan data and record weight updates per epoch.
3. Plot the learned decision boundary.
4. Train on XOR data and show that it fails to converge.
5. Compare step, sigmoid and ReLU outputs for the same inputs.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 17: Perceptron & Forward Propagation ===")

# 1 & 2. Linearly separable loan data (Credit Score [0-1], Debt Ratio [0-1])
X_loan = np.array([
    [0.8, 0.2], [0.9, 0.1], [0.7, 0.3], [0.85, 0.25],
    [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.25, 0.85]
])
y_loan = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Perceptron learning rule
w = np.zeros(2)
b = 0.0
lr = 0.1
epochs = 20
updates_hist = []

for ep in range(epochs):
    updates = 0
    for xi, yi in zip(X_loan, y_loan):
        y_pred = 1 if (np.dot(w, xi) + b) >= 0 else 0
        err = yi - y_pred
        if err != 0:
            w += lr * err * xi
            b += lr * err
            updates += 1
    updates_hist.append(updates)

print(f"Loan Data Converged in {epochs} epochs! Final Weights: {np.round(w, 3)}, Bias: {b:.3f}")

# 4. XOR data demonstration
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
w_xor, b_xor = np.zeros(2), 0.0
xor_updates = []
for _ in range(30):
    u = 0
    for xi, yi in zip(X_xor, y_xor):
        y_pred = 1 if (np.dot(w_xor, xi) + b_xor) >= 0 else 0
        err = yi - y_pred
        if err != 0:
            w_xor += lr * err * xi
            b_xor += lr * err
            u += 1
    xor_updates.append(u)
print(f"XOR Updates in last 5 epochs: {xor_updates[-5:]} -> Fails to converge due to non-linearity!")

# 5. Compare activations
z = np.linspace(-5, 5, 200)
step_out = np.where(z >= 0, 1.0, 0.0)
sig_out = 1.0 / (1.0 + np.exp(-z))
relu_out = np.maximum(0, z)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Decision boundary
axes[0].scatter(X_loan[y_loan==1, 0], X_loan[y_loan==1, 1], color='blue', label='Approved (1)')
axes[0].scatter(X_loan[y_loan==0, 0], X_loan[y_loan==0, 1], color='red', label='Rejected (0)')
x_db = np.linspace(0, 1, 50)
y_db = -(w[0] * x_db + b) / (w[1] + 1e-9)
axes[0].plot(x_db, y_db, 'g--', label='Perceptron Decision Boundary')
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Credit Score')
axes[0].set_ylabel('Debt Ratio')
axes[0].set_title('Loan Data Linear Separation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Activations
axes[1].plot(z, step_out, 'k--', label='Step')
axes[1].plot(z, sig_out, 'b-', label='Sigmoid')
axes[1].plot(z, relu_out, 'r-', label='ReLU')
axes[1].set_xlabel('Input z')
axes[1].set_ylabel('Activation Output')
axes[1].set_title('Activation Functions Comparison')
axes[1].set_ylim(-0.5, 3)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp17_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
