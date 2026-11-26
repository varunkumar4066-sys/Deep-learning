"""Experiment 06: Gradient Descent, SGD, Momentum and Adam Comparison.
Tasks:
1. Implement batch GD, SGD and Momentum from scratch on a 2-D loss surface.
2. Plot each optimiser’s path on a 2-D loss contour.
3. Train regression model with SGD, Momentum and Adam.
4. Plot loss against epochs for all optimisers.
5. Vary learning rate and note stability/convergence.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# Loss function f(w1, w2) = 0.5 * w1^2 + 2.0 * w2^2
def loss(w):
    return 0.5 * w[0]**2 + 2.0 * w[1]**2

def grad(w):
    return np.array([w[0], 4.0 * w[1]])

# Scratch optimizers
steps = 40
w_init = np.array([-4.0, 3.0])

# 1. Batch GD
w_gd = [w_init.copy()]
w = w_init.copy()
lr = 0.2
for _ in range(steps):
    w = w - lr * grad(w)
    w_gd.append(w.copy())

# 2. Momentum
w_mom = [w_init.copy()]
w = w_init.copy()
v = np.zeros(2)
beta = 0.8
for _ in range(steps):
    v = beta * v + lr * grad(w)
    w = w - v
    w_mom.append(w.copy())

# 3. Adam
w_adam = [w_init.copy()]
w = w_init.copy()
m = np.zeros(2)
v_sq = np.zeros(2)
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr_adam = 0.4
for t in range(1, steps + 1):
    g = grad(w)
    m = beta1 * m + (1 - beta1) * g
    v_sq = beta2 * v_sq + (1 - beta2) * (g**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v_sq / (1 - beta2**t)
    w = w - lr_adam * m_hat / (np.sqrt(v_hat) + eps)
    w_adam.append(w.copy())

print("=== Experiment 06: Optimisers Comparison ===")
print(f"Final GD Loss: {loss(w_gd[-1]):.6f} at {np.round(w_gd[-1], 4)}")
print(f"Final Momentum Loss: {loss(w_mom[-1]):.6f} at {np.round(w_mom[-1], 4)}")
print(f"Final Adam Loss: {loss(w_adam[-1]):.6f} at {np.round(w_adam[-1], 4)}")

# Plot contours and trajectories
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

w1_range = np.linspace(-5, 5, 100)
w2_range = np.linspace(-4, 4, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = 0.5 * W1**2 + 2.0 * W2**2

axes[0].contour(W1, W2, Z, levels=25, cmap='copper', alpha=0.6)
w_gd_arr = np.array(w_gd)
w_mom_arr = np.array(w_mom)
w_adam_arr = np.array(w_adam)

axes[0].plot(w_gd_arr[:, 0], w_gd_arr[:, 1], 'ro-', label='Standard GD', ms=3)
axes[0].plot(w_mom_arr[:, 0], w_mom_arr[:, 1], 'go-', label='Momentum', ms=3)
axes[0].plot(w_adam_arr[:, 0], w_adam_arr[:, 1], 'bo-', label='Adam', ms=3)
axes[0].set_title('Optimiser Paths on 2D Contour')
axes[0].set_xlabel('w1')
axes[0].set_ylabel('w2')
axes[0].legend()

# Loss vs Epochs
axes[1].plot([loss(w) for w in w_gd], 'r-', label='GD')
axes[1].plot([loss(w) for w in w_mom], 'g-', label='Momentum')
axes[1].plot([loss(w) for w in w_adam], 'b-', label='Adam')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss Value')
axes[1].set_yscale('log')
axes[1].set_title('Convergence Speed (Log Loss)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp06_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
