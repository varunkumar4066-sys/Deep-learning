"""Experiment 15: Numerical Differentiation and Gradient Verification.
Tasks:
1. Define a differentiable multivariable loss function.
2. Derive its analytical partial derivatives.
3. Estimate gradients using forward and central finite differences.
4. Compare analytical and numerical gradients over multiple step sizes.
5. Plot gradient error and identify a suitable finite-difference step size.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 15: Numerical Gradient Verification ===")

# 1. Multivariable function: f(w1, w2) = w1^2 * w2 + sin(w1) * exp(w2)
def f(w):
    return w[0]**2 * w[1] + np.sin(w[0]) * np.exp(w[1])

# 2. Analytical gradient
def grad_analytical(w):
    df_dw1 = 2 * w[0] * w[1] + np.cos(w[0]) * np.exp(w[1])
    df_dw2 = w[0]**2 + np.sin(w[0]) * np.exp(w[1])
    return np.array([df_dw1, df_dw2])

# Point of evaluation
w_eval = np.array([1.5, 0.8])
true_grad = grad_analytical(w_eval)
print(f"Evaluation Point: {w_eval}")
print(f"Analytical Gradient: {true_grad}")

# 3 & 4. Finite differences across step sizes h
h_values = np.logspace(-1, -10, 20)
forward_errors = []
central_errors = []

for h in h_values:
    # Forward difference: [f(w + h*e_i) - f(w)] / h
    grad_fwd = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        grad_fwd[i] = (f(w_eval + e) - f(w_eval)) / h
    
    # Central difference: [f(w + h*e_i) - f(w - h*e_i)] / (2h)
    grad_cen = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        grad_cen[i] = (f(w_eval + e) - f(w_eval - e)) / (2 * h)
    
    err_fwd = np.linalg.norm(grad_fwd - true_grad)
    err_cen = np.linalg.norm(grad_cen - true_grad)
    forward_errors.append(err_fwd)
    central_errors.append(err_cen)

best_h_idx = np.argmin(central_errors)
print(f"Optimal Step Size (Central): h = {h_values[best_h_idx]:.2e} with error {central_errors[best_h_idx]:.2e}")

# 5. Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(h_values, forward_errors, 'ro-', label='Forward Difference O(h)', ms=4)
ax.loglog(h_values, central_errors, 'bs-', label='Central Difference O(h^2)', ms=4)
ax.set_xlabel('Step Size h')
ax.set_ylabel('Absolute Gradient Error')
ax.set_title('Gradient Error vs Step Size (Truncation vs Roundoff)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp15_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
