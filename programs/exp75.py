"""Experiment 75: Deep Boltzmann Machine Mean-Field Inference.
Tasks:
1. Define a two-hidden-layer Boltzmann Machine with visible and hidden units.
2. Initialise weights and biases for adjacent layers.
3. Implement iterative mean-field updates for hidden-unit expectations.
4. Compare hidden representations for different input patterns.
5. Visualise inferred activations and discuss convergence behaviour.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 75: DBM Mean-Field Inference ===")

n_v, n_h1, n_h2 = 12, 6, 3
W1 = np.random.randn(n_v, n_h1) * 0.3
W2 = np.random.randn(n_h1, n_h2) * 0.3

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

# Compare two distinct input patterns
v1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
v2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

def infer(v):
    mu1 = np.full((1, n_h1), 0.5)
    mu2 = np.full((1, n_h2), 0.5)
    traj = []
    for _ in range(8):
        mu1 = sigmoid(v @ W1 + mu2 @ W2.T)
        mu2 = sigmoid(mu1 @ W2)
        traj.append(mu2.copy())
    return mu2[0], traj

mu2_v1, traj1 = infer(v1)
mu2_v2, traj2 = infer(v2)

print(f"Pattern 1 Inferred Latent Code (Layer 2): {np.round(mu2_v1, 3)}")
print(f"Pattern 2 Inferred Latent Code (Layer 2): {np.round(mu2_v2, 3)}")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot([t[0, 0] for t in traj1], 'b-o', label='Pattern 1 (Unit 1 trajectory)')
ax.plot([t[0, 0] for t in traj2], 'r-s', label='Pattern 2 (Unit 1 trajectory)')
ax.set_xlabel('Mean-Field Iteration Step')
ax.set_ylabel('Expectation E[h2_1]')
ax.set_title('DBM Mean-Field Fixed-Point Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp75_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
