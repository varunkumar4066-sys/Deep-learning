"""Experiment 44: GRU Gate Operation and State Analysis.
Tasks:
1. Create a small numerical input sequence and initialise GRU parameters.
2. Calculate reset and update gate activations.
3. Implement GRU hidden-state update for successive time steps in NumPy.
4. Verify implementation against Keras GRU layer.
5. Plot gate values and hidden-state trajectories.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 44: GRU Gate Operation & State Analysis ===")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

T = 8
d_in = 1
d_h = 3
x_seq = np.sin(np.linspace(0, 4, T))[:, None]

W_z, U_z, b_z = np.random.randn(d_in, d_h)*0.4, np.random.randn(d_h, d_h)*0.4, np.zeros(d_h)
W_r, U_r, b_r = np.random.randn(d_in, d_h)*0.4, np.random.randn(d_h, d_h)*0.4, np.zeros(d_h)
W_h, U_h, b_h = np.random.randn(d_in, d_h)*0.4, np.random.randn(d_h, d_h)*0.4, np.zeros(d_h)

h_t = np.zeros(d_h)
z_hist, r_hist, h_hist = [], [], []

for t in range(T):
    xt = x_seq[t]
    z_t = sigmoid(xt @ W_z + h_t @ U_z + b_z) # Update gate
    r_t = sigmoid(xt @ W_r + h_t @ U_r + b_r) # Reset gate
    h_tilde = np.tanh(xt @ W_h + (r_t * h_t) @ U_h + b_h)
    h_t = (1.0 - z_t) * h_t + z_t * h_tilde
    
    z_hist.append(z_t)
    r_hist.append(r_t)
    h_hist.append(h_t.copy())

z_hist, r_hist, h_hist = np.array(z_hist), np.array(r_hist), np.array(h_hist)
print(f"GRU State Simulation Complete. Final Hidden State h_T: {np.round(h_hist[-1], 3)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(np.mean(z_hist, axis=1), 'b-o', label='Update Gate z_t')
axes[0].plot(np.mean(r_hist, axis=1), 'r-s', label='Reset Gate r_t')
axes[0].set_xlabel('Time Step t')
axes[0].set_ylabel('Mean Gate Value')
axes[0].set_title('GRU Gates Behaviour')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for u in range(d_h):
    axes[1].plot(h_hist[:, u], label=f'Hidden State {u+1}')
axes[1].set_xlabel('Time Step t')
axes[1].set_ylabel('State Value')
axes[1].set_title('GRU Hidden State Trajectories')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp44_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
