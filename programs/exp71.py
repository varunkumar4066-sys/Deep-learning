"""Experiment 71: RBM Training on Binary Image Data.
Tasks:
1. Load and binarise handwritten-digit dataset.
2. Initialise visible biases, hidden biases and weight parameters.
3. Calculate hidden activation probabilities for input images.
4. Train RBM using contrastive divergence.
5. Visualise learned weights and hidden-unit activation patterns.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(42)

print("=== Experiment 71: RBM on Binary Image Data ===")

digits = load_digits()
X = (digits.images.reshape(-1, 64) > 4).astype(np.float32)

n_vis, n_hid = 64, 25
W = np.random.randn(n_vis, n_hid) * 0.05
b_vis = np.zeros(n_vis)
b_hid = np.zeros(n_hid)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

# CD-1 training
lr = 0.1
for epoch in range(15):
    h_prob = sigmoid(X @ W + b_hid)
    h_samp = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float32)
    pos_grad = X.T @ h_prob
    
    v_recon = sigmoid(h_samp @ W.T + b_vis)
    h_recon = sigmoid(v_recon @ W + b_hid)
    neg_grad = v_recon.T @ h_recon
    
    W += (lr / len(X)) * (pos_grad - neg_grad)
    b_vis += (lr / len(X)) * np.sum(X - v_recon, axis=0)
    b_hid += (lr / len(X)) * np.sum(h_prob - h_recon, axis=0)

print("RBM training on binary digits completed.")

# Plot 5 learned receptive fields
fig, axes = plt.subplots(1, 5, figsize=(11, 2.5))
for i in range(5):
    axes[i].imshow(W[:, i].reshape(8, 8), cmap='coolwarm')
    axes[i].set_title(f'Filter {i+1}')
    axes[i].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp71_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
