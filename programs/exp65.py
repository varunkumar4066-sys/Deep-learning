"""Experiment 65: Restricted Boltzmann Machine Energy and Reconstruction.
Tasks:
1. Implement energy function E(v, h) = -a^T v - b^T h - v^T W h in NumPy.
2. Compute hidden activation probabilities p(h|v) and visible probabilities p(v|h).
3. Train RBM with one-step contrastive divergence (CD-1).
4. Plot average free energy and reconstruction error over epochs.
5. Show original and reconstructed samples side by side.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(42)

print("=== Experiment 65: RBM Energy & Reconstruction ===")

# Digits binarized to visible units v in {0, 1}
digits = load_digits()
X = (digits.images.reshape(-1, 64) > 4).astype(np.float32)

n_vis = 64
n_hid = 32
n_samples = len(X)

W = np.random.randn(n_vis, n_hid) * 0.1
a = np.zeros(n_vis) # visible bias
b = np.zeros(n_hid) # hidden bias

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

def free_energy(v):
    # F(v) = -a^T v - sum ln(1 + exp(W^T v + b))
    wx_b = v @ W + b
    return -v @ a - np.sum(np.log(1.0 + np.exp(np.clip(wx_b, -15, 15))), axis=1)

# Train with CD-1
lr = 0.1
epochs = 20
errors, free_energies = [], []

for ep in range(epochs):
    # Positive phase
    h_prob = sigmoid(X @ W + b)
    h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float32)
    pos_grad = X.T @ h_prob
    
    # Negative phase (reconstruction)
    v_recon_prob = sigmoid(h_sample @ W.T + a)
    v_recon_sample = (np.random.rand(*v_recon_prob.shape) < v_recon_prob).astype(np.float32)
    h_recon_prob = sigmoid(v_recon_sample @ W + b)
    neg_grad = v_recon_sample.T @ h_recon_prob
    
    # CD-1 updates
    W += (lr / n_samples) * (pos_grad - neg_grad)
    a += (lr / n_samples) * np.sum(X - v_recon_sample, axis=0)
    b += (lr / n_samples) * np.sum(h_prob - h_recon_prob, axis=0)
    
    err = np.mean((X - v_recon_prob)**2)
    fe = np.mean(free_energy(X))
    errors.append(err)
    free_energies.append(fe)

print(f"RBM Trained for {epochs} epochs | Final Recon MSE: {errors[-1]:.4f} | Final Free Energy: {free_energies[-1]:.2f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(errors, 'r-o', lw=1.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Reconstruction MSE')
axes[0].set_title('RBM Reconstruction Error')
axes[0].grid(True, alpha=0.3)

axes[1].imshow(X[0].reshape(8, 8), cmap='gray')
axes[1].set_title('Original Digit Sample')
axes[1].axis('off')

axes[2].imshow(v_recon_prob[0].reshape(8, 8), cmap='gray')
axes[2].set_title('RBM CD-1 Reconstruction')
axes[2].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp65_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
