"""Experiment 49: PCA and Explained Variance Analysis.
Tasks:
1. Implement PCA from scratch using covariance matrix and eigen-decomposition.
2. Verify result against Scikit-learn’s PCA.
3. Plot scree plot and cumulative explained variance.
4. Find number of components that retain 95% of variance.
5. Visualise top eigenfaces/components and reconstructed faces with 10, 30 and 50 components.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

print("=== Experiment 49: PCA & Explained Variance ===")

digits = load_digits()
X = digits.data # (1797, 64)
n_samples, n_features = X.shape

# 1. Scratch PCA
mean = np.mean(X, axis=0)
X_centered = X - mean
cov = np.cov(X_centered, rowvar=False)
evals, evecs = np.linalg.eigh(cov)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# 2. Sklearn PCA verification
pca_sk = PCA()
pca_sk.fit(X)
print(f"Scratch vs Sklearn Top-3 Eigenvalues Diff: {np.max(np.abs(evals[:3] - pca_sk.explained_variance_[:3])):.2e}")

# 3. Explained variance
var_ratio = evals / np.sum(evals)
cum_var = np.cumsum(var_ratio)
k_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Components needed for 95% variance: {k_95} / {n_features}")

# 5. Reconstructions
sample = X_centered[0:1] # first digit
recons = {}
for k in [5, 15, 30, k_95]:
    Vk = evecs[:, :k]
    recons[k] = (sample @ Vk @ Vk.T) + mean

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(range(1, 41), cum_var[:40] * 100, 'b-o', ms=3, label='Cumulative Variance')
axes[0].axhline(95, color='r', linestyle='--', label='95% Threshold')
axes[0].axvline(k_95, color='g', linestyle=':', label=f'k = {k_95}')
axes[0].set_xlabel('Principal Component Count')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Scree / Cumulative Variance Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Display sample reconstruction
axes[1].imshow(recons[k_95].reshape(8, 8), cmap='gray')
axes[1].set_title(f'PCA Reconstruction (k={k_95}, 95% Var)')
axes[1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp49_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
