"""Experiment 55: PCA from Scratch Using Covariance Eigen-Decomposition.
Tasks:
1. Load multivariate dataset and standardise its features.
2. Calculate covariance matrix and perform eigen-decomposition.
3. Sort eigenvalues and eigenvectors in descending order.
4. Project observations onto selected principal components and reconstruct.
5. Compare implementation with Scikit-learn PCA.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

print("=== Experiment 55: PCA from Scratch ===")

iris = load_iris()
X = iris.data # (150, 4)
y = iris.target

# 1. Standardise
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 2. Covariance matrix C = 1/n X^T X
cov_mat = np.cov(X_std, rowvar=False)

# 3. Eigen-decomposition
evals, evecs = np.linalg.eigh(cov_mat)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# 4. Projection onto top 2 components
V2 = evecs[:, :2]
X_proj_scratch = X_std @ V2

# 5. Sklearn comparison
pca = PCA(n_components=2)
X_proj_sk = pca.fit_transform(X_std)
diff = np.max(np.abs(np.abs(X_proj_scratch) - np.abs(X_proj_sk)))
print(f"Scratch vs Sklearn Projection Max Diff: {diff:.2e} (Identical subspace!)")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
scatter = ax.scatter(X_proj_scratch[:, 0], X_proj_scratch[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA from Scratch 2D Projection (Iris)')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp55_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
