"""Experiment 58: PPCA Latent Variable Sampling.
Tasks:
1. Generate multivariate observations with known latent structure and Gaussian noise.
2. Fit PPCA model or implement its parameter estimation procedure.
3. Estimate latent representation and noise variance.
4. Sample latent variables and generate new observations from fitted model.
5. Compare generated and original feature distributions.
"""
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

print("=== Experiment 58: PPCA Latent Sampling ===")

n, d, q = 400, 6, 2
W_true = np.random.randn(d, q)
z_true = np.random.randn(n, q)
noise_std = 0.3
X = z_true @ W_true.T + np.random.randn(n, d) * noise_std

pca = PCA(n_components=q)
pca.fit(X)
evals = np.linalg.eigvalsh(np.cov(X - X.mean(axis=0), rowvar=False))[::-1]
sigma2 = np.mean(evals[q:])

# Sample 400 new points from generative model
z_new = np.random.randn(n, q)
W_est = pca.components_.T * np.sqrt(np.maximum(0, pca.explained_variance_ - sigma2))
X_gen = z_new @ W_est.T + pca.mean_ + np.random.randn(n, d) * np.sqrt(sigma2)

print(f"Original Feature 1 Mean: {np.mean(X[:, 0]):.3f}, Std: {np.std(X[:, 0]):.3f}")
print(f"Sampled  Feature 1 Mean: {np.mean(X_gen[:, 0]):.3f}, Std: {np.std(X_gen[:, 0]):.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(X[:, 0], X[:, 1], color='steelblue', alpha=0.5, label='Original')
axes[0].scatter(X_gen[:, 0], X_gen[:, 1], color='crimson', alpha=0.5, marker='+', label='PPCA Sampled')
axes[0].set_title('Feature 1 vs Feature 2 Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(X[:, 0], bins=25, density=True, alpha=0.6, color='steelblue', label='Original')
axes[1].hist(X_gen[:, 0], bins=25, density=True, alpha=0.6, color='crimson', label='Sampled')
axes[1].set_title('Marginal Distribution Check')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp58_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
