"""Experiment 50: Probabilistic PCA and Reconstruction Analysis.
Tasks:
1. Explain PPCA generative model x = Wz + mu + epsilon.
2. Fit PPCA (PCA with noise-variance estimate).
3. Estimate noise variance sigma^2.
4. Reconstruct data and compute reconstruction error against component count.
5. Generate new synthetic sensor samples from fitted model.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

print("=== Experiment 50: Probabilistic PCA ===")

# Generate synthetic sensor data with d=5, latent q=2, noise sigma=0.4
n = 300
d = 5
q = 2
true_sigma = 0.4
W_true = np.random.randn(d, q) * 1.5
z_true = np.random.randn(n, q)
mu_true = np.array([10.0, 25.0, 50.0, 5.0, 15.0])
eps = np.random.randn(n, d) * true_sigma
X = z_true @ W_true.T + mu_true + eps

# Fit PCA with q components
pca = PCA(n_components=q)
pca.fit(X)

# Estimate noise variance: average of discarded eigenvalues
evals_all = np.linalg.eigvalsh(np.cov(X - X.mean(axis=0), rowvar=False))[::-1]
sigma2_est = np.mean(evals_all[q:])
print(f"True Noise Variance: {true_sigma**2:.4f} | PPCA Estimated sigma^2: {sigma2_est:.4f}")

# Sample from generative model: z ~ N(0, I), eps ~ N(0, sigma^2 I)
n_gen = 100
z_sample = np.random.randn(n_gen, q)
W_est = pca.components_.T * np.sqrt(np.maximum(0, pca.explained_variance_ - sigma2_est))
eps_sample = np.random.randn(n_gen, d) * np.sqrt(sigma2_est)
X_synthetic = z_sample @ W_est.T + pca.mean_ + eps_sample

print(f"Generated {n_gen} synthetic sensor readings from PPCA model.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(X[:, 0], X[:, 1], color='steelblue', alpha=0.6, label='Observed Sensor Data')
axes[0].scatter(X_synthetic[:, 0], X_synthetic[:, 1], color='darkorange', marker='x', label='PPCA Sampled')
axes[0].set_xlabel('Sensor 1')
axes[0].set_ylabel('Sensor 2')
axes[0].set_title('Original vs PPCA Generated Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstruction MSE vs component count
mses = []
for k in range(1, d + 1):
    p = PCA(n_components=k)
    recon = p.inverse_transform(p.fit_transform(X))
    mses.append(np.mean((X - recon)**2))
axes[1].plot(range(1, d + 1), mses, 'ro-', lw=2)
axes[1].set_xlabel('Latent Dimensions q')
axes[1].set_ylabel('Reconstruction MSE')
axes[1].set_title('PPCA Reconstruction Error vs Dimension')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp50_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
