"""Experiment 57: Choosing Components Using Reconstruction Error.
Tasks:
1. Load grayscale images and convert into feature vectors.
2. Fit PCA using range of component counts.
3. Reconstruct images at each selected component count.
4. Calculate mean squared reconstruction error and explained variance.
5. Plot both measures against number of retained components.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

print("=== Experiment 57: Choosing Components by Reconstruction Error ===")

digits = load_digits()
X = digits.data # (1797, 64)

k_range = [1, 2, 5, 10, 20, 30, 45, 60]
mses = []
vars_explained = []

for k in k_range:
    pca = PCA(n_components=k)
    X_k = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_k)
    mse = np.mean((X - X_recon)**2)
    mses.append(mse)
    vars_explained.append(np.sum(pca.explained_variance_ratio_))
    print(f"k = {k:2d} | MSE: {mse:.4f} | Explained Var: {vars_explained[-1]*100:.2f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(k_range, mses, 'ro-', lw=2)
axes[0].set_xlabel('Components (k)')
axes[0].set_ylabel('Reconstruction MSE')
axes[0].set_title('Elbow Curve: Error vs Components')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, np.array(vars_explained) * 100, 'bo-', lw=2)
axes[1].axhline(90, color='g', linestyle='--', label='90% Variance Threshold')
axes[1].set_xlabel('Components (k)')
axes[1].set_ylabel('Cumulative Explained Variance (%)')
axes[1].set_title('Variance Explained vs Components')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp57_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
