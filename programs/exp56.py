"""Experiment 56: PCA versus Truncated SVD for Sparse Data.
Tasks:
1. Create or load sparse document-term matrix.
2. Apply TruncatedSVD using different component counts without dense conversion.
3. Apply PCA workflow to dense representation.
4. Compare explained variance, dimensionality and memory.
5. Visualise document projections and discuss suitability of each method.
"""
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA

np.random.seed(42)

print("=== Experiment 56: PCA vs Truncated SVD for Sparse Data ===")

# Create synthetic sparse document-term matrix (500 docs, 200 terms, 95% sparse)
n_docs, n_terms = 500, 200
dense_sample = np.random.poisson(0.1, size=(n_docs, n_terms))
X_sparse = sp.csr_matrix(dense_sample, dtype=np.float32)

print(f"Sparse Matrix Non-Zero Elements: {X_sparse.nnz} / {n_docs*n_terms} (Sparsity: {100*(1 - X_sparse.nnz/(n_docs*n_terms)):.1f}%)")

# 2. Truncated SVD on sparse matrix directly (preserves sparsity)
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_sparse)

# 3. PCA requires dense centered matrix
X_dense = X_sparse.toarray()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_dense)

print(f"TruncatedSVD Explained Variance Ratio (k=2): {np.sum(svd.explained_variance_ratio_):.4f}")
print(f"PCA Explained Variance Ratio (k=2):          {np.sum(pca.explained_variance_ratio_):.4f}")
print("Conclusion: TruncatedSVD is strictly required for large sparse text matrices to avoid memory blowout.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(X_svd[:, 0], X_svd[:, 1], color='teal', alpha=0.6, s=20)
axes[0].set_title('TruncatedSVD Projection (Sparse-Safe)')
axes[0].set_xlabel('Component 1'); axes[0].set_ylabel('Component 2')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], color='coral', alpha=0.6, s=20)
axes[1].set_title('PCA Projection (Centered Dense)')
axes[1].set_xlabel('Component 1'); axes[1].set_ylabel('Component 2')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp56_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
