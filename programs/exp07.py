"""Experiment 07: Linear Algebra Operations for Neural Network Data Preparation.
Tasks:
1. Create feature vectors and customer-feature matrix from a sample dataset.
2. Perform matrix addition, subtraction, multiplication, transpose and element-wise ops.
3. Check matrix dimensions and demonstrate dimension mismatch error handling.
4. Apply a matrix transformation to the input features.
5. Visualise original and transformed feature matrices.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 07: Linear Algebra for Data Preparation ===")

# 1. Feature vectors
customers = [f"Cust_{i+1}" for i in range(8)]
features = ['Age', 'Annual_Spend_k', 'Browse_Hrs', 'Purchases']

# Raw data
data = np.array([
    [25, 45, 12, 5],
    [40, 85, 8, 14],
    [31, 60, 20, 9],
    [52, 110, 4, 22],
    [22, 28, 25, 3],
    [38, 72, 15, 11],
    [45, 95, 6, 18],
    [29, 50, 18, 7]
], dtype=np.float32)

df = pd.DataFrame(data, index=customers, columns=features)
print("Customer Feature Matrix (8x4):\n", df)

# 2. Arithmetic operations
mean_vec = np.mean(data, axis=0)
centered = data - mean_vec
print("\nFeature Means:", np.round(mean_vec, 2))
print("Centered Matrix Sample (First 2 rows):\n", np.round(centered[:2], 2))

# 3. Dimension mismatch demonstration
try:
    invalid_op = data @ np.ones((5, 2))
except ValueError as e:
    print(f"\nCaught Dimension Mismatch correctly: {e}")

# 4. Linear Transformation W (4 -> 2 features for latent compression)
W_proj = np.array([
    [0.1, 0.4],
    [0.5, 0.2],
    [-0.3, 0.6],
    [0.7, -0.1]
], dtype=np.float32)

transformed = centered @ W_proj
print("Transformed Matrix Shape (8x2):", transformed.shape)
print("Latent Feature Coordinates Sample:\n", np.round(transformed[:3], 3))

# 5. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(data, cmap='YlGnBu', aspect='auto')
axes[0].set_xticks(range(len(features)))
axes[0].set_xticklabels(features, rotation=30)
axes[0].set_yticks(range(len(customers)))
axes[0].set_yticklabels(customers)
axes[0].set_title('Original Features Matrix (8x4)')
fig.colorbar(im, ax=axes[0])

axes[1].scatter(transformed[:, 0], transformed[:, 1], color='darkorange', s=70, edgecolors='black')
for i, txt in enumerate(customers):
    axes[1].annotate(txt, (transformed[i, 0]+0.5, transformed[i, 1]+0.5), fontsize=8)
axes[1].set_xlabel('Projected Component 1')
axes[1].set_ylabel('Projected Component 2')
axes[1].set_title('Transformed Feature Space (2D Projection)')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp07_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
