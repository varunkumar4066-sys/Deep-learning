"""Experiment 01: Vector and Matrix Operations for Deep Learning.
Tasks:
1. Build customer-genre (6x5) and genre-book (5x4) matrices in NumPy.
2. Compute dot products, matrix product, transpose, inverse (where valid), and element-wise ops.
3. Compute cosine similarity between customers and identify the two most similar.
4. Show that a dense layer is y = Wx + b by computing it manually for one customer.
5. Visualise the score matrix as a heatmap.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Build matrices
customers = ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan', 'Fiona']
genres = ['SciFi', 'Mystery', 'Romance', 'Thriller', 'NonFiction']
books = ['Book_A', 'Book_B', 'Book_C', 'Book_D']

C = np.array([
    [5, 4, 1, 4, 2],
    [1, 2, 5, 2, 4],
    [4, 5, 1, 5, 1],
    [2, 1, 4, 3, 5],
    [5, 5, 2, 4, 1],
    [1, 3, 4, 2, 5]
], dtype=np.float32)

G = np.array([
    [0.9, 0.1, 0.2, 0.8],
    [0.7, 0.8, 0.1, 0.9],
    [0.1, 0.2, 0.9, 0.2],
    [0.8, 0.9, 0.3, 0.7],
    [0.2, 0.3, 0.8, 0.4]
], dtype=np.float32)

print("=== Experiment 01: Vector & Matrix Operations ===")
print("Customer-Genre Matrix C (6x5):\n", C)
print("Genre-Book Matrix G (5x4):\n", G)

# 2. Matrix product and operations
scores = C @ G
print("\nRecommendation Score Matrix (C @ G) [6x4]:\n", np.round(scores, 2))
print("Transpose of C (5x6) shape:", C.T.shape)

# Square submatrix inverse
square_sub = C[:5, :5]
inv_sub = np.linalg.inv(square_sub)
print("Inverse of 5x5 submatrix computed successfully. Det:", round(np.linalg.det(square_sub), 4))
print("Verification (A @ A_inv ~ I):", np.allclose(square_sub @ inv_sub, np.eye(5)))

# 3. Cosine similarity between customers
norms = np.linalg.norm(C, axis=1, keepdims=True)
normalized_C = C / (norms + 1e-9)
cos_sim = normalized_C @ normalized_C.T

np.fill_diagonal(cos_sim, -1.0)
max_idx = np.unravel_index(np.argmax(cos_sim), cos_sim.shape)
c1, c2 = customers[max_idx[0]], customers[max_idx[1]]
print(f"\nMost similar customers: {c1} and {c2} with Cosine Similarity = {cos_sim[max_idx]:.4f}")
np.fill_diagonal(cos_sim, 1.0)

# 4. Dense layer y = Wx + b manually
W = G.T # 4x5
x = C[0] # 5
b = np.array([0.5, -0.2, 0.1, 0.4], dtype=np.float32)
y_dense = W @ x + b
print(f"\nDense layer output y = Wx + b for {customers[0]}: {np.round(y_dense, 3)}")

# 5. Visualise heatmap
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = axes[0].imshow(scores, cmap='viridis', aspect='auto')
axes[0].set_xticks(range(len(books)))
axes[0].set_xticklabels(books)
axes[0].set_yticks(range(len(customers)))
axes[0].set_yticklabels(customers)
axes[0].set_title('Customer-Book Recommendation Scores')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(cos_sim, cmap='coolwarm', aspect='auto')
axes[1].set_xticks(range(len(customers)))
axes[1].set_xticklabels(customers, rotation=45)
axes[1].set_yticks(range(len(customers)))
axes[1].set_yticklabels(customers)
axes[1].set_title('Customer Cosine Similarity Heatmap')
fig.colorbar(im1, ax=axes[1])

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp01_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
