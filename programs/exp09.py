"""Experiment 09: Matrix Rank, Determinant and Invertibility.
Tasks:
1. Construct square and rectangular matrices representing sensor readings.
2. Calculate matrix rank and determinant where applicable.
3. Determine whether each square matrix is invertible.
4. Compute inverse for valid matrices and verify identity product.
5. Introduce linearly dependent columns and observe effect on rank and invertibility.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 09: Matrix Rank, Determinant & Invertibility ===")

# 1. Matrices
A_full = np.array([
    [4.0, 1.0, 2.0],
    [2.0, 5.0, 1.0],
    [1.0, 2.0, 6.0]
])

# Linearly dependent matrix: Col 3 = Col 1 + Col 2
A_singular = np.array([
    [2.0, 3.0, 5.0],
    [1.0, 4.0, 5.0],
    [3.0, 1.0, 4.0]
])

# 2. Rank & Determinant
rank_full = np.linalg.matrix_rank(A_full)
det_full = np.linalg.det(A_full)
print(f"Matrix A (Full): Rank = {rank_full}/3, Det = {det_full:.4f}")

rank_sing = np.linalg.matrix_rank(A_singular)
det_sing = np.linalg.det(A_singular)
print(f"Matrix B (Linearly Dependent): Rank = {rank_sing}/3, Det = {det_sing:.6f}")

# 3. Invertibility check & computation
is_invertible = abs(det_full) > 1e-6
print(f"\nIs Matrix A Invertible? {is_invertible}")
if is_invertible:
    A_inv = np.linalg.inv(A_full)
    print("Inverse A_inv:\n", np.round(A_inv, 4))
    prod = A_full @ A_inv
    print("Verification A @ A_inv = I (allclose):", np.allclose(prod, np.eye(3)))

# 5. Singular values of both
s_full = np.linalg.svd(A_full, compute_uv=False)
s_sing = np.linalg.svd(A_singular, compute_uv=False)
print(f"\nSingular values A: {np.round(s_full, 3)}")
print(f"Singular values B (Notice zero/tiny eigenvalue): {np.round(s_sing, 6)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

bars1 = axes[0].bar(['s1', 's2', 's3'], s_full, color='royalblue')
axes[0].set_title(f'Full Rank Matrix (Rank={rank_full}, Det={det_full:.1f})')
axes[0].set_ylabel('Singular Value Magnitude')
axes[0].grid(True, alpha=0.3)

bars2 = axes[1].bar(['s1', 's2', 's3'], s_sing, color='crimson')
axes[1].set_title(f'Deficient Matrix (Rank={rank_sing}, Det={det_sing:.1e})')
axes[1].set_ylabel('Singular Value Magnitude')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp09_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
