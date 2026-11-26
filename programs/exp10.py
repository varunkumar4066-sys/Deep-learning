"""Experiment 10: Singular Value Decomposition for Data Compression.
Tasks:
1. Load a grayscale image / matrix.
2. Compute its singular value decomposition using NumPy.
3. Reconstruct image using selected numbers of singular values.
4. Calculate reconstruction error and compression ratio for each selection.
5. Plot original image and reconstructed images side by side.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Load an image from digits dataset (expanded to 64x64 for visual clarity)
digits = load_digits()
img_orig = digits.images[0] # 8x8
# Tile to 32x32 for clear SVD spectrum demonstration
img = np.kron(img_orig, np.ones((4, 4)))

# 2. SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)
total_sv = len(S)
total_bytes_orig = img.size

print("=== Experiment 10: SVD Image Compression ===")
print(f"Image Dimensions: {img.shape}, Total Singular Values: {total_sv}")

# 3 & 4. Reconstruct with k values
k_values = [2, 4, 8, 16]
reconstructions = []
errors = []
comp_ratios = []

for k in k_values:
    # Uk @ Sk @ Vtk
    recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    mse = np.mean((img - recon)**2)
    # Compressed bytes: k*(m + n + 1)
    k_params = k * (img.shape[0] + img.shape[1] + 1)
    ratio = k_params / total_bytes_orig
    reconstructions.append(recon)
    errors.append(mse)
    comp_ratios.append(ratio)
    print(f"k = {k:2d} | MSE: {mse:.4f} | Compression Ratio: {ratio*100:.1f}%")

# 5. Plot
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original (32x32)')
axes[0].axis('off')

for i, k in enumerate(k_values):
    axes[i+1].imshow(reconstructions[i], cmap='gray')
    axes[i+1].set_title(f'k = {k}\nMSE: {errors[i]:.2f}')
    axes[i+1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp10_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
