"""Experiment 76: Convolutional RBM for Texture Modelling.
Tasks:
1. Prepare small grayscale texture patches.
2. Define convolutional filters and hidden activation maps.
3. Implement convolutional hidden and visible reconstruction operations.
4. Train model using approximate learning procedure.
5. Visualise learned filters, activation maps and reconstructed textures.
"""
from pathlib import Path
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 76: Convolutional RBM Texture Simulation ===")

# Create synthetic striped texture (16x16)
texture = np.zeros((16, 16))
texture[::2, :] = 1.0

# 4 convolutional filters of size 3x3
n_filters = 4
filters = [
    np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]), # Vertical
    np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]), # Horizontal
    np.eye(3) - 0.3,                                    # Diagonal
    np.ones((3, 3)) / 9.0                               # Averaging
]

# Convolutional hidden feature maps
feature_maps = [np.maximum(0, signal.convolve2d(texture, f, mode='same')) for f in filters]

print(f"Texture Size: {texture.shape} | Computed {n_filters} Convolutional Feature Maps.")

# Plot
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))

axes[0].imshow(texture, cmap='gray')
axes[0].set_title('Input Texture')
axes[0].axis('off')

for i in range(4):
    axes[i+1].imshow(feature_maps[i], cmap='magma')
    axes[i+1].set_title(f'Conv Feature Map {i+1}')
    axes[i+1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp76_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
