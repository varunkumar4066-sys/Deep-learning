"""Experiment 27: Convolution and Pooling Operations.
Tasks:
1. Load a grayscale image and define edge-detection and blur kernels.
2. Apply 2-D convolution with selected stride and padding values.
3. Implement max pooling and average pooling.
4. Compare feature-map dimensions before and after pooling.
5. Visualise original image, filters, convolution outputs and pooled maps.
"""
from pathlib import Path
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

print("=== Experiment 27: Convolution & Pooling Operations ===")

# 1. Grayscale image
img = load_digits().images[3] # 8x8 digit 3
# Enlarge to 16x16
img = np.kron(img, np.ones((2, 2)))

# Filters
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
blur = np.ones((3, 3), dtype=np.float32) / 9.0

# 2. Convolution (same mode)
conv_edge = signal.convolve2d(img, sobel_v, mode='same')
conv_blur = signal.convolve2d(img, blur, mode='same')

# 3. Pooling (2x2)
def pool2d(mat, pool_size=(2, 2), mode='max'):
    h, w = mat.shape
    ph, pw = pool_size
    out = np.zeros((h // ph, w // pw))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = mat[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            out[i, j] = np.max(patch) if mode == 'max' else np.mean(patch)
    return out

max_pooled = pool2d(conv_edge, (2, 2), mode='max')
avg_pooled = pool2d(conv_edge, (2, 2), mode='avg')

print(f"Original Shape: {img.shape}")
print(f"Conv Edge Map Shape: {conv_edge.shape}")
print(f"Max-Pooled Map Shape: {max_pooled.shape} (Dimensions reduced by factor of 2)")

# Plot
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original (16x16)')
axes[0].axis('off')

axes[1].imshow(conv_edge, cmap='gray')
axes[1].set_title('Sobel Edge Conv')
axes[1].axis('off')

axes[2].imshow(conv_blur, cmap='gray')
axes[2].set_title('Blur Conv')
axes[2].axis('off')

axes[3].imshow(max_pooled, cmap='gray')
axes[3].set_title('Max-Pooled (8x8)')
axes[3].axis('off')

axes[4].imshow(avg_pooled, cmap='gray')
axes[4].set_title('Avg-Pooled (8x8)')
axes[4].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp27_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
