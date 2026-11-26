"""Experiment 52: Independent Component Analysis and Component Visualization.
Tasks:
1. Generate source signals (sine, square, sawtooth) and mix them.
2. Apply FastICA to recover sources.
3. Apply PCA to same mixtures for comparison.
4. Plot original, mixed, ICA-recovered and PCA-recovered signals.
5. Measure recovery quality with correlation to true sources.
"""
from pathlib import Path
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

np.random.seed(42)

print("=== Experiment 52: Independent Component Analysis (ICA) ===")

# 1. Sources
n_samples = 1000
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)  # Sine wave
s2 = np.sign(np.sin(3 * time)) # Square wave
s3 = signal.sawtooth(2 * np.pi * time) # Sawtooth wave
S = np.c_[s1, s2, s3]
S += 0.1 * np.random.normal(size=S.shape)

# Mixing
A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T

# 2. FastICA
ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X)

# 3. PCA for comparison
pca = PCA(n_components=3)
S_pca = pca.fit_transform(X)

# 5. Correlation quality
corrs_ica = np.max(np.abs(np.corrcoef(S.T, S_ica.T)[:3, 3:]), axis=1)
corrs_pca = np.max(np.abs(np.corrcoef(S.T, S_pca.T)[:3, 3:]), axis=1)

print(f"ICA Source Recovery Correlations: {np.round(corrs_ica, 4)}")
print(f"PCA Source Recovery Correlations: {np.round(corrs_pca, 4)} (Notice PCA fails to separate non-orthogonal signals)")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

axes[0].plot(time[:200], S[:200], lw=1.5)
axes[0].set_title('Original Independent Sources (Sine, Square, Sawtooth)')

axes[1].plot(time[:200], X[:200], lw=1.2)
axes[1].set_title('Mixed Observed Signals')

axes[2].plot(time[:200], S_ica[:200], lw=1.5)
axes[2].set_title('FastICA Recovered Sources (True Blind Source Separation)')

axes[3].plot(time[:200], S_pca[:200], lw=1.5)
axes[3].set_title('PCA Components (Residual Cross-Talk / Mixed)')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp52_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
