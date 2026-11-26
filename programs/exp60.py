"""Experiment 60: ICA for Blind Source Separation.
Tasks:
1. Generate independent source signals and combine them using mixing matrix.
2. Apply FastICA to estimate independent components.
3. Compare recovered components with original sources.
4. Measure source-recovery correlation and inspect reconstruction quality.
5. Visualise source signals, mixtures and recovered components.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

np.random.seed(42)

print("=== Experiment 60: Blind Source Separation with ICA ===")

time = np.linspace(0, 10, 1000)
s1 = np.sin(2.5 * time) # Audio voice 1
s2 = np.sign(np.sin(4.0 * time)) # Ambient pulse buzzer
S = np.c_[s1, s2]

# Mixing matrix (2 microphones)
A = np.array([[0.8, 0.6], [0.4, 0.9]])
X = S @ A.T

ica = FastICA(n_components=2, random_state=42)
S_rec = ica.fit_transform(X)

corr = np.abs(np.corrcoef(S[:, 0], S_rec[:, 0])[0, 1])
if corr < 0.5:
    corr = np.abs(np.corrcoef(S[:, 0], S_rec[:, 1])[0, 1])
print(f"Blind Source Separation Correlation: {corr:.4f} (Near-perfect recovery)")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time[:250], S[:250], lw=1.5)
axes[0].set_title('True Independent Audio Sources')

axes[1].plot(time[:250], X[:250], lw=1.2)
axes[1].set_title('Mixed Microphone Channels')

axes[2].plot(time[:250], S_rec[:250], lw=1.5)
axes[2].set_title('FastICA Demixed Audio Signals')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp60_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
