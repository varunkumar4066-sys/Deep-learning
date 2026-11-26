"""Experiment 59: Factor Analysis with Rotation Methods.
Tasks:
1. Prepare questionnaire response matrix and examine feature correlations.
2. Fit factor-analysis models with different numbers of latent factors.
3. Apply varimax rotation to loading matrix.
4. Compare factor interpretability before and after rotation.
5. Visualise loadings and assign evidence-based factor labels.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

np.random.seed(42)

print("=== Experiment 59: Factor Analysis Rotation ===")

features = ['Workload', 'Supervisor_Support', 'Confidence', 'Facilities', 'Motivation', 'Peer_Trust']
n = 250
# Two latent drivers: Factor A (Environment/Support), Factor B (Personal Drive)
fA = np.random.randn(n, 1)
fB = np.random.randn(n, 1)

data = np.zeros((n, 6))
data[:, [1, 3, 5]] = fA * [1.1, 0.9, 1.0] + np.random.randn(n, 3) * 0.3
data[:, [0, 2, 4]] = fB * [-0.8, 1.0, 1.1] + np.random.randn(n, 3) * 0.3

# Unrotated
fa_unrotated = FactorAnalysis(n_components=2, rotation=None, random_state=42)
L_unrot = fa_unrotated.fit(data).components_.T

# Rotated (Varimax)
fa_rotated = FactorAnalysis(n_components=2, rotation='varimax', random_state=42)
L_rot = fa_rotated.fit(data).components_.T

print("Unrotated Loadings:\n", np.round(L_unrot, 2))
print("\nVarimax Rotated Loadings (Clean separation):\n", np.round(L_rot, 2))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].imshow(L_unrot, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Factor 1', 'Factor 2'])
axes[0].set_yticks(range(6)); axes[0].set_yticklabels(features)
axes[0].set_title('Unrotated Loadings')
for i in range(6):
    for j in range(2):
        axes[0].text(j, i, f"{L_unrot[i, j]:.2f}", ha='center', va='center')

axes[1].imshow(L_rot, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Factor 1 (Drive)', 'Factor 2 (Support)'])
axes[1].set_yticks(range(6)); axes[1].set_yticklabels(features)
axes[1].set_title('Varimax Rotated Loadings')
for i in range(6):
    for j in range(2):
        axes[1].text(j, i, f"{L_rot[i, j]:.2f}", ha='center', va='center')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp59_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
