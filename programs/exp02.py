"""Experiment 02: Covariance, Eigenvalues and Eigenvectors Analysis.
Tasks:
1. Load patient records with correlated blood parameters into Pandas and standardise.
2. Compute covariance and correlation matrices; plot them as heatmaps.
3. Compute eigenvalues and eigenvectors with NumPy; sort by eigenvalue.
4. Plot data in 2-D with top two eigenvectors drawn as arrows.
5. Report percentage of variance captured by each eigenvalue.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Generate 200 patient records
n = 200
features = ['Haemoglobin', 'RBC_Count', 'Glucose', 'Cholesterol', 'BMI']
base = np.random.randn(n, 3)
haemoglobin = 13.5 + 1.2 * base[:, 0] + 0.3 * np.random.randn(n)
rbc = 4.8 + 0.8 * base[:, 0] + 0.2 * np.random.randn(n)
glucose = 100.0 + 15.0 * base[:, 1] + 5.0 * np.random.randn(n)
cholesterol = 190.0 + 20.0 * base[:, 1] + 10.0 * base[:, 2]
bmi = 24.5 + 2.5 * base[:, 2] + 1.0 * np.random.randn(n)

df = pd.DataFrame(np.column_stack([haemoglobin, rbc, glucose, cholesterol, bmi]), columns=features)
print("=== Experiment 02: Covariance & Eigenvalues Analysis ===")
print("Patient Records Sample:\n", df.head(3))

# Standardise
df_std = (df - df.mean()) / df.std()

# 2. Covariance and correlation
cov_mat = df_std.cov().to_numpy()
corr_mat = df_std.corr().to_numpy()
print("\nCovariance Matrix (Standardised):\n", np.round(cov_mat, 3))

# 3. Eigenvalues and eigenvectors
evals, evecs = np.linalg.eigh(cov_mat)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# 5. Variance explained
total_var = np.sum(evals)
var_pct = (evals / total_var) * 100.0
print("\nEigenvalues & Explained Variance:")
for i, (v, p) in enumerate(zip(evals, var_pct), 1):
    print(f"  lambda_{i}: {v:.4f} ({p:.2f}%)")

# 4. Plot 2D projection and eigenvector arrows
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
im = axes[0].imshow(corr_mat, cmap='Blues', aspect='auto')
axes[0].set_xticks(range(len(features)))
axes[0].set_xticklabels(features, rotation=45, ha='right')
axes[0].set_yticks(range(len(features)))
axes[0].set_yticklabels(features)
axes[0].set_title('Correlation Matrix Heatmap')
for i in range(len(features)):
    for j in range(len(features)):
        axes[0].text(j, i, f"{corr_mat[i, j]:.2f}", ha='center', va='center', color='black')
fig.colorbar(im, ax=axes[0])

# Project onto first 2 features for visualization
x1, x2 = df_std['Haemoglobin'].to_numpy(), df_std['RBC_Count'].to_numpy()
axes[1].scatter(x1, x2, alpha=0.6, color='steelblue', label='Patients')
mean1, mean2 = 0, 0
axes[1].quiver(mean1, mean2, evecs[0, 0]*evals[0], evecs[1, 0]*evals[0], angles='xy', scale_units='xy', scale=1, color='red', label='PC1 vector')
axes[1].quiver(mean1, mean2, evecs[0, 1]*evals[1], evecs[1, 1]*evals[1], angles='xy', scale_units='xy', scale=1, color='green', label='PC2 vector')
axes[1].set_xlabel('Standardised Haemoglobin')
axes[1].set_ylabel('Standardised RBC Count')
axes[1].set_title(f'Top Eigenvectors (Explaining {var_pct[0]+var_pct[1]:.1f}% Var)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp02_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
