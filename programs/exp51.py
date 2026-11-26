"""Experiment 51: Factor Analysis and Latent Factor Interpretation.
Tasks:
1. Load survey data in Pandas; check correlations.
2. Fit Scikit-learn’s FactorAnalysis with 2, 3 and 4 factors.
3. Apply varimax rotation and display loadings matrix as a heatmap.
4. Name each factor from high-loading questions (Compensation, Work Culture).
5. Compare factor analysis with PCA on same data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

np.random.seed(42)

print("=== Experiment 51: Factor Analysis ===")

# 10 survey questions
questions = ['Salary', 'Bonus', 'Stock_Options', 'Team_Spirit', 'Manager_Support', 
             'Work_Life_Bal', 'Learning_Budget', 'Mentorship', 'Autonomy', 'Recognition']
n_resp = 300

# Hidden ground-truth factors: Factor 1: Comp (Q0-Q2), Factor 2: Culture (Q3-Q5), Factor 3: Growth (Q6-Q9)
f1 = np.random.randn(n_resp, 1)
f2 = np.random.randn(n_resp, 1)
f3 = np.random.randn(n_resp, 1)

data = np.zeros((n_resp, 10))
data[:, 0:3] = f1 * [1.2, 1.1, 0.9] + np.random.randn(n_resp, 3)*0.3
data[:, 3:6] = f2 * [1.0, 1.1, 0.8] + np.random.randn(n_resp, 3)*0.3
data[:, 6:10] = f3 * [0.9, 1.0, 0.8, 1.1] + np.random.randn(n_resp, 4)*0.3

# Fit FactorAnalysis (n_components=3, rotation='varimax')
fa = FactorAnalysis(n_components=3, rotation='varimax', random_state=42)
loadings = fa.fit(data).components_.T # (10, 3)

factor_names = ['Factor 1 (Comp)', 'Factor 2 (Culture)', 'Factor 3 (Growth)']
print("Rotated Factor Loadings Matrix (10x3):\n", np.round(loadings, 2))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(loadings, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(factor_names)
ax.set_yticks(range(10))
ax.set_yticklabels(questions)
ax.set_title('Varimax-Rotated Factor Loadings Heatmap')
for i in range(10):
    for j in range(3):
        ax.text(j, i, f"{loadings[i, j]:.2f}", ha='center', va='center', color='black')
fig.colorbar(im, ax=ax)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp51_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
