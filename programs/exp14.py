"""Experiment 14: Sampling Methods and the Central Limit Theorem.
Tasks:
1. Generate observations from a skewed population distribution.
2. Draw repeated samples using different sample sizes.
3. Calculate mean of each sample and plot sampling distributions.
4. Compare sampling distributions with a Gaussian curve.
5. Explain how sample size affects spread and shape of distribution.
"""
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 14: Sampling & Central Limit Theorem ===")

# 1. Skewed population: Exponential with scale = 2.0 (mean = 2.0, std = 2.0)
scale = 2.0
pop_mean = scale
pop_std = scale

# 2. Draw repeated samples of size N = [2, 10, 30, 100]
sample_sizes = [2, 10, 30, 100]
n_samples = 2000

means_dict = {}
for n in sample_sizes:
    # 2000 x n matrix of samples
    samples = np.random.exponential(scale=scale, size=(n_samples, n))
    sample_means = np.mean(samples, axis=1)
    means_dict[n] = sample_means
    print(f"N = {n:3d} | Empirical Mean: {np.mean(sample_means):.4f} | Empirical Std: {np.std(sample_means):.4f} (Theoretical: {pop_std/np.sqrt(n):.4f})")

# Plot
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for i, n in enumerate(sample_sizes):
    m = means_dict[n]
    axes[i].hist(m, bins=30, density=True, alpha=0.6, color='teal', edgecolor='black')
    # Theoretical normal fit
    se = pop_std / np.sqrt(n)
    x = np.linspace(pop_mean - 3.5*se, pop_mean + 3.5*se, 150)
    axes[i].plot(x, stats.norm.pdf(x, pop_mean, se), 'r-', lw=2)
    axes[i].set_title(f'N = {n}\nStd: {np.std(m):.2f}')
    axes[i].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp14_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
