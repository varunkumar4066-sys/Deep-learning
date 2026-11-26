"""Experiment 03: Probability Distributions and Bayes’ Theorem.
Tasks:
1. Generate and plot Bernoulli, Binomial, Poisson, Uniform and Gaussian samples with SciPy; compare with PDFs/PMFs.
2. Compute P(Disease | Positive) using Bayes’ theorem.
3. Simulate 100,000 patients and verify result empirically.
4. Plot how posterior changes as prevalence varies from 0.1% to 20%.
"""
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 03: Probability Distributions & Bayes' Theorem ===")

# 2. Bayes' theorem theoretical calculation
prev = 0.01      # P(Disease)
sens = 0.95      # P(+ | Disease)
spec = 0.90      # P(- | No Disease) -> P(+ | No Disease) = 0.10
p_pos_given_no_d = 1.0 - spec

p_pos = (sens * prev) + (p_pos_given_no_d * (1.0 - prev))
p_disease_given_pos = (sens * prev) / p_pos
print(f"Bayes Theorem: P(Disease | Positive) = {p_disease_given_pos:.4f} ({p_disease_given_pos*100:.2f}%)")

# 3. Monte Carlo simulation with 100,000 patients
N = 100_000
has_disease = np.random.rand(N) < prev
test_pos = np.zeros(N, dtype=bool)

# Sens for diseased, (1-spec) for healthy
test_pos[has_disease] = np.random.rand(np.sum(has_disease)) < sens
test_pos[~has_disease] = np.random.rand(np.sum(~has_disease)) < p_pos_given_no_d

empirical_pos_prob = np.sum(has_disease & test_pos) / np.sum(test_pos)
print(f"Empirical Simulation (N={N}): P(Disease | Positive) = {empirical_pos_prob:.4f} ({empirical_pos_prob*100:.2f}%)")
print(f"Absolute discrepancy: {abs(p_disease_given_pos - empirical_pos_prob):.6f}")

# 4. Posterior vs Prevalence sweep
prevalences = np.linspace(0.001, 0.20, 200)
posteriors = [(sens * p) / ((sens * p) + (p_pos_given_no_d * (1.0 - p))) for p in prevalences]

# 1. Samples from distributions
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# Bernoulli
bern_samples = stats.bernoulli.rvs(p=0.4, size=1000)
axes[0, 0].hist(bern_samples, bins=[-0.5, 0.5, 1.5], rwidth=0.6, density=True, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Bernoulli (p=0.4)')

# Binomial
binom_samples = stats.binom.rvs(n=20, p=0.5, size=1000)
x_b = np.arange(0, 21)
axes[0, 1].hist(binom_samples, bins=20, density=True, alpha=0.6, color='salmon')
axes[0, 1].plot(x_b, stats.binom.pmf(x_b, 20, 0.5), 'ro-', ms=4)
axes[0, 1].set_title('Binomial (n=20, p=0.5)')

# Poisson
pois_samples = stats.poisson.rvs(mu=4, size=1000)
x_p = np.arange(0, 12)
axes[0, 2].hist(pois_samples, bins=12, density=True, alpha=0.6, color='lightgreen')
axes[0, 2].plot(x_p, stats.poisson.pmf(x_p, 4), 'go-', ms=4)
axes[0, 2].set_title('Poisson (mu=4)')

# Uniform & Gaussian
norm_samples = stats.norm.rvs(loc=0, scale=1, size=1000)
x_g = np.linspace(-3, 3, 100)
axes[1, 0].hist(norm_samples, bins=30, density=True, alpha=0.6, color='mediumpurple')
axes[1, 0].plot(x_g, stats.norm.pdf(x_g, 0, 1), 'm-', lw=2)
axes[1, 0].set_title('Gaussian (mu=0, sigma=1)')

# Uniform
unif_samples = stats.uniform.rvs(loc=0, scale=10, size=1000)
axes[1, 1].hist(unif_samples, bins=25, density=True, alpha=0.6, color='orange')
axes[1, 1].set_title('Uniform [0, 10]')

# Posterior vs prevalence
axes[1, 2].plot(prevalences * 100, np.array(posteriors) * 100, 'b-', lw=2.5)
axes[1, 2].axvline(prev * 100, color='r', linestyle='--', label=f'Baseline ({prev*100}%)')
axes[1, 2].scatter([prev*100], [p_disease_given_pos*100], color='red', zorder=5)
axes[1, 2].set_xlabel('Prevalence (%)')
axes[1, 2].set_ylabel('P(Disease | Positive) (%)')
axes[1, 2].set_title('Posterior vs Prevalence Curve')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp03_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
