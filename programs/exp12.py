"""Experiment 12: Maximum Likelihood Estimation for Gaussian Data.
Tasks:
1. Generate Gaussian samples with known mean and variance.
2. Derive and implement maximum-likelihood estimates of mean and variance.
3. Compare estimated parameters for several sample sizes.
4. Plot likelihood surface over candidate parameter values.
5. Discuss difference between estimated and true parameters.
"""
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# True parameters
true_mu = 25.0
true_sigma = 4.0
true_var = true_sigma**2

print("=== Experiment 12: MLE for Gaussian Data ===")
print(f"True Parameters: mu = {true_mu}, sigma = {true_sigma}, var = {true_var}")

# 3. Compare sample sizes N
sizes = [10, 50, 200, 1000]
print("\nSample Size Analysis:")
for n in sizes:
    data = np.random.normal(true_mu, true_sigma, n)
    # MLE formulas
    mu_mle = np.mean(data)
    var_mle = np.mean((data - mu_mle)**2) # Biased MLE: 1/N sum
    var_unbiased = np.var(data, ddof=1)   # Unbiased: 1/(N-1) sum
    print(f"  N = {n:4d} | mu_mle: {mu_mle:.3f} | var_mle: {var_mle:.3f} | var_unbiased: {var_unbiased:.3f}")

# 4. Log-likelihood surface for N = 50
sample_50 = np.random.normal(true_mu, true_sigma, 50)
mu_grid = np.linspace(20, 30, 100)
sigma_grid = np.linspace(2, 6, 100)
M, S = np.meshgrid(mu_grid, sigma_grid)

# Log-likelihood L(mu, sigma) = -N ln(sqrt(2pi)*sigma) - sum((x-mu)^2)/(2 sigma^2)
N = len(sample_50)
log_lik = np.zeros_like(M)
for i in range(100):
    for j in range(100):
        log_lik[i, j] = np.sum(stats.norm.logpdf(sample_50, loc=M[i, j], scale=S[i, j]))

mle_mu_50 = np.mean(sample_50)
mle_sig_50 = np.sqrt(np.mean((sample_50 - mle_mu_50)**2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(sample_50, bins=12, density=True, alpha=0.6, color='steelblue', label='Sample N=50')
x_plot = np.linspace(10, 40, 200)
axes[0].plot(x_plot, stats.norm.pdf(x_plot, true_mu, true_sigma), 'r--', label='True PDF')
axes[0].plot(x_plot, stats.norm.pdf(x_plot, mle_mu_50, mle_sig_50), 'g-', label='Fitted MLE PDF')
axes[0].set_title('Fitted Gaussian Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

cp = axes[1].contourf(M, S, log_lik, levels=25, cmap='viridis')
fig.colorbar(cp, ax=axes[1])
axes[1].plot(true_mu, true_sigma, 'r*', ms=12, label='True (25, 4)')
axes[1].plot(mle_mu_50, mle_sig_50, 'wo', ms=8, label=f'MLE ({mle_mu_50:.2f}, {mle_sig_50:.2f})')
axes[1].set_xlabel('Candidate Mean (mu)')
axes[1].set_ylabel('Candidate Std (sigma)')
axes[1].set_title('Log-Likelihood Surface')
axes[1].legend()

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp12_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
