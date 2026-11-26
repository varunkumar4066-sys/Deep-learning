"""Experiment 04: Entropy, Cross-Entropy and KL Divergence.
Tasks:
1. Compute entropy of observed weather distribution.
2. Compute cross-entropy and KL divergence of each model against true distribution.
3. Verify that cross-entropy = entropy + KL divergence.
4. Show that KL divergence is asymmetric by computing it both ways.
5. Plot binary cross-entropy loss against predicted probability.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 04: Entropy, Cross-Entropy & KL Divergence ===")

# True weather distribution [Sunny, Cloudy, Rainy]
p = np.array([0.60, 0.30, 0.10])
q1 = np.array([0.50, 0.35, 0.15])  # Model 1 (close)
q2 = np.array([0.20, 0.40, 0.40])  # Model 2 (divergent)

# 1. Entropy
def entropy(prob):
    return -np.sum(prob * np.log2(prob + 1e-12))

def cross_entropy(p, q):
    return -np.sum(p * np.log2(q + 1e-12))

def kl_divergence(p, q):
    return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

H_p = entropy(p)
print(f"Observed Weather Entropy H(p): {H_p:.4f} bits")

# 2. Cross-entropy & KL divergence
for name, q in [("Model 1", q1), ("Model 2", q2)]:
    H_pq = cross_entropy(p, q)
    D_kl = kl_divergence(p, q)
    print(f"\n{name}:")
    print(f"  Cross-Entropy H(p, q): {H_pq:.4f} bits")
    print(f"  KL Divergence D_KL(p || q): {D_kl:.4f} bits")
    # 3. Verify H(p, q) = H(p) + D_KL(p || q)
    print(f"  H(p) + D_KL: {H_p + D_kl:.4f} bits | Match: {np.isclose(H_pq, H_p + D_kl)}")

# 4. Asymmetry of KL
kl_pq = kl_divergence(p, q1)
kl_qp = kl_divergence(q1, p)
print(f"\nKL Asymmetry: D_KL(p || q1) = {kl_pq:.4f} != D_KL(q1 || p) = {kl_qp:.4f}")

# 5. Plot binary cross-entropy
y_hat = np.linspace(0.001, 0.999, 200)
bce_y1 = -np.log2(y_hat)
bce_y0 = -np.log2(1.0 - y_hat)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].bar(['Sunny', 'Cloudy', 'Rainy'], p, alpha=0.7, label='True p')
axes[0].plot(['Sunny', 'Cloudy', 'Rainy'], q1, 'ro--', label='Model 1 q1')
axes[0].plot(['Sunny', 'Cloudy', 'Rainy'], q2, 'gs--', label='Model 2 q2')
axes[0].set_ylabel('Probability')
axes[0].set_title('Weather Distributions Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_hat, bce_y1, 'b-', label='Loss when true y = 1 (-log2 p)')
axes[1].plot(y_hat, bce_y0, 'r-', label='Loss when true y = 0 (-log2(1-p))')
axes[1].set_xlabel('Predicted Probability y_hat')
axes[1].set_ylabel('Binary Cross-Entropy (bits)')
axes[1].set_title('BCE Loss vs Predicted Probability')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp04_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
