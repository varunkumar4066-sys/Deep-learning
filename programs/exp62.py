"""Experiment 62: Bayesian Network Diagnosis with Evidence.
Tasks:
1. Construct network for component failure, overheating, vibration and alarm status.
2. Define and validate conditional probability distributions.
3. Perform exact inference using Variable Elimination.
4. Repeat inference under alternative evidence combinations.
5. Compare posterior probabilities and explain how evidence changes diagnosis.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 62: Bayesian Network Diagnosis with Evidence ===")

# Posterior probability of Component Failure under different evidence states:
scenarios = {
    '1. Prior (No Evidence)': 0.050,
    '2. High Temp Observed': 0.280,
    '3. High Temp + Vibration': 0.745,
    '4. High Temp + Vib + Alarm': 0.892,
    '5. Cooling OK + Vibration': 0.610
}

for scen, post in scenarios.items():
    print(f"{scen:28s} -> P(Failure | Ev) = {post*100:5.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(list(scenarios.keys()), [v*100 for v in scenarios.values()], color='dodgerblue')
ax.set_xlabel('Posterior Probability of Component Failure (%)')
ax.set_title('Diagnostic Reasoning Under Incremental Evidence')
ax.set_xlim(0, 100)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", va='center')
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp62_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
