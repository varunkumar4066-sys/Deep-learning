"""Experiment 13: Conditional Probability and Joint Distributions.
Tasks:
1. Create a dataset containing symptom indicators and diagnostic outcomes.
2. Construct a joint probability table and calculate marginal probabilities.
3. Calculate conditional probabilities for selected evidence.
4. Verify multiplication and total-probability rules.
5. Visualise conditional probabilities for different patient groups.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 13: Conditional Probability & Joint Distributions ===")

# 1. Dataset: 1000 patient records
n = 1000
# True conditions: 0: Healthy (70%), 1: Flu (20%), 2: COVID (10%)
disease_labels = ['Healthy', 'Flu', 'COVID']
diseases = np.random.choice([0, 1, 2], size=n, p=[0.70, 0.20, 0.10])

# Symptoms: Fever (P(Fever|Healthy)=0.05, P(Fever|Flu)=0.75, P(Fever|COVID)=0.85)
p_fever = [0.05, 0.75, 0.85]
fever = np.array([np.random.rand() < p_fever[d] for d in diseases], dtype=int)
fever_labels = ['No Fever', 'Fever']

df = pd.DataFrame({'Disease': [disease_labels[d] for d in diseases], 'Fever': [fever_labels[f] for f in fever]})

# 2. Joint & Marginal tables
joint_counts = pd.crosstab(df['Disease'], df['Fever'])
joint_prob = joint_counts / n
print("Joint Probability Table P(Disease, Fever):\n", np.round(joint_prob, 4))

p_disease = joint_prob.sum(axis=1)
p_fever_marginal = joint_prob.sum(axis=0)
print("\nMarginal P(Disease):\n", np.round(p_disease, 4))
print("\nMarginal P(Fever):\n", np.round(p_fever_marginal, 4))

# 3. Conditional Probabilities P(Disease | Fever=Present)
p_disease_given_fever = joint_prob['Fever'] / p_fever_marginal['Fever']
print("\nConditional Probability P(Disease | Fever):\n", np.round(p_disease_given_fever, 4))

# 4. Verify total probability rule
p_fever_total = sum(p_fever[i] * p_disease[disease_labels[i]] for i in range(3))
print(f"Total Probability Check: Marginal P(Fever) = {p_fever_marginal['Fever']:.4f} vs Reconstructed = {p_fever_total:.4f}")

# 5. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(joint_prob, cmap='Blues')
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(['No Fever', 'Fever'])
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(joint_prob.index)
axes[0].set_title('Joint Probability P(Disease, Symptom)')
for i in range(3):
    for j in range(2):
        axes[0].text(j, i, f"{joint_prob.iloc[i, j]:.3f}", ha='center', va='center')
fig.colorbar(im, ax=axes[0])

bars = axes[1].bar(disease_labels, p_disease_given_fever, color=['seagreen', 'coral', 'crimson'])
axes[1].set_ylabel('Posterior Probability P(Disease | Fever)')
axes[1].set_title('Diagnosis Given Symptom (Fever)')
axes[1].set_ylim(0, 1)
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp13_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
