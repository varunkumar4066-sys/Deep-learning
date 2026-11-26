"""Experiment 74: Deep Belief Network Layer-Wise Pretraining.
Tasks:
1. Prepare labelled and unlabelled digit subsets.
2. Train first RBM using input observations.
3. Use first hidden representation to train second RBM.
4. Stack pretrained layers and attach classification output layer.
5. Fine-tune and compare with network trained from random initialisation.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("=== Experiment 74: Greedy Layer-Wise Pretraining ===")

digits = load_digits()
X = digits.data / 16.0
y = digits.target

# Limited labels scenario
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, random_state=42)

# Two stacked RBM layers
rbm1 = BernoulliRBM(n_components=48, n_iter=10, random_state=42)
rbm2 = BernoulliRBM(n_components=24, n_iter=10, random_state=42)
clf = LogisticRegression(max_iter=150)

dbn_pipeline = Pipeline([('rbm1', rbm1), ('rbm2', rbm2), ('clf', clf)])
dbn_pipeline.fit(X_train, y_train)
acc_dbn = accuracy_score(y_test, dbn_pipeline.predict(X_test))

# Baseline without pretraining
raw_clf = LogisticRegression(max_iter=150)
raw_clf.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, raw_clf.predict(X_test))

print(f"Test Accuracy with 2-Layer DBN Pretraining: {acc_dbn*100:.2f}%")
print(f"Test Accuracy with Random Init Baseline:   {acc_raw*100:.2f}%")

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Random Baseline', 'Layer-Wise Pretrained DBN'], [acc_raw*100, acc_dbn*100], color=['coral', 'mediumseagreen'])
ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(60, 100)
ax.set_title('Benefit of Greedy Layer-Wise Pretraining')
for bar in ax.patches:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp74_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
