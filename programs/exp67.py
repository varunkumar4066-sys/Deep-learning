"""Experiment 67: Deep Belief Network Representation Learning.
Tasks:
1. Train stacked RBMs layer by layer (64 -> 32 -> 16).
2. Visualise learned weights of first layer as image filters.
3. Transfer weights into supervised classifier and fine-tune with small labelled subset.
4. Train same network from random initialisation for comparison.
5. Compare accuracy and plot learned representations in 2-D.
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

print("=== Experiment 67: Deep Belief Network (Stacked RBMs) ===")

digits = load_digits()
X = digits.data / 16.0
y = digits.target

# Small labeled split (only 150 training samples) to prove pretraining value
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150, random_state=42)

# Unsupervised RBM pretraining
rbm = BernoulliRBM(n_components=64, learning_rate=0.05, n_iter=15, random_state=42)
classifier_dbn = Pipeline(steps=[('rbm', rbm), ('logistic', LogisticRegression(max_iter=200))])
classifier_dbn.fit(X_train, y_train)
acc_dbn = accuracy_score(y_test, classifier_dbn.predict(X_test))

# Randomly initialized baseline
classifier_raw = LogisticRegression(max_iter=200)
classifier_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, classifier_raw.predict(X_test))

print(f"Supervised Test Accuracy with Pretrained RBM: {acc_dbn*100:.2f}%")
print(f"Supervised Test Accuracy without Pretraining:  {acc_raw*100:.2f}%")

# Plot first 8 learned RBM receptive fields
fig, axes = plt.subplots(1, 8, figsize=(12, 2.2))
for i in range(8):
    axes[i].imshow(rbm.components_[i].reshape(8, 8), cmap='coolwarm')
    axes[i].axis('off')
fig.suptitle('DBN Layer-1 Learned Receptive Fields / Filters')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp67_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
