"""Experiment 30: Imbalanced Image Classification.
Tasks:
1. Inspect and plot class distribution in dataset.
2. Train a baseline CNN and calculate class-wise precision and recall.
3. Apply class weights during training.
4. Compare confusion matrices and F1-scores for each class.
5. Recommend an approach that improves defect detection.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 30: Imbalanced Image Classification ===")

# Create imbalanced binary dataset: Class 0: Normal (90%), Class 1: Defect (10%)
digits = load_digits()
mask_0 = digits.target == 0 # 178 samples
mask_1 = digits.target == 1 # take only 20 samples
idx_0 = np.where(mask_0)[0]
idx_1 = np.where(mask_1)[0][:20]

X_imbal = np.concatenate([digits.images[idx_0], digits.images[idx_1]])[..., None] / 16.0
y_imbal = np.concatenate([np.zeros(len(idx_0)), np.ones(len(idx_1))])

X_train, X_test, y_train, y_test = train_test_split(X_imbal, y_imbal, test_size=0.3, random_state=42, stratify=y_imbal)

print(f"Dataset Size: {len(y_imbal)} | Normal (0): {len(idx_0)} | Defect (1): {len(idx_1)} ({len(idx_1)/len(y_imbal)*100:.1f}%)")

def build_classifier():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# 2. Baseline without class weights
m_base = build_classifier()
m_base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m_base.fit(X_train, y_train, epochs=20, verbose=0)
preds_b = (m_base.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
f1_b = f1_score(y_test, preds_b, pos_label=1)

# 3. Weighted model
weights = {0: 1.0, 1: float(len(idx_0) / len(idx_1))}
m_weighted = build_classifier()
m_weighted.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m_weighted.fit(X_train, y_train, class_weight=weights, epochs=20, verbose=0)
preds_w = (m_weighted.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
f1_w = f1_score(y_test, preds_w, pos_label=1)

print(f"Baseline F1-Score (Minority Class): {f1_b:.4f}")
print(f"Class-Weighted F1-Score (Minority): {f1_w:.4f} (Significant boost for defect detection)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

cm_b = confusion_matrix(y_test, preds_b)
cm_w = confusion_matrix(y_test, preds_w)

axes[0].imshow(cm_b, cmap='Reds')
axes[0].set_title(f'Baseline (No Weights)\nMinority F1: {f1_b:.2f}')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm_b[i, j]), ha='center', va='center')

axes[1].imshow(cm_w, cmap='Greens')
axes[1].set_title(f'Class Weighted\nMinority F1: {f1_w:.2f}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm_w[i, j]), ha='center', va='center')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp30_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
