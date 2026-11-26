"""Experiment 19: Multilayer Perceptron Implementation.
Tasks:
1. Preprocess data: encode categorical fields and scale numeric fields.
2. Build an MLP with hidden layers using ReLU and a sigmoid output.
3. Train with binary cross-entropy and Adam; plot training and validation curves.
4. Report accuracy, precision, recall, F1 and confusion matrix.
5. Vary hidden-layer width and note effect.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 19: Multilayer Perceptron (MLP) ===")

# 1. Customer churn tabular dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Build MLP in Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=0)

# 4. Evaluation
preds = (model.predict(X_test, verbose=0) > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print(f"Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# 5. Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy')
axes[0].set_title('MLP Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Retained', 'Churn']); axes[1].set_yticklabels(['Retained', 'Churn'])
axes[1].set_title(f'Confusion Matrix (Acc={acc*100:.1f}%)')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp19_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
