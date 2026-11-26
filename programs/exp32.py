"""Experiment 32: Model Evaluation and Error Analysis for CNNs.
Tasks:
1. Generate predictions for held-out test dataset.
2. Calculate accuracy, precision, recall, F1-score and confusion matrix.
3. Identify most frequently confused class pairs.
4. Display correctly classified and misclassified examples.
5. Prepare an error-analysis report with possible corrective actions.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 32: CNN Error Analysis ===")

digits = load_digits()
X = digits.images[..., None] / 16.0
y = digits.target
y_cat = tf.keras.utils.to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=12, batch_size=32, verbose=0)

preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, preds)

print("Classification Report:\n", classification_report(y_test, preds, digits=3))

# Find misclassified examples
misclassified_idx = np.where(preds != y_test)[0]
print(f"Total Test Samples: {len(y_test)} | Misclassified: {len(misclassified_idx)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('True Class')
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)

if len(misclassified_idx) > 0:
    m_i = misclassified_idx[0]
    axes[1].imshow(X_test[m_i, :, :, 0], cmap='gray')
    axes[1].set_title(f'Misclassified Sample:\nTrue={y_test[m_i]}, Pred={preds[m_i]}')
    axes[1].axis('off')
else:
    axes[1].text(0.5, 0.5, '100% Accuracy on Test Split', ha='center', va='center')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp32_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
