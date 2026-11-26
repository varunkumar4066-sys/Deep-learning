"""Experiment 24: Softmax Classification and Cross-Entropy Loss.
Tasks:
1. Create a labelled three-class dataset and encode target labels.
2. Implement softmax function and verify probabilities sum to one.
3. Calculate categorical and sparse categorical cross-entropy.
4. Train small neural classifier and record loss across epochs.
5. Compare predicted class probabilities with true labels.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 24: Softmax & Cross-Entropy Loss ===")

# 1. Three-class dataset (Normal, Warning, Critical)
X, y = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
y_cat = tf.keras.utils.to_categorical(y, 3)

# 2. Softmax function verification
def softmax_fn(logits):
    exp_z = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

test_logits = np.array([[2.0, 1.0, 0.1], [0.5, 3.2, 1.1]])
probs = softmax_fn(test_logits)
print("Softmax Output:\n", np.round(probs, 4))
print("Sum of Probabilities per row:", np.sum(probs, axis=1), "AllClose to 1.0:", np.allclose(np.sum(probs, axis=1), 1.0))

# 4. Train classifier
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y_cat, epochs=35, batch_size=16, verbose=0)

loss_final, acc_final = model.evaluate(X, y_cat, verbose=0)
print(f"Final Model Cross-Entropy Loss: {loss_final:.4f} | Accuracy: {acc_final*100:.2f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(history.history['loss'], 'r-', lw=2, label='Categorical Cross-Entropy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss Value')
axes[0].set_title('Cross-Entropy Loss Minimization')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predicted probabilities distribution for 5 sample points
sample_probs = model.predict(X[:5], verbose=0)
x_bar = np.arange(5)
width = 0.25
axes[1].bar(x_bar - width, sample_probs[:, 0], width, label='Normal (0)', color='seagreen')
axes[1].bar(x_bar, sample_probs[:, 1], width, label='Warning (1)', color='orange')
axes[1].bar(x_bar + width, sample_probs[:, 2], width, label='Critical (2)', color='crimson')
axes[1].set_xticks(x_bar)
axes[1].set_xticklabels([f"Pt {i+1} (True {y[i]})" for i in range(5)])
axes[1].set_ylabel('Softmax Probability')
axes[1].set_title('Predicted Probabilities vs Ground Truth')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp24_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
