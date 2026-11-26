"""Experiment 26: Batch Normalisation in Deep Neural Networks.
Tasks:
1. Prepare a classification dataset and build baseline MLP.
2. Build equivalent MLP with batch-normalisation layers.
3. Train both models using identical data splits and optimiser settings.
4. Compare loss curves, validation accuracy and training time.
5. Discuss observed effect of batch normalisation.
"""
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 26: Batch Normalisation ===")

X, y = make_classification(n_samples=1500, n_features=30, n_informative=15, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline without BN
m_base = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
m_base.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

# With BatchNorm
m_bn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
m_bn.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

t0 = time.time()
h_base = m_base.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=0)
time_base = time.time() - t0

t1 = time.time()
h_bn = m_bn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=0)
time_bn = time.time() - t1

print(f"Baseline | Test Acc: {h_base.history['val_accuracy'][-1]*100:.2f}% | Final Loss: {h_base.history['val_loss'][-1]:.4f}")
print(f"With BN  | Test Acc: {h_bn.history['val_accuracy'][-1]*100:.2f}% | Final Loss: {h_bn.history['val_loss'][-1]:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(h_base.history['val_loss'], 'r--', label='Baseline Val Loss')
axes[0].plot(h_bn.history['val_loss'], 'b-', label='BatchNorm Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Validation Loss Trajectory')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(h_base.history['val_accuracy'], 'r--', label='Baseline Val Acc')
axes[1].plot(h_bn.history['val_accuracy'], 'b-', label='BatchNorm Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy Convergence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp26_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
