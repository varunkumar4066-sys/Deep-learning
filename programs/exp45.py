"""Experiment 45: Bidirectional RNN for Sequence Classification.
Tasks:
1. Prepare and tokenise a labelled text dataset.
2. Build a unidirectional RNN baseline.
3. Build a bidirectional RNN using same recurrent cell and comparable units.
4. Train and evaluate both models using identical data splits.
5. Compare accuracy, F1-score, parameter count and training time.
"""
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 45: Bidirectional RNN Classification ===")

n_samples = 400
seq_len = 15
vocab_size = 50

X = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
# Target label depends on BOTH prefix and suffix
y = ((X[:, 0] > 25) ^ (X[:, -1] > 25)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Unidirectional
m_uni = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len,)),
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
m_uni.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t0 = time.time()
h_uni = m_uni.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, verbose=0)
time_uni = time.time() - t0
preds_uni = (m_uni.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
acc_uni = accuracy_score(y_test, preds_uni)
f1_uni = f1_score(y_test, preds_uni)

# Bidirectional
m_bi = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len,)),
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
m_bi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t1 = time.time()
h_bi = m_bi.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, verbose=0)
time_bi = time.time() - t1
preds_bi = (m_bi.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
acc_bi = accuracy_score(y_test, preds_bi)
f1_bi = f1_score(y_test, preds_bi)

print(f"Unidirectional RNN | Acc: {acc_uni*100:.2f}% | F1: {f1_uni:.4f} | Time: {time_uni:.2f}s")
print(f"Bidirectional RNN  | Acc: {acc_bi*100:.2f}% | F1: {f1_bi:.4f} | Time: {time_bi:.2f}s")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(h_uni.history['val_accuracy'], 'r--', label=f'Unidirectional (Acc: {acc_uni*100:.1f}%)')
ax.plot(h_bi.history['val_accuracy'], 'g-', label=f'Bidirectional (Acc: {acc_bi*100:.1f}%)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Unidirectional vs Bidirectional Recurrent Performance')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp45_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
