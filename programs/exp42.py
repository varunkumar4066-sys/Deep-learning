"""Experiment 42: Many-to-Many RNN Sequence Labelling.
Tasks:
1. Prepare a small token-level labelled sequence dataset.
2. Encode tokens and labels with consistent sequence lengths.
3. Build a recurrent model that returns output at every time step (TimeDistributed).
4. Train using sequence loss.
5. Compare predicted token labels with reference sequences.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 42: Many-to-Many Sequence Labelling ===")

# Token tags: 0: Other, 1: Person, 2: Location
n_samples = 200
seq_len = 8
vocab_size = 40
n_tags = 3

X = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
# Rule: If token id is divisible by 5 -> Tag 1 (Person), divisible by 7 -> Tag 2 (Loc), else 0
y = np.zeros((n_samples, seq_len), dtype=int)
y[X % 5 == 0] = 1
y[X % 7 == 0] = 2
y_cat = tf.keras.utils.to_categorical(y, n_tags)

# Architecture: return_sequences=True
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len,)),
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_tags, activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X, y_cat, epochs=12, batch_size=16, verbose=0)

loss_final, acc_final = model.evaluate(X, y_cat, verbose=0)
print(f"Sequence Token Tagging Accuracy: {acc_final*100:.2f}% | Loss: {loss_final:.4f}")

# Sample test comparison
sample_pred = np.argmax(model.predict(X[0:1], verbose=0)[0], axis=1)
print(f"Sample Input Tokens:   {X[0]}")
print(f"Sample True Tags:      {y[0]}")
print(f"Sample Predicted Tags: {sample_pred}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h.history['accuracy'], 'g-', lw=2, label='Token-Level Accuracy')
ax.plot(h.history['loss'], 'r--', lw=2, label='Sequence Cross-Entropy Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Metric Value')
ax.set_title('Many-to-Many RNN Token Labelling Training')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp42_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
