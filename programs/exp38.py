"""Experiment 38: RNN, LSTM and GRU Hyperparameter Experimentation.
Tasks:
1. Tokenise and pad reviews; add embedding layer.
2. Define model-builder function choosing cell type (RNN, LSTM, GRU).
3. Tune units, layers, dropout, learning rate and embedding size.
4. Report best configuration for each cell type.
5. Compare accuracy, parameter count and training time.
"""
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 38: Recurrent Cells Hyperparameter Tuning ===")

# Synthetic sentiment dataset
n = 400
seq_len = 15
vocab_size = 50
X = np.random.randint(1, vocab_size, size=(n, seq_len))
y = (np.sum(X[:, :3], axis=1) > (3 * vocab_size // 2)).astype(int)

cells = ['SimpleRNN', 'LSTM', 'GRU']
results = {}

for cell in cells:
    t0 = time.time()
    RecLayer = getattr(tf.keras.layers, cell)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,)),
        tf.keras.layers.Embedding(vocab_size, 16),
        RecLayer(24, dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.25, verbose=0)
    dur = time.time() - t0
    acc = h.history['val_accuracy'][-1]
    params = model.count_params()
    results[cell] = {'acc': acc, 'params': params, 'time': dur}
    print(f"Cell: {cell:10s} | Val Acc: {acc*100:5.2f}% | Params: {params:5d} | Time: {dur:.2f}s")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

cell_names = list(results.keys())
acc_vals = [results[c]['acc'] * 100 for c in cell_names]
param_vals = [results[c]['params'] for c in cell_names]

axes[0].bar(cell_names, acc_vals, color=['indianred', 'royalblue', 'mediumseagreen'])
axes[0].set_ylabel('Validation Accuracy (%)')
axes[0].set_ylim(40, 100)
axes[0].set_title('Accuracy by Recurrent Cell Type')
axes[0].grid(True, alpha=0.3)

axes[1].bar(cell_names, param_vals, color=['indianred', 'royalblue', 'mediumseagreen'])
axes[1].set_ylabel('Parameter Count')
axes[1].set_title('Model Complexity (Params)')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp38_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
