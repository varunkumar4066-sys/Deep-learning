"""Experiment 22: CNN Architecture and Hyperparameter Comparison.
Tasks:
1. Vary number of convolution layers (2, 3).
2. Vary filter size (3x3, 5x5).
3. Compare max pooling and average pooling with batch normalisation.
4. Vary learning rate and batch size.
5. Tabulate accuracy, parameter count and training time.
"""
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 22: CNN Hyperparameter Comparison ===")

digits = load_digits()
X = digits.images[..., None] / 16.0
y = tf.keras.utils.to_categorical(digits.target, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

configs = [
    {'name': '2-Conv (3x3, MaxPool)', 'layers': 2, 'k': 3, 'pool': 'max', 'bn': False},
    {'name': '3-Conv (3x3, MaxPool)', 'layers': 3, 'k': 3, 'pool': 'max', 'bn': False},
    {'name': '2-Conv (5x5, AvgPool)', 'layers': 2, 'k': 5, 'pool': 'avg', 'bn': False},
    {'name': '2-Conv (3x3, MaxPool+BN)', 'layers': 2, 'k': 3, 'pool': 'max', 'bn': True}
]

results = []

for cfg in configs:
    t0 = time.time()
    layers = [tf.keras.layers.Input(shape=(8, 8, 1))]
    for i in range(cfg['layers']):
        layers.append(tf.keras.layers.Conv2D(16 * (i+1), (cfg['k'], cfg['k']), padding='same', activation='relu'))
        if cfg['bn']:
            layers.append(tf.keras.layers.BatchNormalization())
        if i == 0:
            if cfg['pool'] == 'max':
                layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
            else:
                layers.append(tf.keras.layers.AveragePooling2D((2, 2)))
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    dur = time.time() - t0
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    params = model.count_params()
    results.append({'name': cfg['name'], 'acc': acc, 'params': params, 'time': dur})
    print(f"Config: {cfg['name']:25s} | Acc: {acc*100:5.2f}% | Params: {params:6d} | Time: {dur:.2f}s")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

names = [r['name'] for r in results]
accs = [r['acc'] * 100 for r in results]
params = [r['params'] for r in results]

axes[0].barh(names, accs, color='steelblue')
axes[0].set_xlabel('Test Accuracy (%)')
axes[0].set_title('CNN Architecture Accuracy')
axes[0].set_xlim(85, 100)
axes[0].grid(True, alpha=0.3)

axes[1].barh(names, params, color='coral')
axes[1].set_xlabel('Trainable Parameter Count')
axes[1].set_title('Model Complexity Comparison')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp22_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
