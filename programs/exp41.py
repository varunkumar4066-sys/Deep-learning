"""Experiment 41: Many-to-One RNN for Text Classification.
Tasks:
1. Preprocess text and encode article categories (sports, business, tech, health).
2. Convert articles into padded input sequences.
3. Build an embedding and SimpleRNN classification model.
4. Train model and plot training and validation curves.
5. Evaluate class-wise performance using confusion matrix and F1-score.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 41: Many-to-One RNN Classification ===")

# Synthetic article dataset (4 categories)
categories = ['Sports', 'Business', 'Tech', 'Health']
vocab_size = 60
seq_len = 12
n_samples = 400

X = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
y = np.random.randint(0, 4, size=n_samples)
y_cat = tf.keras.utils.to_categorical(y, 4)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len,)),
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X, y_cat, epochs=12, batch_size=16, validation_split=0.25, verbose=0)

preds = np.argmax(model.predict(X[-100:], verbose=0), axis=1)
y_true = y[-100:]
cm = confusion_matrix(y_true, preds)
macro_f1 = f1_score(y_true, preds, average='macro')
print(f"Many-to-One RNN Macro F1-Score: {macro_f1:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(h.history['loss'], label='Train Loss')
axes[0].plot(h.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Classification Learning Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(categories, rotation=20)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(categories)
axes[1].set_title('Category Confusion Matrix')
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp41_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
