"""Experiment 40: Word Embedding Visualisation.
Tasks:
1. Prepare and tokenise a small review corpus.
2. Create an embedding layer and train it as part of text model.
3. Extract learned word-embedding matrix.
4. Project selected word vectors into two dimensions using PCA.
5. Visualise and interpret similarities among selected words.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 40: Word Embedding Visualisation ===")

corpus = [
    "great movie loved it excellent acting",
    "terrible film awful plot boring",
    "loved the story wonderful performance",
    "waste of time bad acting",
    "brilliant direction beautiful visuals",
    "poor script horrible ending",
    "fantastic cinema truly enjoyed",
    "worst movie ever hated it"
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=40)
tokenizer.fit_on_texts(corpus)
seqs = tokenizer.texts_to_sequences(corpus)
X = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=6)

# Embedding dimension = 8
vocab_size = len(tokenizer.word_index) + 1
emb_dim = 8

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Embedding(vocab_size, emb_dim, name='embed'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=30, verbose=0)

# Extract embedding weights
weights = model.get_layer('embed').get_weights()[0]
print(f"Learned Embedding Matrix Shape: {weights.shape}")

# Project top words with PCA
selected_words = ['great', 'loved', 'excellent', 'terrible', 'awful', 'bad', 'worst']
word_vecs = []
valid_words = []
for w in selected_words:
    if w in tokenizer.word_index:
        idx = tokenizer.word_index[w]
        word_vecs.append(weights[idx])
        valid_words.append(w)

pca = PCA(n_components=2)
coords = pca.fit_transform(word_vecs)

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(coords[:, 0], coords[:, 1], color='darkblue', s=70)
for i, txt in enumerate(valid_words):
    color = 'green' if txt in ['great', 'loved', 'excellent'] else 'red'
    ax.annotate(txt, (coords[i, 0] + 0.02, coords[i, 1] + 0.02), fontsize=11, color=color, weight='bold')
ax.set_title('Word Embedding 2D PCA Projection (Semantic Clusters)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp40_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
