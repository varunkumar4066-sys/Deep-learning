"""Experiment 39: Text Tokenisation and Sequence Padding.
Tasks:
1. Clean sample text dataset and inspect message lengths.
2. Create vocabulary and convert tokens into integer sequences.
3. Apply padding and truncation to selected sequence length.
4. Examine effect of vocabulary size and sequence length.
5. Visualise distribution of message lengths before and after padding.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

print("=== Experiment 39: Text Tokenisation & Sequence Padding ===")

texts = [
    "Order not delivered yet, please help",
    "How to cancel my subscription?",
    "Need refund for broken item",
    "App crashes when opening settings page on Android",
    "Thank you for quick support!",
    "Wrong product received in the mail yesterday",
    "Cannot reset password with email link",
    "Payment failed multiple times today"
]

# 1. Inspect raw lengths
word_counts = [len(t.split()) for t in texts]
print(f"Raw Message Lengths: {word_counts} | Min: {min(word_counts)}, Max: {max(word_counts)}")

# 2. Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
seqs = tokenizer.texts_to_sequences(texts)
print("\nWord Index Sample (First 5):", dict(list(tokenizer.word_index.items())[:5]))
print("Integer Sequences Sample (First 2):\n", seqs[:2])

# 3. Padding (maxlen = 6)
padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=6, padding='post', truncating='post')
print("\nPadded Sequences (maxlen=6):\n", padded)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(word_counts, bins=6, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Number of Messages')
axes[0].set_title('Raw Message Lengths')
axes[0].grid(True, alpha=0.3)

im = axes[1].imshow(padded, cmap='Blues', aspect='auto')
axes[1].set_xlabel('Sequence Token Position (1-6)')
axes[1].set_ylabel('Message Index (0-7)')
axes[1].set_title('Padded Sequence Matrix Representation')
fig.colorbar(im, ax=axes[1])

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp39_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
