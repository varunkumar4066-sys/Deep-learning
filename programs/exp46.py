"""Experiment 46: Encoder–Decoder RNN for Sequence Generation.
Tasks:
1. Prepare paired input and target sequences.
2. Tokenise and pad encoder and decoder inputs.
3. Build an encoder that produces context representation.
4. Build a decoder that generates target sequence step by step.
5. Evaluate generated sequences against reference outputs.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 46: Encoder-Decoder Seq2Seq ===")

# Task: Sequence reversal task (e.g. [1, 2, 3, 4] -> [4, 3, 2, 1])
n_samples = 300
seq_len = 5
vocab_size = 20

X_enc = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
y_dec = X_enc[:, ::-1] # reversed
y_dec_cat = tf.keras.utils.to_categorical(y_dec, vocab_size)

# Encoder
enc_inputs = tf.keras.layers.Input(shape=(seq_len,))
enc_emb = tf.keras.layers.Embedding(vocab_size, 16)(enc_inputs)
_, state_h = tf.keras.layers.GRU(32, return_state=True)(enc_emb)

# Repeat context vector for decoder steps
repeated_context = tf.keras.layers.RepeatVector(seq_len)(state_h)
dec_out = tf.keras.layers.GRU(32, return_sequences=True)(repeated_context)
dec_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(dec_out)

model = tf.keras.Model(inputs=enc_inputs, outputs=dec_dense)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X_enc, y_dec_cat, epochs=25, batch_size=16, verbose=0)

loss_final, acc_final = model.evaluate(X_enc, y_dec_cat, verbose=0)
print(f"Seq2Seq Sequence Inversion Accuracy: {acc_final*100:.2f}% | Loss: {loss_final:.4f}")

# Sample prediction
sample = X_enc[0:1]
pred_seq = np.argmax(model.predict(sample, verbose=0)[0], axis=1)
print(f"Input Sequence:     {sample[0]}")
print(f"True Reversed:      {y_dec[0]}")
print(f"Predicted Reversed: {pred_seq}")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(h.history['accuracy'], 'b-', lw=2, label='Seq2Seq Training Accuracy')
ax.plot(h.history['loss'], 'r--', lw=2, label='Cross-Entropy Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Metric Value')
ax.set_title('Encoder-Decoder Seq2Seq Learning Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp46_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
