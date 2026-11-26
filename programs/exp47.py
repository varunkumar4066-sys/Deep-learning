"""Experiment 47: Attention Mechanism for Sequence Modelling.
Tasks:
1. Prepare a small sequence-to-sequence dataset.
2. Build an encoder and retain hidden states at each position.
3. Calculate attention scores and normalised attention weights.
4. Use weighted context vector in decoder.
5. Visualise attention weights as a heatmap.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

print("=== Experiment 47: Attention Mechanism Visualisation ===")

# Compute Bahdanau / Dot-product Attention weights between Encoder states and Decoder query
T_enc = 6 # Source length
T_dec = 6 # Target length
d_model = 16

# Simulated encoder states H_enc (T_enc, d_model)
H_enc = np.random.randn(T_enc, d_model)
# Simulated decoder states S_dec (T_dec, d_model)
S_dec = np.random.randn(T_dec, d_model)

# Dot-product attention scores: Score(i, j) = S_dec[i] @ H_enc[j].T / sqrt(d)
scores = (S_dec @ H_enc.T) / np.sqrt(d_model)

# Softmax over source positions (axis=1)
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

print("Attention Weights Matrix (Decoder Steps x Encoder Steps):\n", np.round(attention_weights, 3))
print("Sum of Attention across encoder tokens for each decoder step:", np.sum(attention_weights, axis=1))

# Plot Attention Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(attention_weights, cmap='YlGnBu')
ax.set_xticks(range(T_enc))
ax.set_xticklabels([f"Source_{j+1}" for j in range(T_enc)])
ax.set_yticks(range(T_dec))
ax.set_yticklabels([f"Target_{i+1}" for i in range(T_dec)])
ax.set_xlabel('Encoder Positions (Source Input)')
ax.set_ylabel('Decoder Positions (Target Generated)')
ax.set_title('Attention Alignment Matrix Heatmap')
fig.colorbar(im, ax=ax)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp47_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
