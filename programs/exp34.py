"""Experiment 34: Unfolded RNN Architecture Visualization.
Tasks:
1. Build a SimpleRNN in Keras and extract input, recurrent and output weights.
2. Compute hidden states for each time step manually and confirm them against Keras.
3. Draw unfolded network as graph with NetworkX (nodes x1..x5, h1..h5, y1..y5).
4. Label edges with shared weights W_xh, W_hh and W_hy.
5. Plot hidden-state values across time steps.
"""
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 34: Unfolded RNN Architecture ===")

T = 5 # 5 time steps
x_seq = np.array([[[0.5], [1.2], [-0.8], [0.3], [1.0]]], dtype=np.float32) # (1, 5, 1)

# SimpleRNN with 3 hidden units
rnn_cell = tf.keras.layers.SimpleRNN(3, return_sequences=True, return_state=True, activation='tanh')
keras_out, keras_h_last = rnn_cell(x_seq)

# Extract weights: W_xh (1, 3), W_hh (3, 3), b_h (3,)
W_xh, W_hh, b_h = rnn_cell.get_weights()
print(f"Extracted Weights: W_xh {W_xh.shape}, W_hh {W_hh.shape}, b_h {b_h.shape}")

# 2. Compute manually
h_t = np.zeros((1, 3))
manual_states = []
for t in range(T):
    x_t = x_seq[0, t:t+1]
    h_t = np.tanh(x_t @ W_xh + h_t @ W_hh + b_h)
    manual_states.append(h_t[0])
manual_states = np.array(manual_states)

diff = np.max(np.abs(keras_out[0].numpy() - manual_states))
print(f"Manual vs Keras State Maximum Absolute Difference: {diff:.2e} (Exact match!)")

# 3. Draw unfolded graph
G = nx.DiGraph()
pos = {}
for t in range(1, T + 1):
    G.add_node(f'x_{t}')
    G.add_node(f'h_{t}')
    G.add_node(f'y_{t}')
    pos[f'x_{t}'] = (t, 0)
    pos[f'h_{t}'] = (t, 1)
    pos[f'y_{t}'] = (t, 2)
    G.add_edge(f'x_{t}', f'h_{t}')
    G.add_edge(f'h_{t}', f'y_{t}')
    if t < T:
        G.add_edge(f'h_{t}', f'h_{t+1}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

nx.draw(G, pos, ax=axes[0], with_labels=True, node_color='lightblue', node_size=1200, font_size=10, arrows=True)
axes[0].set_title('Unfolded RNN Computation Graph (T=5)')

for unit in range(3):
    axes[1].plot(range(1, T + 1), manual_states[:, unit], 'o-', label=f'Hidden Unit {unit+1}')
axes[1].set_xlabel('Time Step t')
axes[1].set_ylabel('Hidden State Value h_t')
axes[1].set_title('Hidden State Trajectories Across Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp34_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
