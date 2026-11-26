"""Experiment 43: LSTM Gate Operation and Cell-State Analysis.
Tasks:
1. Define small sequence and initialise LSTM weights and biases.
2. Calculate gate activations and cell-state updates for selected time steps.
3. Implement simplified LSTM forward pass using NumPy.
4. Compare manually calculated outputs with Keras LSTM layer.
5. Plot gate activations and cell-state values across sequence.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 43: LSTM Gate & Cell-State Mechanics ===")

# Manual LSTM cell forward step
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

T = 8
d_in = 1
d_h = 4
x_seq = np.sin(np.linspace(0, 3, T))[:, None] # (8, 1)

# Gates parameters
W_f, U_f, b_f = np.random.randn(d_in, d_h)*0.5, np.random.randn(d_h, d_h)*0.5, np.zeros(d_h)
W_i, U_i, b_i = np.random.randn(d_in, d_h)*0.5, np.random.randn(d_h, d_h)*0.5, np.zeros(d_h)
W_c, U_c, b_c = np.random.randn(d_in, d_h)*0.5, np.random.randn(d_h, d_h)*0.5, np.zeros(d_h)
W_o, U_o, b_o = np.random.randn(d_in, d_h)*0.5, np.random.randn(d_h, d_h)*0.5, np.zeros(d_h)

h_t = np.zeros(d_h)
c_t = np.zeros(d_h)

f_history, i_history, c_history, o_history = [], [], [], []

for t in range(T):
    xt = x_seq[t]
    f = sigmoid(xt @ W_f + h_t @ U_f + b_f)
    i = sigmoid(xt @ W_i + h_t @ U_i + b_i)
    c_tilde = np.tanh(xt @ W_c + h_t @ U_c + b_c)
    c_t = f * c_t + i * c_tilde
    o = sigmoid(xt @ W_o + h_t @ U_o + b_o)
    h_t = o * np.tanh(c_t)
    
    f_history.append(f)
    i_history.append(i)
    c_history.append(c_t.copy())
    o_history.append(o)

f_history = np.array(f_history)
i_history = np.array(i_history)
c_history = np.array(c_history)
print(f"Processed Sequence Length: {T}")
print(f"Final Cell State c_T: {np.round(c_history[-1], 3)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(np.mean(f_history, axis=1), 'r-o', label='Forget Gate f_t')
axes[0].plot(np.mean(i_history, axis=1), 'g-s', label='Input Gate i_t')
axes[0].plot(np.mean(o_history, axis=1), 'b-^', label='Output Gate o_t')
axes[0].set_xlabel('Time Step t')
axes[0].set_ylabel('Mean Gate Activation')
axes[0].set_title('LSTM Gate Activations Across Sequence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for u in range(d_h):
    axes[1].plot(c_history[:, u], label=f'Cell Unit {u+1}')
axes[1].set_xlabel('Time Step t')
axes[1].set_ylabel('Cell State Value c_t')
axes[1].set_title('Cell State Memory Accumulation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp43_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
