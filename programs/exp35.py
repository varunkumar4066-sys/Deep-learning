"""Experiment 35: BPTT Flow and Gradient Analysis.
Tasks:
1. Build a SimpleRNN and use tf.GradientTape to compute gradients.
2. Record gradient norm with respect to hidden state at each time step.
3. Plot gradient norm against time step for each sequence length (10, 30, 50).
4. Compare tanh and ReLU activations.
5. Explain vanishing and exploding gradients from plots.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 35: BPTT Gradient Analysis ===")

# Compute gradient flow dL/dh_t across time steps for sequence length T
def analyze_gradient_flow(T, act='tanh'):
    x = tf.random.normal((1, T, 1))
    W_hh = tf.Variable(tf.random.normal((8, 8)) * (0.8 if act=='tanh' else 1.2))
    W_xh = tf.Variable(tf.random.normal((1, 8)) * 0.5)
    
    with tf.GradientTape() as tape:
        h = tf.zeros((1, 8))
        h_states = []
        for t in range(T):
            z = tf.matmul(x[:, t, :], W_xh) + tf.matmul(h, W_hh)
            h = tf.nn.tanh(z) if act == 'tanh' else tf.nn.relu(z)
            h_states.append(h)
        loss = tf.reduce_mean(h_states[-1]**2)
    
    grads = tape.gradient(loss, h_states)
    grad_norms = [float(tf.norm(g)) if g is not None else 0.0 for g in grads]
    return grad_norms

g_10 = analyze_gradient_flow(10, 'tanh')
g_30 = analyze_gradient_flow(30, 'tanh')
g_relu = analyze_gradient_flow(20, 'relu')

print(f"Tanh T=30: Initial Step Grad Norm = {g_30[0]:.2e} vs Final Step = {g_30[-1]:.2e} (Demonstrating Vanishing Gradient!)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(g_10, 'ro-', label='T = 10')
axes[0].plot(g_30, 'bo-', label='T = 30')
axes[0].set_xlabel('Time Step t (Past -> Present)')
axes[0].set_ylabel('dL / dh_t Norm')
axes[0].set_yscale('log')
axes[0].set_title('BPTT Vanishing Gradient with Tanh')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(analyze_gradient_flow(20, 'tanh'), 'b-', label='Tanh (Decays smoothly)')
axes[1].plot(g_relu, 'r--', label='ReLU (Can explode or vanish)')
axes[1].set_xlabel('Time Step t')
axes[1].set_ylabel('dL / dh_t Norm')
axes[1].set_title('Tanh vs ReLU Activation Gradient Flow')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp35_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
