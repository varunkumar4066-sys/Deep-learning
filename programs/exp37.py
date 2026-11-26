"""Experiment 37: RNN Gradient Analysis and Gradient Clipping.
Tasks:
1. Train an RNN without clipping and log global gradient norm each step.
2. Identify steps where gradient spikes.
3. Retrain with clipnorm at different thresholds (0.5, 1.0, 5.0).
4. Plot gradient norm and loss curves for each setting.
5. Recommend a clipping threshold.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 37: Gradient Clipping in RNNs ===")

# Create synthetic sequence with high learning rate to trigger exploding gradients
X = np.random.randn(100, 20, 1).astype(np.float32)
y = np.sum(X[:, :5, 0], axis=1, keepdims=True)

# Train step with and without clipnorm
def train_with_clip(clip_val=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, 1)),
        tf.keras.layers.SimpleRNN(32, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.08, clipnorm=clip_val)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    loss_hist, grad_norm_hist = [], []
    for step in range(30):
        with tf.GradientTape() as tape:
            preds = model(X, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        gnorm = float(tf.linalg.global_norm(grads))
        grad_norm_hist.append(gnorm)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        loss_hist.append(float(loss))
    return loss_hist, grad_norm_hist

loss_unclipped, g_unclipped = train_with_clip(None)
loss_clip1, g_clip1 = train_with_clip(1.0)
loss_clip5, g_clip5 = train_with_clip(5.0)

print(f"Unclipped Max Gradient Norm: {max(g_unclipped):.2f} | Final Loss: {loss_unclipped[-1]:.4f}")
print(f"Clipped (1.0) Max Norm:      {max(g_clip1):.2f} | Final Loss: {loss_clip1[-1]:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(g_unclipped, 'r--', label='Unclipped Norm')
axes[0].plot(g_clip1, 'g-', label='Clipped (norm=1.0)')
axes[0].plot(g_clip5, 'b-.', label='Clipped (norm=5.0)')
axes[0].set_xlabel('Iteration Step')
axes[0].set_ylabel('Global Gradient Norm')
axes[0].set_title('Gradient Norm Stabilization')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_unclipped, 'r--', label='Unclipped Loss')
axes[1].plot(loss_clip1, 'g-', label='Clipped (1.0) Loss')
axes[1].plot(loss_clip5, 'b-.', label='Clipped (5.0) Loss')
axes[1].set_xlabel('Iteration Step')
axes[1].set_ylabel('Loss Value')
axes[1].set_title('Training Stability Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp37_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
