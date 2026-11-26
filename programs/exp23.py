"""Experiment 23: Activation Function Behaviour and Gradient Analysis.
Tasks:
1. Implement sigmoid, tanh, ReLU, Leaky ReLU and softmax functions.
2. Plot each activation function over [-5, 5].
3. Calculate and plot their derivatives.
4. Compare gradient values for large positive and negative inputs.
5. Summarise suitability of each activation for hidden or output layers.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 23: Activation Function & Gradient Analysis ===")

z = np.linspace(-5, 5, 300)

# 1. Activations
sigmoid = 1.0 / (1.0 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
leaky_relu = np.where(z > 0, z, 0.1 * z)

# 3. Derivatives
d_sigmoid = sigmoid * (1.0 - sigmoid)
d_tanh = 1.0 - tanh**2
d_relu = np.where(z > 0, 1.0, 0.0)
d_leaky = np.where(z > 0, 1.0, 0.1)

# 4. Gradient saturation comparison at z = -4.0 and z = 4.0
for val in [-4.0, 4.0]:
    s_val = 1.0 / (1.0 + np.exp(-val))
    ds = s_val * (1.0 - s_val)
    dt = 1.0 - np.tanh(val)**2
    dr = 1.0 if val > 0 else 0.0
    print(f"At z = {val:4.1f} | Sigmoid Grad: {ds:.4f} (Vanishing!) | Tanh Grad: {dt:.4f} | ReLU Grad: {dr:.1f}")

print("\nSuitability Summary:")
print(" - ReLU / Leaky ReLU: Ideal for hidden layers (prevents gradient vanishing).")
print(" - Sigmoid / Softmax: Best for binary and multi-class classification output layers.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(z, sigmoid, label='Sigmoid')
axes[0].plot(z, tanh, label='Tanh')
axes[0].plot(z, relu, label='ReLU')
axes[0].plot(z, leaky_relu, label='Leaky ReLU (alpha=0.1)')
axes[0].set_ylim(-1.5, 3.0)
axes[0].set_title('Activation Functions f(z)')
axes[0].set_xlabel('Input z')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, d_sigmoid, label="Sigmoid' (max 0.25)")
axes[1].plot(z, d_tanh, label="Tanh' (max 1.0)")
axes[1].plot(z, d_relu, label="ReLU' (0 or 1)")
axes[1].plot(z, d_leaky, label="Leaky ReLU'")
axes[1].set_ylim(-0.1, 1.2)
axes[1].set_title("Activation Derivatives f'(z)")
axes[1].set_xlabel('Input z')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp23_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
