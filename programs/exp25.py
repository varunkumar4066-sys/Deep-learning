"""Experiment 25: Weight Initialisation Strategies.
Tasks:
1. Build a multilayer network with fixed architecture (e.g. 5 layers).
2. Initialise weights using Zeros, Random Normal, Xavier and He methods.
3. Record activation distributions at each hidden layer.
4. Compare gradient magnitudes and training convergence.
5. Recommend an initialisation method for the selected activation functions.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 25: Weight Initialisation Strategies ===")

# Forward pass through 5 deep layers (100 units each)
n_in = 100
n_hidden = 100
depth = 5
X = np.random.randn(1000, n_in)

methods = {
    'Zeros': lambda din, dout: np.zeros((din, dout)),
    'Normal (std=1.0)': lambda din, dout: np.random.randn(din, dout) * 1.0,
    'Xavier/Glorot': lambda din, dout: np.random.randn(din, dout) * np.sqrt(1.0 / din),
    'He/Kaiming': lambda din, dout: np.random.randn(din, dout) * np.sqrt(2.0 / din)
}

layer_acts = {}

for name, init_fn in methods.items():
    a = X
    acts = []
    for l in range(depth):
        W = init_fn(a.shape[1], n_hidden)
        z = a @ W
        a = np.maximum(0, z) # ReLU
        acts.append(a)
    layer_acts[name] = acts
    print(f"Method: {name:18s} | Layer 5 Mean: {np.mean(acts[-1]):.4f} | Layer 5 Std: {np.std(acts[-1]):.4f}")

print("\nRecommendation: He/Kaiming initialisation prevents variance collapse with ReLU activations.")

# Plot layer 5 activation distributions
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for i, (name, acts) in enumerate(layer_acts.items()):
    last_act = acts[-1].flatten()
    axes[i].hist(last_act, bins=30, range=(0, 5), color='royalblue', density=True)
    axes[i].set_title(f'{name}\nStd: {np.std(last_act):.2f}')
    axes[i].set_xlabel('Layer 5 Activation')
    axes[i].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp25_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
