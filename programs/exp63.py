"""Experiment 63: Markov Random Field Construction.
Tasks:
1. Create a small grid of image pixels with observed intensity values.
2. Define unary and pairwise potential functions.
3. Construct neighbourhood graph for image grid.
4. Calculate energy of selected label configurations.
5. Visualise graph and compare configurations with different energy values.
"""
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 63: Markov Random Field (MRF) Grid ===")

# 4x4 image grid
grid_size = 4
G = nx.grid_2d_graph(grid_size, grid_size)

# Unary energy (agreement with noisy observation) + Pairwise energy (smoothness)
def mrf_energy(y_labels, x_obs, beta=1.5):
    unary = np.sum((y_labels - x_obs)**2)
    pairwise = 0.0
    for u, v in G.edges():
        if y_labels[u] != y_labels[v]:
            pairwise += beta
    return unary + pairwise

# Noisy observed grid (binary 0/1)
np.random.seed(42)
clean_grid = np.zeros((4, 4))
clean_grid[:2, :] = 1.0 # Top half 1, bottom half 0
noisy_grid = clean_grid.copy()
noisy_grid[1, 1] = 0.0 # flipped noise

clean_energy = mrf_energy(clean_grid, noisy_grid)
noisy_energy = mrf_energy(noisy_grid, noisy_grid)

print(f"Clean Configuration Energy: {clean_energy:.2f} (Low Energy = Higher Probability)")
print(f"Noisy Configuration Energy: {noisy_energy:.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

pos = {(r, c): (c, -r) for r in range(grid_size) for c in range(grid_size)}
nx.draw(G, pos, ax=axes[0], node_color=['steelblue' if clean_grid[r, c]==1 else 'orange' for r, c in G.nodes()], 
        node_size=600, with_labels=False)
axes[0].set_title('4x4 4-Connected MRF Lattice')

axes[1].imshow(clean_grid, cmap='bwr')
axes[1].set_title(f'Smoothed State (Energy: {clean_energy:.1f})')
axes[1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp63_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
