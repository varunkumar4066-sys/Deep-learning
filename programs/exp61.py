"""Experiment 61: Bayesian Network Conditional Probability Tables.
Tasks:
1. Define variables and directed dependencies in a Bayesian network.
2. Create conditional probability distributions for all nodes.
3. Build and validate network structure.
4. Visualise directed graph and inspect probability tables.
5. Identify conditional independencies implied by graph.
"""
from pathlib import Path
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 61: Bayesian Network CPTs ===")

# Maintenance Network:
# Overheat (O), CoolingFailure (C) -> Vibration (V), HighTemp (H)
# Vibration (V), HighTemp (H) -> Alarm (A)
G = nx.DiGraph([
    ('Overheat', 'HighTemp'),
    ('CoolingFailure', 'HighTemp'),
    ('Overheat', 'Vibration'),
    ('HighTemp', 'Alarm'),
    ('Vibration', 'Alarm')
])

print(f"Network Nodes: {list(G.nodes)}")
print(f"Network Edges: {list(G.edges)}")

# Conditional Probability Table for Alarm given HighTemp, Vibration
cpt_alarm = pd.DataFrame({
    'HighTemp': [0, 0, 1, 1],
    'Vibration': [0, 1, 0, 1],
    'P(Alarm=0)': [0.99, 0.40, 0.30, 0.05],
    'P(Alarm=1)': [0.01, 0.60, 0.70, 0.95]
})
print("\nConditional Probability Table for Alarm P(A | H, V):\n", cpt_alarm)

# Plot DAG
fig, ax = plt.subplots(figsize=(7, 4.5))
pos = {
    'Overheat': (0, 2), 'CoolingFailure': (2, 2),
    'HighTemp': (0.5, 1), 'Vibration': (1.5, 1),
    'Alarm': (1, 0)
}
nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10, arrows=True, arrowsize=18)
ax.set_title('Machine Maintenance Bayesian Network')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp61_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
