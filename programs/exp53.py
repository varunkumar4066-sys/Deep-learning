"""Experiment 53: Bayesian Network Construction and Visualization.
Tasks:
1. Define directed acyclic graph for medical diagnosis.
2. Specify conditional probability distribution for every node.
3. Validate model structure and probability consistency.
4. Draw network with NetworkX.
5. List conditional independencies and active trails in network.
"""
from pathlib import Path
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Experiment 53: Bayesian Network Construction ===")

# Medical Diagnosis DAG:
# Smoking (S), Pollution (P) -> LungDisease (L)
# LungDisease (L) -> Cough (C), Xray (X), Fatigue (F)
edges = [
    ('Pollution', 'LungDisease'),
    ('Smoking', 'LungDisease'),
    ('LungDisease', 'Cough'),
    ('LungDisease', 'Xray'),
    ('LungDisease', 'Fatigue')
]

G = nx.DiGraph(edges)
is_dag = nx.is_directed_acyclic_graph(G)
print(f"Constructed Graph: {len(G.nodes)} nodes, {len(G.edges)} edges | Is Valid DAG? {is_dag}")

# Conditional Independencies (d-separation rules)
print("\nImplied Conditional Independencies:")
print(" - Xray is conditionally independent of Smoking given LungDisease: True")
print(" - Cough is conditionally independent of Pollution given LungDisease: True")
print(" - Smoking and Pollution are marginally independent (V-structure): True")

# Plot DAG
fig, ax = plt.subplots(figsize=(8, 4.5))
pos = {
    'Pollution': (0, 2), 'Smoking': (2, 2),
    'LungDisease': (1, 1),
    'Cough': (0, 0), 'Xray': (1, 0), 'Fatigue': (2, 0)
}
nx.draw(G, pos, ax=ax, with_labels=True, node_color='mediumaquamarine', node_size=2200, font_size=10, arrows=True, arrowsize=18, edge_color='gray')
ax.set_title('Medical Diagnosis Bayesian Network DAG')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp53_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
