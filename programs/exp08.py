"""Experiment 08: Vector Norms and Distance Measures.
Tasks:
1. Create learner feature vectors and calculate L1, L2 and infinity norms.
2. Compute Euclidean, Manhattan and cosine distances between learners.
3. Compare distance values before and after feature standardisation.
4. Identify the closest learner pair using each distance measure.
5. Plot a 2-D projection and annotate distance comparisons.
"""
from pathlib import Path
import numpy as np
import scipy.spatial.distance as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Experiment 08: Vector Norms & Distance Measures ===")

learners = ['Learner_A', 'Learner_B', 'Learner_C', 'Learner_D', 'Learner_E']
# Features: [Time_Spent (hrs), Lessons_Done, Quiz_Attempts, Avg_Score (%)]
raw_data = np.array([
    [45.0, 15.0, 20.0, 88.0],
    [50.0, 18.0, 22.0, 92.0],
    [10.0,  3.0,  5.0, 45.0],
    [38.0, 14.0, 19.0, 80.0],
    [12.0,  4.0,  6.0, 50.0]
], dtype=np.float32)

# 1. Norms
for name, v in zip(learners, raw_data):
    l1 = np.linalg.norm(v, ord=1)
    l2 = np.linalg.norm(v, ord=2)
    linf = np.linalg.norm(v, ord=np.inf)
    print(f"{name:9s} | L1: {l1:6.1f} | L2: {l2:6.2f} | L_inf: {linf:5.1f}")

# 2. Distances (Raw)
euc_raw = dist.squareform(dist.pdist(raw_data, metric='euclidean'))
man_raw = dist.squareform(dist.pdist(raw_data, metric='cityblock'))
cos_raw = dist.squareform(dist.pdist(raw_data, metric='cosine'))

# 3. Standardised distances
std_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0)
euc_std = dist.squareform(dist.pdist(std_data, metric='euclidean'))

# 4. Closest pairs
def get_closest_pair(matrix, labels):
    mat = matrix.copy()
    np.fill_diagonal(mat, np.inf)
    idx = np.unravel_index(np.argmin(mat), mat.shape)
    return labels[idx[0]], labels[idx[1]], mat[idx]

p_euc = get_closest_pair(euc_std, learners)
p_cos = get_closest_pair(cos_raw, learners)
print(f"\nClosest pair by Euclidean (Std): {p_euc[0]} & {p_euc[1]} (dist: {p_euc[2]:.4f})")
print(f"Closest pair by Cosine Distance: {p_cos[0]} & {p_cos[1]} (dist: {p_cos[2]:.4f})")

# 5. Plot 2D projection
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(euc_std, cmap='plasma')
axes[0].set_xticks(range(len(learners)))
axes[0].set_xticklabels(learners, rotation=30)
axes[0].set_yticks(range(len(learners)))
axes[0].set_yticklabels(learners)
axes[0].set_title('Standardised Euclidean Distance Matrix')
for i in range(len(learners)):
    for j in range(len(learners)):
        axes[0].text(j, i, f"{euc_std[i, j]:.2f}", ha='center', va='center', color='white' if euc_std[i, j] < 2 else 'black')
fig.colorbar(im, ax=axes[0])

# 2D scatter of Time Spent vs Score
axes[1].scatter(raw_data[:, 0], raw_data[:, 3], color='dodgerblue', s=80, edgecolors='black')
for i, name in enumerate(learners):
    axes[1].annotate(name, (raw_data[i, 0] + 1, raw_data[i, 3] - 1), fontsize=9)
# Draw connection between closest pair
i1 = learners.index(p_euc[0])
i2 = learners.index(p_euc[1])
axes[1].plot([raw_data[i1, 0], raw_data[i2, 0]], [raw_data[i1, 3], raw_data[i2, 3]], 'r--', lw=2, label='Closest Pair')
axes[1].set_xlabel('Time Spent (Hours)')
axes[1].set_ylabel('Average Quiz Score (%)')
axes[1].set_title('Learner Profiles & Similarity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp08_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
