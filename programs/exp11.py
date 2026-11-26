"""Experiment 11: Feature Scaling and Numerical Conditioning.
Tasks:
1. Generate vehicle dataset containing features with vastly different scales.
2. Apply min-max scaling, standardisation and robust scaling.
3. Compare feature distributions before and after transformation.
4. Train regression model on each representation and record convergence.
5. Recommend a scaling method based on observed behaviour.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import SGDRegressor

np.random.seed(42)

# 1. Dataset: Mileage (10k-150k), Engine (800-4000), Age (1-15)
n = 300
mileage = np.random.uniform(10_000, 150_000, n)
engine = np.random.uniform(800, 4000, n)
age = np.random.uniform(1, 15, n)
# Add some outliers to mileage
mileage[:5] = [280_000, 320_000, 350_000, 400_000, 450_000]

# Target Price ($)
price = 45000 - 0.15 * mileage + 5.0 * engine - 1200 * age + np.random.normal(0, 2000, n)
X_raw = np.column_stack([mileage, engine, age])

print("=== Experiment 11: Feature Scaling & Conditioning ===")
print("Condition Number of X^T X (Raw unscaled):", f"{np.linalg.cond(X_raw.T @ X_raw):.2e}")

# 2. Scaling transformations
scalers = {
    'Raw (Unscaled)': X_raw,
    'MinMax': MinMaxScaler().fit_transform(X_raw),
    'Standard': StandardScaler().fit_transform(X_raw),
    'Robust': RobustScaler().fit_transform(X_raw)
}

# 4. Train SGDRegressor on each and compare convergence
epochs = 30
losses = {}
for name, X_s in scalers.items():
    cond = np.linalg.cond(X_s.T @ X_s)
    model = SGDRegressor(max_iter=1, tol=None, warm_start=True, random_state=42, learning_rate='constant', eta0=1e-3 if name != 'Raw (Unscaled)' else 1e-12)
    history = []
    y_scaled = (price - price.mean()) / price.std()
    for _ in range(epochs):
        model.partial_fit(X_s, y_scaled)
        preds = model.predict(X_s)
        history.append(np.mean((y_scaled - preds)**2))
    losses[name] = history
    print(f"Scaler: {name:15s} | Condition No: {cond:8.2f} | Final Loss: {history[-1]:.4f}")

print("\nRecommendation: StandardScaler or RobustScaler provides optimal conditioning and rapid GD convergence.")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].boxplot([scalers['Raw (Unscaled)'][:, 0]/1000, scalers['Standard'][:, 0], scalers['Robust'][:, 0]], 
                tick_labels=['Mileage/1k', 'Standard', 'Robust'])
axes[0].set_title('Feature Distribution After Scaling')
axes[0].grid(True, alpha=0.3)

for name in ['MinMax', 'Standard', 'Robust']:
    axes[1].plot(losses[name], label=name, lw=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('Optimization Convergence by Scaler')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp11_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
