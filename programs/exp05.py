"""Experiment 05: Bias–Variance, Overfitting and Underfitting Analysis.
Tasks:
1. Fit polynomial regression of degree 1, 3, 9, and 15 with Scikit-learn.
2. Plot each fitted curve over the data.
3. Plot training and validation error against model complexity.
4. Estimate bias^2 and variance through repeated resampling.
5. Select best degree and justify it.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Generate true nonlinear dataset
n = 150
X = np.sort(np.random.uniform(0, 1, n))
y_true = np.sin(1.5 * np.pi * X)
y = y_true + np.random.normal(0, 0.2, n)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
degrees = [1, 3, 9, 15]

print("=== Experiment 05: Bias-Variance & Model Complexity ===")
train_errors, val_errors = [], []
fitted_models = {}

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train[:, None], y_train)
    fitted_models[d] = model
    tr_err = mean_squared_error(y_train, model.predict(X_train[:, None]))
    val_err = mean_squared_error(y_val, model.predict(X_val[:, None]))
    train_errors.append(tr_err)
    val_errors.append(val_err)
    print(f"Degree {d:2d} | Train MSE: {tr_err:.4f} | Val MSE: {val_err:.4f}")

# 4. Bias^2 and Variance estimation via 100 resamples
n_boots = 100
X_eval = np.linspace(0, 1, 50)
y_eval_true = np.sin(1.5 * np.pi * X_eval)
bias2_list, var_list = [], []

for d in degrees:
    preds = np.zeros((n_boots, len(X_eval)))
    for b in range(n_boots):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        m.fit(X[idx, None], y[idx])
        preds[b] = m.predict(X_eval[:, None])
    mean_pred = np.mean(preds, axis=0)
    b2 = np.mean((mean_pred - y_eval_true)**2)
    var = np.mean(np.var(preds, axis=0))
    bias2_list.append(b2)
    var_list.append(var)
    print(f"Degree {d:2d} -> Bias^2: {b2:.4f}, Variance: {var:.4f}, Total Risk: {b2+var:.4f}")

best_deg = degrees[np.argmin(val_errors)]
print(f"\nOptimal Model: Degree {best_deg} (Minimizes validation MSE and balances Bias-Variance)")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

X_plot = np.linspace(0, 1, 200)
axes[0].scatter(X, y, color='gray', alpha=0.5, s=15, label='Data')
axes[0].plot(X_plot, np.sin(1.5 * np.pi * X_plot), 'k--', label='True function')
colors = ['red', 'green', 'blue', 'orange']
for d, c in zip(degrees, colors):
    axes[0].plot(X_plot, fitted_models[d].predict(X_plot[:, None]), label=f'Deg {d}', color=c, lw=1.8)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_title('Polynomial Fits vs Complexity')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(degrees, train_errors, 'bo-', label='Train MSE')
axes[1].plot(degrees, val_errors, 'ro-', label='Val MSE')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('Overfitting & Underfitting Curves')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp05_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
