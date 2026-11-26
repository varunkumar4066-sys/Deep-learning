"""Experiment 33: Sequential Dataset Preparation and RNN Prediction.
Tasks:
1. Load series with Pandas; handle missing values and scale it.
2. Create sliding windows (24 past hours -> next hour).
3. Split chronologically into train and test sets.
4. Build and train a SimpleRNN model in Keras.
5. Plot actual against predicted demand and report MAE and RMSE.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 33: Sequential Preparation & SimpleRNN ===")

# 1. Synthesize hourly power demand (365 days = 8760 hours, subsampled to 500 hours for fast run)
t = np.linspace(0, 50, 500)
demand = 50 + 20 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 2, 500)
df = pd.DataFrame({'Demand': demand})

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Demand']])

# 2. Sliding windows: lookback = 24
lookback = 24
X, y = [], []
for i in range(len(scaled) - lookback):
    X.append(scaled[i:i+lookback])
    y.append(scaled[i+lookback])
X, y = np.array(X), np.array(y)

# 3. Chronological split (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. SimpleRNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(lookback, 1)),
    tf.keras.layers.SimpleRNN(16, activation='tanh'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=12, batch_size=16, verbose=0)

# 5. Evaluate
preds = scaler.inverse_transform(model.predict(X_test, verbose=0))
actuals = scaler.inverse_transform(y_test)
mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
print(f"Test MAE: {mae:.2f} MW | Test RMSE: {rmse:.2f} MW")

# Plot
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(actuals, 'k-', label='Actual Demand', lw=1.5)
ax.plot(preds, 'r--', label='SimpleRNN Prediction', lw=1.5)
ax.set_xlabel('Time Step (Hours)')
ax.set_ylabel('Electricity Demand (MW)')
ax.set_title(f'Sequential Forecasting with SimpleRNN (MAE: {mae:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp33_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
