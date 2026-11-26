"""Experiment 48: Comparative Forecasting with RNN, LSTM and GRU.
Tasks:
1. Prepare chronological time-series dataset with common input windows.
2. Train SimpleRNN, LSTM and GRU models with comparable configurations.
3. Record training time and loss.
4. Calculate MAE, RMSE and MAPE on same test period.
5. Plot forecasts and tabulate comparative results.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 48: Comparative Forecasting (RNN, LSTM, GRU) ===")

t = np.linspace(0, 40, 500)
signal = np.sin(t) + 0.5 * np.cos(3*t) + 0.1 * t + np.random.normal(0, 0.1, 500)
signal = (signal - signal.mean()) / signal.std()

lookback = 20
X, y = [], []
for i in range(len(signal) - lookback):
    X.append(signal[i:i+lookback])
    y.append(signal[i+lookback])
X, y = np.array(X)[..., None], np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def train_model(layer_cls):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        layer_cls(32),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    t0 = time.time()
    m.fit(X_train, y_train, epochs=12, batch_size=16, verbose=0)
    dur = time.time() - t0
    preds = m.predict(X_test, verbose=0).ravel()
    return preds, dur

preds_rnn, t_rnn = train_model(tf.keras.layers.SimpleRNN)
preds_lstm, t_lstm = train_model(tf.keras.layers.LSTM)
preds_gru, t_gru = train_model(tf.keras.layers.GRU)

models = ['SimpleRNN', 'LSTM', 'GRU']
preds_all = [preds_rnn, preds_lstm, preds_gru]
times = [t_rnn, t_lstm, t_gru]

results = []
for name, p, dur in zip(models, preds_all, times):
    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'Time': dur})
    print(f"Model: {name:10s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Time: {dur:.2f}s")

# Plot
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(y_test[:80], 'k-', label='Actual Ground Truth', lw=2)
ax.plot(preds_rnn[:80], 'r--', label='SimpleRNN')
ax.plot(preds_lstm[:80], 'b-.', label='LSTM')
ax.plot(preds_gru[:80], 'g:', lw=2, label='GRU')
ax.set_xlabel('Test Time Steps')
ax.set_ylabel('Normalised Value')
ax.set_title('Comparative Forecasting Performance (RNN vs LSTM vs GRU)')
ax.legend()
ax.grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp48_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
