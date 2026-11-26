"""Experiment 20: L2, Dropout and Early Stopping Comparison.
Tasks:
1. Train a deliberately large baseline MLP and plot its overfitting.
2. Retrain with L2 regularisation at three lambda values.
3. Retrain with dropout at rates 0.2, 0.3 and 0.5.
4. Retrain with early stopping using validation loss.
5. Tabulate test accuracy and train–test gap for every model.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 20: Regularization (L2, Dropout, Early Stopping) ===")

# Small clinical dataset prone to overfitting
X, y = make_classification(n_samples=300, n_features=20, n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(reg=None, dropout=0.0):
    regularizer = tf.keras.regularizers.l2(reg) if reg else None
    layers = [tf.keras.layers.Input(shape=(20,))]
    layers.append(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer))
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# 1. Baseline (Overfitting)
m_base = build_model()
h_base = m_base.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, verbose=0)
tr_acc_b = h_base.history['accuracy'][-1]
te_acc_b = h_base.history['val_accuracy'][-1]

# 2. L2 Regularization (lambda = 0.01)
m_l2 = build_model(reg=0.01)
h_l2 = m_l2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, verbose=0)
tr_acc_l2 = h_l2.history['accuracy'][-1]
te_acc_l2 = h_l2.history['val_accuracy'][-1]

# 3. Dropout (rate = 0.3)
m_drop = build_model(dropout=0.3)
h_drop = m_drop.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, verbose=0)
tr_acc_d = h_drop.history['accuracy'][-1]
te_acc_d = h_drop.history['val_accuracy'][-1]

# 4. Early Stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
m_es = build_model()
h_es = m_es.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, callbacks=[es], verbose=0)
tr_acc_es = h_es.history['accuracy'][-1]
te_acc_es = h_es.history['val_accuracy'][-1]

print(f"Baseline        | Train Acc: {tr_acc_b:.4f} | Test Acc: {te_acc_b:.4f} | Gap: {tr_acc_b - te_acc_b:.4f}")
print(f"L2 (lambda=0.01)| Train Acc: {tr_acc_l2:.4f} | Test Acc: {te_acc_l2:.4f} | Gap: {tr_acc_l2 - te_acc_l2:.4f}")
print(f"Dropout (0.3)   | Train Acc: {tr_acc_d:.4f} | Test Acc: {te_acc_d:.4f} | Gap: {tr_acc_d - te_acc_d:.4f}")
print(f"Early Stopping  | Train Acc: {tr_acc_es:.4f} | Test Acc: {te_acc_es:.4f} | Gap: {tr_acc_es - te_acc_es:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(h_base.history['val_loss'], 'r--', label='Baseline Val Loss')
axes[0].plot(h_l2.history['val_loss'], 'g-', label='L2 Val Loss')
axes[0].plot(h_drop.history['val_loss'], 'b-', label='Dropout Val Loss')
axes[0].plot(h_es.history['val_loss'], 'm-', label='Early Stopping')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Regularisation Comparison on Val Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

models = ['Baseline', 'L2 Reg', 'Dropout', 'EarlyStop']
gaps = [tr_acc_b - te_acc_b, tr_acc_l2 - te_acc_l2, tr_acc_d - te_acc_d, tr_acc_es - te_acc_es]
axes[1].bar(models, gaps, color=['crimson', 'mediumseagreen', 'royalblue', 'purple'])
axes[1].set_ylabel('Train-Test Accuracy Gap')
axes[1].set_title('Overfitting Gap Reduction')
axes[1].grid(True, alpha=0.3)

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp20_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
