"""Experiment 21: CNN for Image Classification.
Tasks:
1. Load, resize and normalise images; split into train, val and test sets.
2. Build a CNN with convolution, pooling, flatten and dense layers.
3. Train with categorical cross-entropy and plot accuracy and loss curves.
4. Show confusion matrix and sample misclassifications.
5. Visualise feature maps of first convolution layer.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 21: CNN for Image Classification ===")

# 1. Dataset: 8x8 Digits
digits = load_digits()
X = digits.images[..., None] / 16.0 # Normalise [0, 1]
y = tf.keras.utils.to_categorical(digits.target, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build CNN
conv_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8, 8, 1)),
    conv_layer,
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=0)

# 4. Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
preds = model.predict(X_test, verbose=0)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(preds, axis=1))
print(f"CNN Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

# 5. Extract feature maps for first test image
feat_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('conv1').output)
sample_img = X_test[0:1]
feat_maps = feat_extractor.predict(sample_img, verbose=0)[0] # 8x8x16

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(history.history['accuracy'], label='Train Acc')
axes[0].plot(history.history['val_accuracy'], label='Val Acc')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title(f'CNN Learning Curves (Test Acc: {test_acc*100:.1f}%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# First 4 feature maps in a 2x2 grid
axes[1].imshow(feat_maps[:, :, 0], cmap='viridis')
axes[1].set_title('First Layer Conv Feature Map (Filter 0)')
axes[1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp21_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
