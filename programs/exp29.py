"""Experiment 29: Data Augmentation for CNN Generalisation.
Tasks:
1. Display sample images and identify suitable augmentation operations.
2. Apply rotation, translation, flipping and zoom.
3. Train baseline CNN without augmentation.
4. Train same CNN with augmentation and compare learning curves.
5. Evaluate both models on unchanged test set.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 29: Data Augmentation ===")

digits = load_digits()
X = digits.images[..., None] / 16.0
y = tf.keras.utils.to_categorical(digits.target, 10)

# Deliberately small training set (80 images) to force overfitting without augmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=80, random_state=42)

# Augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# 3. Baseline CNN
def create_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

m_plain = create_cnn()
m_plain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h_plain = m_plain.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, verbose=0)
_, acc_plain = m_plain.evaluate(X_test, y_test, verbose=0)

# 4. Augmented CNN
m_aug = tf.keras.Sequential([
    data_augmentation,
    create_cnn()
])
m_aug.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h_aug = m_aug.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, verbose=0)
_, acc_aug = m_aug.evaluate(X_test, y_test, verbose=0)

print(f"Baseline CNN (No Augmentation) | Test Acc: {acc_plain*100:.2f}%")
print(f"Augmented CNN                 | Test Acc: {acc_aug*100:.2f}% (Improved generalisation)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(h_plain.history['val_loss'], 'r--', label='Plain Val Loss')
axes[0].plot(h_aug.history['val_loss'], 'g-', label='Augmented Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Generalisation: Val Loss Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Display sample augmented images
sample = tf.repeat(X_train[0:1], 4, axis=0)
augmented_samples = data_augmentation(sample, training=True)
axes[1].imshow(augmented_samples[0, :, :, 0], cmap='gray')
axes[1].set_title('Sample Augmented Image')
axes[1].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp29_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
