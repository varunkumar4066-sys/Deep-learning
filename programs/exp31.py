"""Experiment 31: CNN Feature Extraction and Visualisation.
Tasks:
1. Train a CNN for a small image-classification dataset.
2. Create intermediate model that outputs selected convolutional layers.
3. Extract feature maps for representative input images.
4. Visualise activation maps from early, middle and deeper layers.
5. Discuss differences between low-level and higher-level features.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 31: CNN Feature Visualization ===")

digits = load_digits()
X = digits.images[..., None] / 16.0
y = tf.keras.utils.to_categorical(digits.target, 10)

# Build 3-stage CNN
inputs = tf.keras.layers.Input(shape=(8, 8, 1))
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='early_conv')(inputs)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='mid_conv')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='deep_conv')(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Intermediate extractor
extractor = tf.keras.Model(
    inputs=inputs,
    outputs=[model.get_layer('early_conv').output, model.get_layer('mid_conv').output, model.get_layer('deep_conv').output]
)

sample = X[0:1] # digit 0
early_maps, mid_maps, deep_maps = extractor.predict(sample, verbose=0)
print(f"Early Maps Shape: {early_maps.shape} | Mid Maps Shape: {mid_maps.shape} | Deep Maps Shape: {deep_maps.shape}")

# Plot
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))

axes[0].imshow(sample[0, :, :, 0], cmap='gray')
axes[0].set_title('Input Image (Digit 0)')
axes[0].axis('off')

axes[1].imshow(early_maps[0, :, :, 0], cmap='magma')
axes[1].set_title('Early: Edges & Gradients')
axes[1].axis('off')

axes[2].imshow(mid_maps[0, :, :, 0], cmap='magma')
axes[2].set_title('Mid: Shapes & Curves')
axes[2].axis('off')

axes[3].imshow(deep_maps[0, :, :, 0], cmap='magma')
axes[3].set_title('Deep: High-Level Semantics')
axes[3].axis('off')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp31_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
