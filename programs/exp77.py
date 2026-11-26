"""Experiment 77: Variational Autoencoder Latent-Space Interpolation.
Tasks:
1. Prepare and normalise a suitable image dataset.
2. Build VAE encoder that outputs latent mean and log-variance.
3. Implement reparameterisation and decoder.
4. Select two latent representations and interpolate between them.
5. Visualise generated images along interpolation path.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 77: VAE Latent-Space Interpolation ===")

digits = load_digits()
X = digits.data / 16.0

# Build compact VAE
latent_dim = 2
enc_in = tf.keras.layers.Input(shape=(64,))
h = tf.keras.layers.Dense(32, activation='relu')(enc_in)
z_mean = tf.keras.layers.Dense(latent_dim)(h)
z_log_var = tf.keras.layers.Dense(latent_dim)(h)

def sampling(args):
    zm, zlv = args
    eps = tf.random.normal(shape=tf.shape(zm))
    return zm + tf.exp(0.5 * zlv) * eps

z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

dec_in = tf.keras.layers.Input(shape=(latent_dim,))
dh = tf.keras.layers.Dense(32, activation='relu')(dec_in)
x_out = tf.keras.layers.Dense(64, activation='sigmoid')(dh)
decoder = tf.keras.Model(dec_in, x_out)

vae_out = decoder(z)
vae = tf.keras.Model(enc_in, vae_out)
vae.compile(optimizer='adam', loss='binary_crossentropy')
vae.fit(X, X, epochs=15, batch_size=32, verbose=0)

# Encoder model to obtain latent representations
encoder = tf.keras.Model(enc_in, z_mean)
z_start = encoder.predict(X[0:1], verbose=0)[0] # Digit 0
z_end = encoder.predict(X[1:2], verbose=0)[0]   # Digit 1

# Interpolate alpha in [0, 1] across 6 steps
alphas = np.linspace(0, 1, 6)
interpolated_latents = np.array([(1 - a) * z_start + a * z_end for a in alphas])
morph_images = decoder.predict(interpolated_latents, verbose=0)

# Plot interpolation strip
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for i, a in enumerate(alphas):
    axes[i].imshow(morph_images[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f'alpha={a:.2f}')
    axes[i].axis('off')
fig.suptitle('Continuous VAE Latent Space Morphing')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp77_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
