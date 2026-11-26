"""Experiment 69: VAE/GAN Sample Generation.
Tasks:
1. Build a VAE with encoder, reparameterisation and decoder; train with reconstruction + KL loss.
2. Generate images by sampling latent space; plot a 2-D latent grid.
3. Build a simple GAN with generator and discriminator.
4. Train GAN and save generated images.
5. Plot generator and discriminator loss curves.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

print("=== Experiment 69: VAE / GAN Sample Generation ===")

digits = load_digits()
X = digits.data.astype('float32') / 16.0
dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X), seed=42).batch(64)

# 1. Variational Autoencoder (VAE)
class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.enc = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        self.dec = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='sigmoid')
        ])
    def call(self, x, training=False):
        mean, logvar = tf.split(self.enc(x), 2, axis=1)
        z = mean + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mean))
        return self.dec(z), mean, logvar

vae = VAE()
opt_vae = tf.keras.optimizers.Adam(0.003)

for epoch in range(10):
    for batch in dataset:
        with tf.GradientTape() as tape:
            rec, mu, lv = vae(batch, training=True)
            loss_rec = tf.reduce_mean(tf.reduce_sum(tf.square(batch - rec), axis=1))
            loss_kl = 0.02 * tf.reduce_mean(tf.reduce_sum(mu**2 + tf.exp(lv) - lv - 1, axis=1))
            loss = loss_rec + loss_kl
        grad = tape.gradient(loss, vae.trainable_variables)
        opt_vae.apply_gradients(zip(grad, vae.trainable_variables))

vae_samples = vae.dec(tf.random.normal((4, 8))).numpy()
print(f"VAE Trained. Final Loss: {float(loss):.4f}")

# 2. Generative Adversarial Network (GAN)
G = tf.keras.Sequential([
    tf.keras.layers.Input((16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid')
])
D = tf.keras.Sequential([
    tf.keras.layers.Input((64,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
opt_g = tf.keras.optimizers.Adam(0.002)
opt_d = tf.keras.optimizers.Adam(0.002)
bce = tf.keras.losses.BinaryCrossentropy()

for epoch in range(10):
    for batch in dataset:
        z = tf.random.normal((tf.shape(batch)[0], 16))
        with tf.GradientTape() as td, tf.GradientTape() as tg:
            fake = G(z, training=True)
            real_prob = D(batch, training=True)
            fake_prob = D(fake, training=True)
            dloss = bce(tf.ones_like(real_prob)*0.9, real_prob) + bce(tf.zeros_like(fake_prob), fake_prob)
            gloss = bce(tf.ones_like(fake_prob), fake_prob)
        dg = td.gradient(dloss, D.trainable_variables)
        gg = tg.gradient(gloss, G.trainable_variables)
        opt_d.apply_gradients(zip(dg, D.trainable_variables))
        opt_g.apply_gradients(zip(gg, G.trainable_variables))

gan_samples = G(tf.random.normal((4, 16)), training=False).numpy()
print(f"GAN Trained. Final D Loss: {float(dloss):.4f} | Final G Loss: {float(gloss):.4f}")

# Plot
fig, axes = plt.subplots(2, 4, figsize=(9, 4.5))
for i in range(4):
    axes[0, i].imshow(vae_samples[i].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[1, i].imshow(gan_samples[i].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')

axes[0, 0].set_title('VAE Samples', loc='left')
axes[1, 0].set_title('GAN Samples', loc='left')

out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / "exp69_plot.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path.name}")
