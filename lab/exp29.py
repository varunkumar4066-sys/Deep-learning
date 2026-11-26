"""Experiment 29: VAE / GAN Sample Generation."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish,tf_setup
tf=tf_setup();from sklearn.datasets import load_digits
X=load_digits().data.astype('float32')/16.;dataset=tf.data.Dataset.from_tensor_slices(X).shuffle(len(X),seed=42).batch(64)
class VAE(tf.keras.Model):
 def __init__(self):
  super().__init__();self.enc=tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(16)]);self.dec=tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(64,activation='sigmoid')])
 def call(self,x,training=False):
  mean,logvar=tf.split(self.enc(x),2,axis=1);z=mean+tf.exp(.5*logvar)*tf.random.normal(tf.shape(mean));return self.dec(z),mean,logvar
vae=VAE();opt=tf.keras.optimizers.Adam(.002)
for epoch in range(12):
 for batch in dataset:
  with tf.GradientTape() as tape:
   rec,mu,lv=vae(batch,training=True);loss=tf.reduce_mean(tf.reduce_sum(tf.square(batch-rec),axis=1))+.02*tf.reduce_mean(tf.reduce_sum(mu**2+tf.exp(lv)-lv-1,axis=1))
  grad=tape.gradient(loss,vae.trainable_variables);opt.apply_gradients(zip(grad,vae.trainable_variables))
print('VAE final loss',float(loss));vae_samples=vae.dec(tf.random.normal((8,8))).numpy()
# Basic GAN trained on the same 8x8 digits.
G=tf.keras.Sequential([tf.keras.layers.Input((16,)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dense(64,activation='sigmoid')]);D=tf.keras.Sequential([tf.keras.layers.Input((64,)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(1,activation='sigmoid')]);go=tf.keras.optimizers.Adam(.001);do=tf.keras.optimizers.Adam(.001);bce=tf.keras.losses.BinaryCrossentropy()
for epoch in range(12):
 for batch in dataset:
  z=tf.random.normal((tf.shape(batch)[0],16))
  with tf.GradientTape() as td,tf.GradientTape() as tg:
   fake=G(z,training=True);real_prob=D(batch,training=True);fake_prob=D(fake,training=True)
   dloss=bce(tf.ones_like(real_prob)*.9,real_prob)+bce(tf.zeros_like(fake_prob),fake_prob);gloss=bce(tf.ones_like(fake_prob),fake_prob)
  dg=td.gradient(dloss,D.trainable_variables);gg=tg.gradient(gloss,G.trainable_variables);do.apply_gradients(zip(dg,D.trainable_variables));go.apply_gradients(zip(gg,G.trainable_variables))
gan_samples=G(tf.random.normal((8,16)),training=False).numpy();print('GAN final discriminator/generator loss',float(dloss),float(gloss))
fig,ax=plt.subplots(2,8,figsize=(12,4))
for i in range(8):
 ax[0,i].imshow(vae_samples[i].reshape(8,8),cmap='gray',vmin=0,vmax=1);ax[1,i].imshow(gan_samples[i].reshape(8,8),cmap='gray',vmin=0,vmax=1)
 for j in range(2):ax[j,i].axis('off')
ax[0,0].set_title('VAE samples');ax[1,0].set_title('GAN samples');finish('exp29_vae_gan')
