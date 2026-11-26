"""Experiment 30: Generative Model Output Comparison and Evaluation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from common import finish,tf_setup
tf=tf_setup();X=load_digits().data.astype('float32')/16.;rng=np.random.default_rng(42)
# Train two lightweight generative baselines: PCA Gaussian and VAE. The GAN is trained in Exp29.
pca=PCA(n_components=12,random_state=42).fit(X);latent=pca.transform(X);z=rng.multivariate_normal(latent.mean(0),np.cov(latent,rowvar=False),size=300);pca_samples=np.clip(pca.inverse_transform(z),0,1)
enc=tf.keras.Sequential([tf.keras.layers.Input((64,)),tf.keras.layers.Dense(40,activation='relu'),tf.keras.layers.Dense(12)]);dec=tf.keras.Sequential([tf.keras.layers.Input((6,)),tf.keras.layers.Dense(40,activation='relu'),tf.keras.layers.Dense(64,activation='sigmoid')]);opt=tf.keras.optimizers.Adam(.002)
data=tf.data.Dataset.from_tensor_slices(X).shuffle(len(X),seed=42).batch(64)
for epoch in range(15):
 for x in data:
  with tf.GradientTape() as tape:
   mu,lv=tf.split(enc(x),2,axis=1);sample=mu+tf.exp(.5*lv)*tf.random.normal(tf.shape(mu));rec=dec(sample)
   loss=tf.reduce_mean(tf.reduce_sum((rec-x)**2,axis=1))+.03*tf.reduce_mean(tf.reduce_sum(mu**2+tf.exp(lv)-lv-1,axis=1))
  params=enc.trainable_variables+dec.trainable_variables;grads=tape.gradient(loss,params);opt.apply_gradients(zip(grads,params))
vae_samples=dec(tf.random.normal((300,6)),training=False).numpy();reference=X[rng.choice(len(X),300,replace=False)]
# Proxy measures, NOT FID or IS: closest-sample distance, diversity, per-pixel mean difference.
rows=[]
for name,s in [('PCA-Gaussian',pca_samples),('VAE',vae_samples)]:
 nearest=pairwise_distances(s,reference).min(axis=1).mean();diversity=pairwise_distances(s[:100]).mean();mean_err=np.abs(s.mean(0)-reference.mean(0)).mean()
 rows.append((name,nearest,diversity,mean_err))
print(pd.DataFrame(rows,columns=['Model','Mean distance to real','Sample diversity','Pixel mean error']).round(4))
fig,ax=plt.subplots(2,6,figsize=(9,4))
for i in range(6):
 ax[0,i].imshow(pca_samples[i].reshape(8,8),cmap='gray',vmin=0,vmax=1);ax[1,i].imshow(vae_samples[i].reshape(8,8),cmap='gray',vmin=0,vmax=1)
 for j in range(2):ax[j,i].axis('off')
ax[0,0].set_title('PCA Gaussian');ax[1,0].set_title('VAE');finish('exp30_generative_comparison')
