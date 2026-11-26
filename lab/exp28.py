"""Experiment 28: Deep Boltzmann Machine / Convolutional Boltzmann Machine Simulation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from common import finish
# Small two-hidden-layer undirected DBM: Gibbs updates for binary units.
rng=np.random.default_rng(42);W1=rng.normal(0,.5,(16,8));W2=rng.normal(0,.4,(8,4))
v=(rng.random(16)>.5).astype(float);h1=np.zeros(8);h2=np.zeros(4);samples=[]
for t in range(250):
 p1=expit(v@W1+h2@W2.T);h1=(rng.random(8)<p1).astype(float)
 p2=expit(h1@W2);h2=(rng.random(4)<p2).astype(float)
 pv=expit(W1@h1);v=(rng.random(16)<pv).astype(float)
 if t>=50:samples.append(v.copy())
samples=np.array(samples);print('DBM visible mean:',samples.mean(0).round(3));print('Approx. Gibbs samples:',len(samples))
fig,ax=plt.subplots(1,2,figsize=(7,3));ax[0].imshow(samples[:1].reshape(4,4),cmap='gray',vmin=0,vmax=1);ax[0].set_title('Sample');ax[1].imshow(samples.mean(0).reshape(4,4),cmap='magma',vmin=0,vmax=1);ax[1].set_title('Average activation');finish('exp28_dbm')
