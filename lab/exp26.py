"""Experiment 26: Gibbs Sampling and Contrastive Divergence."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from common import finish
rng=np.random.default_rng(42);data=np.array([[1,1,0,0],[1,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,0,0]],dtype=float)
W=rng.normal(0,.1,(4,2));a=np.zeros(4);b=np.zeros(2);history=[];epochs=400
for ep in range(epochs):
 v0=data;ph0=expit(v0@W+b);h0=(rng.random(ph0.shape)<ph0).astype(float)
 # CD-1: one hidden-to-visible and visible-to-hidden Gibbs step
 pv1=expit(h0@W.T+a);v1=(rng.random(pv1.shape)<pv1).astype(float);ph1=expit(v1@W+b)
 lr=.1;W+=lr*(v0.T@ph0-v1.T@ph1)/len(data);a+=lr*(v0-v1).mean(0);b+=lr*(ph0-ph1).mean(0)
 history.append(np.mean((v0-pv1)**2))
print('Reconstruction loss first/last:',history[0],history[-1]);print('Learned visible bias',a.round(3))
plt.plot(history);plt.xlabel('Epoch');plt.ylabel('CD-1 reconstruction error');finish('exp26_cd')
