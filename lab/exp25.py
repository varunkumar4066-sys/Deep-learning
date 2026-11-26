"""Experiment 25: Restricted Boltzmann Machine Energy and Reconstruction."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from common import finish
rng=np.random.default_rng(42);W=rng.normal(scale=.6,size=(6,3));vb=np.zeros(6);hb=np.zeros(3)
v=np.array([1,0,1,0,1,0]);hprob=expit(v@W+hb);h=(rng.random(3)<hprob).astype(int);vprob=expit(h@W.T+vb)
energy=lambda vv,hh:-vv@vb-hh@hb-vv@W@hh
free_energy=lambda vv:-vv@vb-np.logaddexp(0,vv@W+hb).sum()
print('Visible:',v,'Hidden p(h=1|v):',hprob.round(3),'Reconstruction p(v=1|h):',vprob.round(3))
print('Energy E(v,h):',round(energy(v,h),4),'Free energy F(v):',round(free_energy(v),4))
plt.bar(range(6),vprob);plt.ylim(0,1);plt.xlabel('Visible bit');plt.ylabel('Reconstruction probability');finish('exp25_rbm_energy')
