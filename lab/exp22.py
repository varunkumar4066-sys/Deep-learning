"""Experiment 22: Independent Component Analysis and Visualization."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from common import finish
rng=np.random.default_rng(42);t=np.linspace(0,8,1500);sources=np.column_stack([np.sin(5*t),np.sign(np.sin(3*t)),rng.laplace(size=len(t))]);sources/=sources.std(0)
A=np.array([[1.,.5,.4],[.4,1.,.2],[.3,.6,1.]]);mix=sources@A.T
recovered=FastICA(n_components=3,random_state=42,max_iter=1000,tol=.001,whiten='unit-variance').fit_transform(mix)
fig,axes=plt.subplots(3,1,figsize=(9,6))
for i,(data,title) in enumerate([(sources,'Original'),(mix,'Mixed'),(recovered,'Recovered (permuted/sign-flipped)')]):axes[i].plot(t[:200],data[:200]);axes[i].set_title(title)
finish('exp22_ica')
