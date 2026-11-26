"""Experiment 08: Two-Layer Neural Network with Backpropagation (XOR)."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish
rng=np.random.default_rng(42);X=np.array([[0,0],[0,1],[1,0],[1,1.]],float);y=np.array([[0],[1],[1],[0.]])
W1=rng.normal(size=(2,8))*.7;b1=np.zeros((1,8));W2=rng.normal(size=(8,1))*.7;b2=np.zeros((1,1))
sig=lambda z:1/(1+np.exp(-z)); losses=[]
for epoch in range(5000):
 h=sig(X@W1+b1);p=sig(h@W2+b2)
 loss=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9));losses.append(loss)
 dz2=(p-y)/len(X);dW2=h.T@dz2;db2=dz2.sum(axis=0,keepdims=True)
 dz1=(dz2@W2.T)*h*(1-h);dW1=X.T@dz1;db1=dz1.sum(axis=0,keepdims=True)
 W2-=1.2*dW2;b2-=1.2*db2;W1-=1.2*dW1;b1-=1.2*db1
print('XOR prediction:',sig(sig(X@W1+b1)@W2+b2).round(3).ravel());plt.plot(losses);plt.xlabel('Epoch');plt.ylabel('Binary cross entropy');finish('exp08_backprop')
