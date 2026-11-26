"""Experiment 27: Deep Belief Network Representation Learning (stacked RBMs)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from common import finish
X,y=load_digits(return_X_y=True);X=np.clip(X/16,0,1);xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
rbm1=BernoulliRBM(n_components=64,learning_rate=.04,n_iter=7,batch_size=32,random_state=42,verbose=False)
rbm2=BernoulliRBM(n_components=32,learning_rate=.04,n_iter=7,batch_size=32,random_state=42,verbose=False)
z1=rbm1.fit_transform(xtr);z2=rbm2.fit_transform(z1);test_z2=rbm2.transform(rbm1.transform(xte))
clf=LogisticRegression(max_iter=800,random_state=42).fit(z2,ytr);acc=clf.score(test_z2,yte)
print('Stacked RBM hidden shapes:',z1.shape,z2.shape,'classification accuracy:',round(acc,3))
plt.imshow(z2[:70].T,aspect='auto',cmap='viridis');plt.xlabel('Training samples');plt.ylabel('Second-layer features');finish('exp27_dbn')
