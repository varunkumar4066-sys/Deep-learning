"""Experiment 02: Covariance, Eigenvalues and Eigenvectors Analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import finish
rng=np.random.default_rng(42); x=rng.normal(size=200); y=1.8*x+rng.normal(scale=.5,size=200)
X=np.column_stack([x,y]); cov=np.cov(X,rowvar=False)
values, vectors=np.linalg.eigh(cov); indices=np.argsort(values)[::-1];values=values[indices];vectors=vectors[:,indices]
print(pd.DataFrame(X,columns=['feature_x','feature_y']).head());print('Covariance:',cov,'Eigenvalues:',values,'Eigenvectors:',vectors,sep='\n')
plt.scatter(x,y,s=9,alpha=.5);center=X.mean(axis=0)
for i in range(2):
 d=vectors[:,i]*np.sqrt(values[i])*2;plt.arrow(*center,*d,width=.025,label=f'PC{i+1}')
plt.axis('equal');plt.title('Covariance eigenvectors (principal axes)');finish('exp02_eigen')
