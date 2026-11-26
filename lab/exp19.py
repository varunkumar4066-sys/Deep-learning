"""Experiment 19: PCA and Explained Variance Analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import finish
X,y=load_iris(return_X_y=True);X=StandardScaler().fit_transform(X);pca=PCA();Z=pca.fit_transform(X)
print('Explained variance ratio:',pca.explained_variance_ratio_);print('Cumulative:',np.cumsum(pca.explained_variance_ratio_))
fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].scatter(Z[:,0],Z[:,1],c=y,cmap='viridis');axs[0].set_xlabel('PC1');axs[0].set_ylabel('PC2')
axs[1].plot(range(1,5),np.cumsum(pca.explained_variance_ratio_),'o-');axs[1].set_xlabel('Components');axs[1].set_ylabel('Cumulative explained variance');finish('exp19_pca')
