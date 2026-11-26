"""Experiment 20: Probabilistic PCA and Reconstruction Analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from common import finish
X=load_digits().data/16.;results=[]
for dim in [2,5,10,20,40,64]:
 model=PCA(n_components=dim,svd_solver='full');Xrec=model.inverse_transform(model.fit_transform(X));mse=mean_squared_error(X,Xrec);results.append(mse)
 print('latent dim',dim,'reconstruction MSE',round(mse,6),'noise variance',round(float(model.noise_variance_),6))
plt.plot([2,5,10,20,40,64],results,'o-');plt.xlabel('Latent dimensions');plt.ylabel('Reconstruction MSE');plt.title('Probabilistic PCA noise estimate and reconstruction');finish('exp20_ppca')
