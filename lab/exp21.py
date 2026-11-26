"""Experiment 21: Factor Analysis and Latent Factor Interpretation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from common import finish
rng=np.random.default_rng(42);f=rng.normal(size=(500,2));loadings=np.array([[.9,.1],[.8,.2],[.7,.1],[.1,.9],[.2,.8],[0,.85]])
X=f@loadings.T+rng.normal(scale=.25,size=(500,6));names=['Math','Physics','Coding','Art','Music','Design'];X=StandardScaler().fit_transform(X)
model=FactorAnalysis(n_components=2,random_state=42);latent=model.fit_transform(X)
print('Factor loadings (rotation/sign indeterminate):');print(pd.DataFrame(model.components_.T,index=names,columns=['Factor1','Factor2']).round(3))
plt.imshow(model.components_.T,cmap='coolwarm',aspect='auto',vmin=-1,vmax=1);plt.yticks(range(6),names);plt.xticks([0,1],['F1','F2']);plt.colorbar();finish('exp21_factor_analysis')
