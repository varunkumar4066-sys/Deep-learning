"""Experiment 05: Bias-Variance, Overfitting and Underfitting Analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from common import finish
rng=np.random.default_rng(42);x=np.linspace(-3,3,80);y=np.sin(x)+rng.normal(0,.25,len(x));idx=rng.permutation(len(x));train=idx[:25];test=idx[25:]
models=[1,3,12]; xx=np.linspace(-3,3,300)
fig,axs=plt.subplots(1,2,figsize=(11,4));tr=[];te=[]
for degree in models:
 model=make_pipeline(PolynomialFeatures(degree),LinearRegression());model.fit(x[train,None],y[train]);pred=model.predict(xx[:,None]);axs[0].plot(xx,pred,label=f'degree {degree}');tr.append(mean_squared_error(y[train],model.predict(x[train,None])));te.append(mean_squared_error(y[test],model.predict(x[test,None])))
axs[0].scatter(x[train],y[train],s=12,color='black');axs[0].set_ylim(-2,2);axs[0].legend();axs[1].plot(models,tr,'o-',label='train');axs[1].plot(models,te,'s-',label='test');axs[1].set_xlabel('Polynomial degree');axs[1].set_ylabel('MSE');axs[1].legend()
print('degree train_MSE test_MSE');[print(d,round(a,3),round(b,3)) for d,a,b in zip(models,tr,te)];finish('exp05_bias_variance')
