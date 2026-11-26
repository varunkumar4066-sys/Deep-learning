"""Experiment 13: Sequential Dataset Preparation and RNN Prediction."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from common import finish,tf_setup,regression_sequence
tf=tf_setup();X,y=regression_sequence();split=int(len(X)*.8);xtr,xte=X[:split],X[split:];ytr,yte=y[:split],y[split:]
m=tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],1)),tf.keras.layers.SimpleRNN(16),tf.keras.layers.Dense(1)]);m.compile('adam','mse');m.fit(xtr,ytr,epochs=15,validation_split=.2,verbose=0)
p=m.predict(xte,verbose=0).ravel();print('Test MSE',np.mean((p-yte.ravel())**2));print(pd.DataFrame({'actual':yte.ravel()[:6],'predicted':p[:6]}))
plt.plot(yte.ravel(),label='actual');plt.plot(p,label='RNN prediction');plt.legend();finish('exp13_rnn_prediction')
