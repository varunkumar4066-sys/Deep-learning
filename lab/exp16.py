"""Experiment 16: LSTM and RNN Performance Comparison."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import finish,tf_setup,regression_sequence
tf=tf_setup();X,y=regression_sequence(n=650,length=18);split=int(.8*len(X));res={}
for name,layer in [('SimpleRNN',tf.keras.layers.SimpleRNN),('LSTM',tf.keras.layers.LSTM)]:
 m=tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],1)),layer(24),tf.keras.layers.Dense(1)]);m.compile('adam','mse')
 hist=m.fit(X[:split],y[:split],epochs=20,validation_split=.15,verbose=0)
 pred=m.predict(X[split:],verbose=0);mse=float(np.mean((pred-y[split:])**2));res[name]=hist.history['val_loss'];print(name,'MSE',mse)
for k,v in res.items():plt.plot(v,label=k)
plt.xlabel('Epoch');plt.ylabel('Validation MSE');plt.legend();finish('exp16_rnn_lstm')
