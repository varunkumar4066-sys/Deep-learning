"""Experiment 18: RNN, LSTM, GRU Hyperparameter Experimentation."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish,tf_setup,regression_sequence
tf=tf_setup();X,y=regression_sequence(n=450,length=14);split=int(.75*len(X));results=[]
for layer_name in ['SimpleRNN','LSTM','GRU']:
 for units in [8,16]:
  cls=getattr(tf.keras.layers,layer_name);m=tf.keras.Sequential([tf.keras.layers.Input((14,1)),cls(units),tf.keras.layers.Dense(1)]);m.compile('adam','mse');m.fit(X[:split],y[:split],epochs=12,verbose=0)
  mse=float(m.evaluate(X[split:],y[split:],verbose=0));results.append((f'{layer_name}-{units}',mse));print(layer_name,units,'MSE',mse)
plt.bar([x for x,_ in results],[y for _,y in results]);plt.xticks(rotation=45,ha='right');plt.ylabel('Holdout MSE');finish('exp18_rnn_hyperparams')
