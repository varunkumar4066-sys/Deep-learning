"""Experiment 10: L2, Dropout, Early Stopping Comparison."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from common import finish,tf_setup
tf=tf_setup();X,y=make_moons(n_samples=700,noise=.3,random_state=42);xtr,xte,ytr,yte=train_test_split(X.astype('float32'),y,test_size=.3,random_state=42)
results={}
for kind in ['baseline','l2','dropout','early_stopping']:
 layers=[tf.keras.layers.Input((2,)),tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(.01) if kind=='l2' else None)]
 if kind=='dropout':layers.append(tf.keras.layers.Dropout(.35))
 layers += [tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(1,activation='sigmoid')]
 m=tf.keras.Sequential(layers);m.compile('adam','binary_crossentropy',metrics=['accuracy'])
 callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)] if kind=='early_stopping' else []
 h=m.fit(xtr,ytr,validation_split=.2,epochs=45,batch_size=32,verbose=0,callbacks=callbacks);results[kind]=(h.history['val_loss'],m.evaluate(xte,yte,verbose=0)[1])
 print(kind,'test accuracy',round(results[kind][1],3),'epochs',len(h.history['loss']))
for kind,(loss,acc) in results.items():plt.plot(loss,label=kind)
plt.legend();plt.xlabel('Epoch');plt.ylabel('Validation loss');finish('exp10_regularization')
