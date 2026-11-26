"""Experiment 09: Multilayer Perceptron Implementation (TensorFlow/Keras)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from common import finish,tf_setup
tf=tf_setup();ds=load_digits();X=ds.data.astype('float32')/16;y=ds.target
xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
model=tf.keras.Sequential([tf.keras.layers.Input((64,)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']);h=model.fit(xtr,ytr,validation_split=.15,epochs=12,batch_size=32,verbose=0)
print('Test loss and accuracy',model.evaluate(xte,yte,verbose=0));plt.plot(h.history['accuracy'],label='train');plt.plot(h.history['val_accuracy'],label='validation');plt.legend();plt.title('MLP digit classification');finish('exp09_mlp')
