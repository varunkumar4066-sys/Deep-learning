"""Experiment 11: CNN for Image Classification (sklearn digits)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from common import finish,tf_setup
tf=tf_setup();X,y=load_digits(return_X_y=True);X=X.reshape(-1,8,8,1).astype('float32')/16
xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
m=tf.keras.Sequential([tf.keras.layers.Input((8,8,1)),tf.keras.layers.Conv2D(16,3,activation='relu',padding='same'),tf.keras.layers.MaxPool2D(),tf.keras.layers.Conv2D(32,3,activation='relu'),tf.keras.layers.Flatten(),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(10,activation='softmax')])
m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']);m.fit(xtr,ytr,epochs=10,batch_size=32,verbose=0)
print('CNN test accuracy',m.evaluate(xte,yte,verbose=0)[1]);pred=m.predict(xte[:8],verbose=0).argmax(1)
fig,axs=plt.subplots(2,4,figsize=(9,5))
for i,ax in enumerate(axs.flat):ax.imshow(xte[i,:,:,0],cmap='gray');ax.set_title(f'True {yte[i]}, Pred {pred[i]}');ax.axis('off')
finish('exp11_cnn')
