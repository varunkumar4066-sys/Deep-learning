"""Experiment 12: CNN Architecture and Hyperparameter Comparison."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from common import finish,tf_setup
tf=tf_setup();X,y=load_digits(return_X_y=True);X=X.reshape(-1,8,8,1).astype('float32')/16;xtr,xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
results=[]
for filters,kernel,batch in [(8,3,32),(16,3,32),(16,5,64),(32,3,64)]:
 model=tf.keras.Sequential([tf.keras.layers.Input((8,8,1)),tf.keras.layers.Conv2D(filters,kernel,padding='same',activation='relu'),tf.keras.layers.MaxPool2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(10,activation='softmax')]);model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']);model.fit(xtr,ytr,epochs=8,batch_size=batch,verbose=0)
 acc=model.evaluate(xte,yte,verbose=0)[1];results.append(acc);print(filters,kernel,batch,'parameters',model.count_params(),'accuracy',round(acc,3))
plt.bar(range(len(results)),results);plt.xticks(range(len(results)),['8/3/32','16/3/32','16/5/64','32/3/64']);plt.ylim(0,1);plt.ylabel('Test accuracy');plt.xlabel('Filters/kernel/batch');finish('exp12_cnn_hyperparameters')
