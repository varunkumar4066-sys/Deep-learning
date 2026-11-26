"""Experiment 17: RNN Gradient Analysis and Gradient Clipping."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish,tf_setup
tf=tf_setup();rng=np.random.default_rng(42);x=tf.constant(rng.normal(size=(20,28,1)).astype('float32'));y=tf.ones((20,1),dtype=tf.float32)
results={}
for clipping in [None,1.0]:
 model=tf.keras.Sequential([tf.keras.layers.Input((28,1)),tf.keras.layers.SimpleRNN(24),tf.keras.layers.Dense(1)])
 opt=tf.keras.optimizers.SGD(learning_rate=.2);norms=[]
 for step in range(30):
  with tf.GradientTape() as tape:loss=tf.reduce_mean((model(x,training=True)-y)**2)
  grad=tape.gradient(loss,model.trainable_variables);norm=float(tf.linalg.global_norm(grad));norms.append(norm)
  if clipping is not None:grad,_=tf.clip_by_global_norm(grad,clipping)
  opt.apply_gradients(zip(grad,model.trainable_variables))
 key='clip=1' if clipping else 'no clipping';results[key]=norms;print(key,'maximum un-clipped gradient norm',max(norms))
for k,v in results.items():plt.plot(v,label=k)
plt.yscale('log');plt.ylabel('Raw gradient norm');plt.xlabel('Training step');plt.legend();finish('exp17_clipping')
