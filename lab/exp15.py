"""Experiment 15: Backpropagation Through Time Gradient Analysis."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish,tf_setup
tf=tf_setup();rng=np.random.default_rng(42);seq=tf.constant(rng.normal(size=(1,30,1)).astype('float32'))
results={}
for scale in [.5,1.0,1.5]:
 W=tf.Variable([[scale]],dtype=tf.float32);U=tf.constant([[1.]],tf.float32)
 with tf.GradientTape(persistent=True) as tape:
  h=tf.zeros((1,1));states=[]
  for t in range(seq.shape[1]):h=tf.tanh(seq[:,t,:]@U+h@W);states.append(h)
  loss=tf.reduce_sum(states[-1]**2)
 gradients=[float(tf.linalg.global_norm([tape.gradient(loss,s)])) for s in states]
 results[str(scale)]=gradients;print('recurrent scale',scale,'gradient at initial state',gradients[0])
 del tape
for label,g in results.items():plt.semilogy(range(30),np.maximum(g,1e-12),label=f'W={label}')
plt.xlabel('Time-step index');plt.ylabel('Gradient magnitude');plt.legend();finish('exp15_bptt_gradients')
