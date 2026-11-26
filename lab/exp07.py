"""Experiment 07: Perceptron and Forward Propagation (AND gate)."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish
X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,0,1]);w=np.zeros(2);b=0.;history=[]
for epoch in range(12):
 mistakes=0
 for xi, target in zip(X,y):
  pred=int(xi@w+b>0);delta=target-pred;w+=.3*delta*xi;b+=.3*delta;mistakes+=int(delta!=0)
 history.append(mistakes)
print('Final weights',w,'bias',b,'outputs',[(xi.tolist(),int(xi@w+b>0)) for xi in X]);plt.plot(history,'o-');plt.ylabel('Errors per epoch');plt.xlabel('Epoch');finish('exp07_perceptron')
