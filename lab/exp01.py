"""Experiment 01: Vector and Matrix Operations for Deep Learning."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish
v=np.array([1.,2.,3.]); w=np.array([4.,-1.,2.]); A=np.array([[1.,2.],[3.,4.]])
B=np.array([[2.,0.],[1.,3.]])
print('v+w:',v+w,'dot:',v@w,'L2 norms:',np.linalg.norm(v),np.linalg.norm(w))
print('A+B:',A+B,'A@B:',A@B,'transpose:',A.T,sep='\n')
print('Matrix-vector:',A@np.array([2.,1.]),'det(A):',np.linalg.det(A))
plt.quiver([0,0],[0,0],[v[0],w[0]],[v[1],w[1]],angles='xy',scale_units='xy',scale=1)
plt.xlim(-2,6);plt.ylim(-3,5);plt.grid();plt.title('Vector directions (first two coordinates)');finish('exp01_vectors')
