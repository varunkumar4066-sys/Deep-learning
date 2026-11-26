"""Experiment 06: Gradient Descent, SGD, Momentum and Adam Comparison."""
import numpy as np
import matplotlib.pyplot as plt
from common import finish
# Minimize f(w)=(w-3)^2+1; stochastic variant uses noisy gradient.
rng=np.random.default_rng(42);hist={}
for method in ['GD','SGD','Momentum','Adam']:
 w=-5.;v=0.;m=0.;u=0.;track=[]
 for t in range(1,101):
  g=2*(w-3)+(rng.normal(scale=2) if method=='SGD' else 0)
  if method in ['GD','SGD']:w-=.07*g
  elif method=='Momentum':v=.85*v+.025*g;w-=v
  else:
   m=.9*m+.1*g;u=.999*u+.001*g*g;w-=.2*(m/(1-.9**t))/(np.sqrt(u/(1-.999**t))+1e-8)
  track.append((w-3)**2+1)
 hist[method]=track;print(method,'final weight',round(w,4),'loss',round(track[-1],4))
for method, values in hist.items():plt.plot(values,label=method)
plt.yscale('log');plt.xlabel('Iteration');plt.ylabel('Objective');plt.legend();plt.title('Optimizer comparison');finish('exp06_optimizers')
