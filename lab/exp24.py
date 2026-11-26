"""Experiment 24: Graphical Model Sampling and Probabilistic Inference."""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from common import finish
rng=np.random.default_rng(42);N=40000;pB=.01;pE=.02;pA={(0,0):.001,(0,1):.29,(1,0):.94,(1,1):.95}
B=rng.random(N)<pB;E=rng.random(N)<pE;A=np.array([rng.random()<pA[(int(b),int(e))] for b,e in zip(B,E)]);C=rng.random(N)<np.where(A,.90,.05)
df=pd.DataFrame({'Burglary':B,'Earthquake':E,'Alarm':A,'Call':C})
mc=float(B[C].mean());numerator=0.;denominator=0.
for b in (0,1):
 for e in (0,1):
  prior=(pB if b else 1-pB)*(pE if e else 1-pE);likelihood=pA[(b,e)]*.9+(1-pA[(b,e)])*.05
  denominator+=prior*likelihood
  if b:numerator+=prior*likelihood
exact=numerator/denominator
print('P(Burglary|Call) exact',round(exact,5),'Monte Carlo',round(mc,5),'absolute error',round(abs(exact-mc),5));print(df.head())
plt.bar(['Exact','Monte Carlo'],[exact,mc]);plt.ylim(0,max(exact,mc)*1.5);plt.ylabel('P(Burglary | Call)');finish('exp24_inference')
