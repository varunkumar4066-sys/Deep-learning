"""Experiment 04: Entropy, Cross-Entropy, and KL Divergence."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from common import finish
p=np.array([.7,.2,.1]);q=np.array([.5,.3,.2]);eps=1e-12
H=-np.sum(p*np.log2(p+eps));H_cross=-np.sum(p*np.log2(q+eps));KL=np.sum(rel_entr(p,q))/np.log(2)
print(f'Entropy {H:.5f} bits, cross-entropy {H_cross:.5f} bits, KL {KL:.5f} bits');print('Check CE-H = KL:',H_cross-H)
x=np.arange(len(p));plt.bar(x-.2,p,.4,label='P');plt.bar(x+.2,q,.4,label='Q');plt.xticks(x,['A','B','C']);plt.legend();plt.title('Comparing distributions');finish('exp04_entropy')
