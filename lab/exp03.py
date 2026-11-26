"""Experiment 03: Probability Distributions and Bayes Theorem."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from common import finish
prior=.01;sensitivity=.95;specificity=.90
posterior=sensitivity*prior/(sensitivity*prior+(1-specificity)*(1-prior))
print(f'P(disease | positive) = {posterior:.5f}')
x=np.linspace(-4,4,300);fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(x,norm.pdf(x,0,1),label='Normal 0,1');axes[0].plot(x,norm.pdf(x,1,1.5),label='Normal 1,1.5');axes[0].legend()
axes[1].bar(np.arange(11),binom.pmf(np.arange(11),10,.3));axes[1].set_title('Binomial n=10, p=.3');finish('exp03_probability')
