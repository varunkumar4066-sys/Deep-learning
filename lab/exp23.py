"""Experiment 23: Bayesian Network Construction and Visualization."""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from common import finish
# Burglary -> Alarm <- Earthquake; Alarm -> Call
edges=[('Burglary','Alarm'),('Earthquake','Alarm'),('Alarm','Call')]
g=nx.DiGraph(edges);assert nx.is_directed_acyclic_graph(g)
# Use explicit CPTs for valid probabilistic interpretation.
P_B=.01;P_E=.02
p_alarm={(0,0):.001,(0,1):.29,(1,0):.94,(1,1):.95};p_call={0:.05,1:.90}
p_call_total=sum((P_B if b else 1-P_B)*(P_E if e else 1-P_E)*(p_alarm[(b,e)]*p_call[1]+(1-p_alarm[(b,e)])*p_call[0]) for b in (0,1) for e in (0,1))
print('P(Call=1)=',round(p_call_total,6),'DAG edges=',edges)
pos=nx.spring_layout(g,seed=42);nx.draw_networkx(g,pos,arrows=True,node_size=2200,node_color='#d6ecff',font_size=10);plt.axis('off');finish('exp23_bayesian_network')
# To implement with pgmpy, see https://pgmpy.org/models/bayesiannetwork.html
