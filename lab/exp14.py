"""Experiment 14: Unfolded RNN Architecture Visualization."""
import networkx as nx
import matplotlib.pyplot as plt
from common import finish,tf_setup
tf=tf_setup();rnn=tf.keras.layers.SimpleRNN(4,return_sequences=True);rnn.build((None,5,1));print('RNN parameters:',rnn.count_params())
g=nx.DiGraph();pos={}
for t in range(5):
 for label,y in [('x',0),('h',1),('y',2)]:g.add_node(f'{label}{t}');pos[f'{label}{t}']=(t,y)
 g.add_edge(f'x{t}',f'h{t}');g.add_edge(f'h{t}',f'y{t}')
 if t:g.add_edge(f'h{t-1}',f'h{t}')
nx.draw(g,pos,with_labels=True,node_color='#d9eafc',node_size=1100,arrows=True);plt.title('RNN unrolled across five time steps');finish('exp14_unfolded_rnn')
