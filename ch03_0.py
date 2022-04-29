#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:08:49 2022

@author: ahmad
"""
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
    
download('https://github.com/AllenDowney/ThinkComplexity2/raw/master/notebooks/utils.py')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from utils import decorate, savefig

# Set the random seed so the notebook 
# produces the same results every time.
np.random.seed(17)

# make a directory for figures
!mkdir -p figs

# node colors for drawing networks
colors = sns.color_palette('pastel', 5)
#sns.palplot(colors)
sns.set_palette(colors)

def adjacent_edges(nodes, halfk):
    n = len(nodes)
    for i, u in enumerate(nodes):
        for j in range(i+1, i+halfk+1):
            v = nodes[j % n]
            yield u, v
    
nodes = range(4)
for edge in adjacent_edges(nodes, 2):
    print(edge)
    
def make_ring_lattice(n,k):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(adjacent_edges(nodes,k//2))
    return G


rlgraph = make_ring_lattice(10,2)

nx.draw_circular(rlgraph,
                 node_color='C2',
                 node_size=1000,
                 with_labels=True)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
