#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 22:40:02 2022

@author: ahmad
"""
# initial setup
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

# Complete graph
def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j,v in enumerate(nodes):
            if i< j:
                yield u,v

def make_complete_graph(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G


complete = make_complete_graph(10)
complete.number_of_nodes()

nx.draw_circular(complete, 
                 node_color='C2', 
                 node_size=1000, 
                 with_labels=True)
    
def reachable_nodes(G,start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen

reachable_nodes(complete,0)

def is_connected(G):
    start=next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)

# Erdos-Renyi (ER) graph

def random_pairs(nodes,p):
    for edge in all_pairs(nodes):
        if flip(p):
            yield edge
            
def flip(p):
    return np.random.random() < p

def make_random_graph(n,p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes,p))
    return G
            
random_graph = make_random_graph(10,0.3)

nx.draw_circular(random_graph,
                 node_color='C2',
                 node_size=1000,
                 with_labels=True)

is_connected(random_graph)

def prob_connected(n, p, iters=1000):
    tf = [is_connected(make_random_graph(n, p)) for i in range(iters)]
    return np.mean(tf)

prob_connected(10,0.23,10000)

n = 100
ps = np.logspace(-2.5,0,11)
ys = [prob_connected(n, p) for p in ps]

# Exercise 2.3

def reachable_nodes_sets(G,start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(set(G.neighbors(node)).difference(seen))
    return seen

def is_connected_sets(G):
    start=next(iter(G))
    reachable = reachable_nodes_sets(G, start)
    return len(reachable) == len(G)
            
reachable_nodes(complete,0) 
reachable_nodes_sets(complete,0)          
            
def prob_connected_sets(n, p, iters=1000):
       tf = [is_connected_sets(make_random_graph(n, p)) for i in range(iters)]
       return np.mean(tf)     

prob_connected(10,0.23,100000)
prob_connected_sets(10,0.23,100000)    

%timeit prob_connected(10,0.23,10000)
%timeit prob_connected_sets(10,0,10000)

# Exercise 2.4

import random

def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j,v in enumerate(nodes):
            if i< j:
                yield u,v

def m_pairs(nodes,m):
    edge_list = list(all_pairs(nodes))
    edge_sample = random.sample(edge_list,m)
    return edge_sample

def make_m_graph(n,m):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(m_pairs(nodes,m))
    return G

m_graph = make_m_graph(10,10)

nx.draw_circular(m_graph,
                 node_color='C2',
                 node_size=1000,
                 with_labels=True)

is_connected(m_graph)
    
def prob_connected_mgraph(n, m, iters=1000):
    tf = [is_connected(make_m_graph(n, m)) for i in range(iters)]
    return np.mean(tf)

prob_connected_mgraph(10,15)

x = range(40)
y = [prob_connected_mgraph(20,m) for m in x]
list(enumerate(y))

    
    
    
    
    




    

