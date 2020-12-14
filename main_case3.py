#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:11:10 2020

@author: vicky
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_circles
from sklearn.neighbors import NearestNeighbors
from PageRankNibble_undirected import PageRankNibble 
from sklearn.datasets import load_wine



if __name__ == "__main__":  
    np.random.seed(17)
    
    # Load iris dataset  
    wine = load_wine()
    
    data = wine.data
    user_data = data.tolist()
    
    target = wine.target
    user_labels_data = target.tolist()



    # Conver data to graph
    X = np.array(user_data)
    nbrs = NearestNeighbors(n_neighbors=25, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    G = nx.from_numpy_array(nbrs.kneighbors_graph(X).toarray())


    # Run PRN
    Eps = 0.000001
    alpha = 0.15  
    Seed = 0
    
    PR = PageRankNibble(G, Seed, alpha, Eps)
    H = G.subgraph(PR)
    
    
    
    # Plot original graph
    pos = nx.kamada_kawai_layout(G)
    options = {"node_size": 20, "alpha": 0.8}
    
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color="b", **options)
    nx.draw_networkx_edges(G,pos,edgelist=list(G.edges),width=1,alpha=0.5,edge_color="b",arrows=False)   
    plt.title('Original data points in graph format')
    plt.show()  
    
    
    
    
    # Plot PRN result    
    diff_node_lst = [i for i in G.nodes() if i not in H.nodes()]
    diff_edge_lst = [j for j in G.edges if j not in H.edges]
    
    nx.draw_networkx_nodes(G, pos, nodelist=diff_node_lst, node_color="b", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=H.nodes(), node_color='r', **options)
    
    nx.draw_networkx_edges(G,pos,edgelist=diff_edge_lst,width=1,alpha=0.5,edge_color="b",arrows=False)
    nx.draw_networkx_edges(G,pos,edgelist=list(H.edges),width=1,alpha=0.5,edge_color="r",arrows=False)
    plt.title('Result of PRN in graph format')
    plt.show()
    
    
    
    # Calculate accuracy
    cluster_points_idx = list(H.nodes())
    cluster_points_label = [user_labels_data[i] for i in cluster_points_idx]
    
    num_correct = len([item for item in cluster_points_label if item==user_labels_data[Seed]])
    num_pred_points = len(list(H.nodes()))
    num_cls_points = len([item for item in user_labels_data if item==user_labels_data[Seed]])
    
    accu = num_correct/max(num_pred_points,num_cls_points)
    # print(str(num_correct), '/', str(num_pred_points))
    print("Accuracy is: ", accu)
    
    
    
    