#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:51:29 2020

@author: vicky
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from PageRankNibble_undirected import PageRankNibble 
import random
import warnings


def rand_around_centers(centers, number=50, rdm_seed=17):
    
    """
    This function has been taken from Albert's github repository
    https://github.com/AlbMLpy/Clustering_nearby
        Produce random points around centers.
        
        Parameters.
        ----------
        centers : numpy.ndarray
            This array includes x, y coordinates of centers;
        
        number : int
            What number of points do you want scattered around centers;
        
        Returns.
        ----------
        out : numpy.ndarray, numpy.ndarray
            The first array is filled with x, y coordinates of a point;
            The second array is filled with labels tailored to a point;    
    """
    np.random.seed(rdm_seed) 
    result = []
    labels = []
    for c, j in zip(centers, range(len(centers))):
        result.append(c)
        labels.append(j)
        for i in range(number):
            result.append(
                [c[0] + 0.9*np.random.rand(), c[1] + 0.8*np.random.rand()]
            )
            labels.append(j)
    return result, labels



if __name__ == "__main__":  
    warnings.filterwarnings("ignore")
    
    # Make and plot some points(5 clusters):     
    centers = [[1,1],[2,2],[3,3],[4,4],[5,5]]
    user_data, user_labels_data = rand_around_centers(centers, number=50)
    
    x=[]
    y=[]
    for i in user_data:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y,c=user_labels_data,s=10,cmap='viridis')
    plt.xlim((0, 7))
    plt.ylim((0, 7))
    plt.title('Original data points in X-Y plane')
    plt.show()
    
    
    
    # Conver data to graph
    X = np.array(user_data)
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    G = nx.from_numpy_array(nbrs.kneighbors_graph(X).toarray())
    
    
    
    # Run PRN
    Eps = 0.000001  
    alpha = 0.15  
    Seed = 15
    
    PR = PageRankNibble(G, Seed, alpha, Eps)
    H = G.subgraph(PR)
    
    
    
    # Plot original graph
    pos = nx.kamada_kawai_layout(G)
    options = {"node_size": 20, "alpha": 0.8}
    
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color="b", **options)
    nx.draw_networkx_edges(G,pos,edgelist=list(G.edges),width=1,alpha=0.5,edge_color="b",arrows=False)   
    plt.title('Original data points in graph format')
    plt.show()    

    
    
    # Plot the result on X-Y plane (plot the seed as well)
    cluster_points_idx = list(H.nodes())
    cluster_points_label = [user_labels_data[i] for i in cluster_points_idx]
    xx = [x[i] for i in cluster_points_idx]
    yy = [y[i] for i in cluster_points_idx]
    
    plt.scatter(xx,yy,marker='o',c='',edgecolors='r',s=50,alpha=0.5,cmap='viridis') 
    plt.scatter(x,y,c=user_labels_data,s=10,cmap='viridis')
    plt.scatter(x[Seed],y[Seed],c='m',s=100, marker = '*')
    
    plt.xlim((0, 7))
    plt.ylim((0, 7))
    plt.title('Result of PRN in X-Y plane')
    plt.show()
    
    
    
    # Plot PRN result in graph 
    diff_node_lst = [i for i in G.nodes() if i not in H.nodes()]
    diff_edge_lst = [j for j in G.edges if j not in H.edges]
    
    nx.draw_networkx_nodes(G, pos, nodelist=diff_node_lst, node_color="b", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=H.nodes(), node_color='r', **options)
    
    nx.draw_networkx_edges(G,pos,edgelist=diff_edge_lst,width=1,alpha=0.5,edge_color="b",arrows=False)
    nx.draw_networkx_edges(G,pos,edgelist=list(H.edges),width=1,alpha=0.5,edge_color="r",arrows=False)
    plt.title('Result of PRN in graph format')
    plt.show()
    
    
    
    
    # Calculate accuracy
    num_correct = len([item for item in cluster_points_label if item==user_labels_data[Seed]])
    num_pred_points = len(list(H.nodes()))
    num_cls_points = len([item for item in user_labels_data if item==user_labels_data[Seed]])
    
    accu = num_correct/max(num_pred_points,num_cls_points)
    # print(str(num_correct), '/', str(num_pred_points))
    print("Accuracy is: ", accu)