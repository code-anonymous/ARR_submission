import networkx as nx
import networkit as nk
from typing import List, Dict
import scipy as sp
import numpy as np
import random

def get_max_subgraph(graph):
    #sub_graph_nodes = max(nx.connected_components(graph), key=len)
    #graph = graph.subgraph(sub_graph_nodes)
    for u, v in graph.edges():
        graph[u][v]['weight'] = 1
    graph = nk.nxadapter.nx2nk(graph, weightAttr="weight")
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    components = cc.getComponents()
    largest_component = max(components, key=len)
    graph = nk.graphtools.subgraphFromNodes(graph, largest_component, compact=True)
    graph.indexEdges()
    # att = graph.attachEdgeAttribute("weight", float)
    return graph

def generate_graph(dict_graph_info:Dict, num_graphs:int, save_dir:str)->List[nx.Graph]:

    type_graph = dict_graph_info['graph_type']
    list_graphs = []

    for ii in range(num_graphs):
       #seed = ii
        #fast_gnp_random_graph
        if type_graph == 'fast_gnp_random_graph':
            seed = dict_graph_info['seed']
            n = dict_graph_info['num_nodes']
            p = dict_graph_info['p']
            G = nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)
            G = get_max_subgraph(G)
            list_graphs.append(G)

        #powerlaw_cluster_graph
        elif type_graph == 'powerlaw_cluster_graph':
            seed = dict_graph_info['seed']
            n = dict_graph_info['num_nodes']
            m = dict_graph_info['m']
            p = dict_graph_info['p']
            G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
            G = get_max_subgraph(G)
            list_graphs.append(G)

        #barabasi_albert_graph
        elif type_graph == 'barabasi_albert_graph':
            seed = dict_graph_info['seed']
            n = dict_graph_info['num_nodes']
            m = dict_graph_info['m']
            G = nx.barabasi_albert_graph(n, m, seed=seed)
            G = get_max_subgraph(G)
            list_graphs.append(G)

        #gaussian_random_partition_graph
        elif type_graph == 'gaussian_random_partition_graph':
            seed = dict_graph_info['seed']
            n = dict_graph_info['num_nodes']
            s = dict_graph_info['s']
            v = dict_graph_info['v']
            p_in = dict_graph_info['p_in']
            p_out = dict_graph_info['p_out']
            G = nx.gaussian_random_partition_graph(n, s, v, p_in, p_out, seed=seed, directed=False)
            G = get_max_subgraph(G)
            list_graphs.append(G)

    return list_graphs

def random_walk(adj, start_node, walk_length=100, seed=0):
    np.random.seed(seed)
    list_walk = [start_node]
    curr_node = start_node
    while len(list_walk)<=walk_length:
        neigh_arr = adj[curr_node,:]
        tmp_neigh = neigh_arr.nonzero()[1]
        select_node = int(random.choice(tmp_neigh))
        list_walk.append(int(select_node))
        curr_node = select_node

    return list_walk


def my_code(G):
    data_edge = G.iterEdgesWeights()
    u_row_idx, u_col_idx, u_val = map(np.array, zip(*data_edge))
    row_idx = np.concatenate([u_row_idx, u_col_idx])
    col_idx = np.concatenate([u_col_idx, u_row_idx])
    val = np.concatenate([u_val, u_val])
    N = G.numberOfNodes()
    A = sp.sparse.csr_matrix((val, (row_idx, col_idx)), shape=(N,N))
    return A


def get_adjacency_matrix(graph, nodelist):

    ##adj_matrix = nx.adjacency_matrix(graph, nodelist=nodelist)
    #networkit implementation
    #adj_matrix = nk.algebraic.adjacencyMatrix(graph, matrixType="sparse")
    #adj_matrix = adjacencyMatrix(graph, matrixType="sparse")
    adj_matrix = my_code(graph)
    #print(f"Match: {(adj_matrix != adj_matrix1).nnz == 0}")
    rowsum = np.array(adj_matrix.sum(axis=1))
    r_inv = (1/rowsum).flatten()
    diag_mat = sp.sparse.diags(r_inv, offsets=0, shape=(len(nodelist), len(nodelist)))
    adj_matrix = diag_mat.dot(adj_matrix) #normalize the adjacency matrix

    return adj_matrix
