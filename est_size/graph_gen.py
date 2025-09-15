import networkx as nx
import json
from typing import List, Dict
import os
import time
import networkit as nk
from collections import defaultdict
import string
import argparse


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate graphs and estimate properties")
    parser.add_argument("--config_file", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        data_graph = json.load(f)

    num_graphs = 1
    save_dir = "./generated_graphs/" # directory to save generated graphs



    for idx, dict_graph_info in data_graph.items():
        graph_id = dict_graph_info['graph_id']
        graph_type = dict_graph_info['graph_type']
        graph_size = dict_graph_info['graph_size']
        os.makedirs(save_dir+graph_type, exist_ok=True)
        list_graphs = generate_graph(dict_graph_info, num_graphs, save_dir)
