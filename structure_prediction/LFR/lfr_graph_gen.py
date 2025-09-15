import networkit as nk
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from utils import random_walk, get_adjacency_matrix
import random
import tiktoken
import pickle
import json
random.seed(42)

def generate_graph_nk(list_param):
    t1 = time.time()
    n, mincomm, maxcomm, comm_exp, avg_deg, max_deg, deg_exp = list_param
    lfr = nk.generators.LFRGenerator(n)
    lfr.generatePowerlawDegreeSequence(mincomm, maxcomm, comm_exp)
    lfr.generatePowerlawCommunitySizeSequence(avg_deg, max_deg, deg_exp)
    lfr.setMu(0.3)

    lfr.run()
    graph = lfr.getGraph()
    partition = lfr.getPartition()
    num_partitions = len(Counter(partition))
    g_nx = nk.nxadapter.nk2nx(graph)
    print(f"Graph info: {graph}")
    print(f"Number of partitions: {num_partitions}")
    t2 = time.time()
    print(f"Time taken to generate graph: {t2 - t1} seconds")
    return g_nx, partition

def generate_graph_nx(list_param):
    t1 = time.time()
    n, mincomm, maxcomm, comm_exp, avg_deg, max_deg, deg_exp, seed = list_param
    #catch errors
    try_count = 10
    for ii in range(try_count):
        try:
            t1 = time.time()
            #tau1 = degree distribution of the graph
            #tau2 = community size distribution of the graph
            graph = nx.generators.LFR_benchmark_graph(n, tau1=comm_exp, tau2=deg_exp, average_degree=avg_deg, min_community=mincomm,max_community=maxcomm,mu=0.02, seed=seed)
            partition = {frozenset(graph.nodes[i]['community']) for i in graph}
            print(f"Number of partitions: {len(partition)}")
            print(f"Graph info: {graph}")
            t2 = time.time()
            print(f"Time taken to generate graph: {t2 - t1} seconds")
            return graph, partition
        except Exception as e:
            print(f"Error generating graph: {e}")
            seed += 100
            if ii == 9:
                return None, None


def get_prompt(graph, list_start_nodes, walk_length):
    nodelist = sorted(list(graph.iterNodes()))
    adj_matrix = get_adjacency_matrix(graph, nodelist)

    initial_prompt = (f"You are a graph theory expert. Following are statistics of simple random walks performed on a large graph. The walks may not cover the whole graph.\n"
        f"For each walk, walk statistics are provided as a dictionary: {{<node name>:(<number of times node appears in the walk>, <degree of the node>), }}.\n"
        f"Use this information to predict if the graph is one of the following types: Barabasi-Albert (scale-free graph), Erdos-Renyi (random graph), LFR (Lancichinetti–Fortunato–Radicchi benchmark graph), or a grid graph (regular lattice). \n"
        f"Provide your answer in the following format, Graph Type: <type of graph> (choose one from: BA, ER, LFR, Grid).\n")


    for idx, start_node in enumerate(list_start_nodes):
        seed = idx
        list_walk = random_walk(adj_matrix, start_node, walk_length=walk_length, seed=seed)
        unique_walk_nodes = list(set(list_walk))
        degree_walk_nodes = [graph.degree(nn) for nn in unique_walk_nodes]
        rev_sort_idx = np.argsort(degree_walk_nodes)[::-1]

        count_walk = Counter(list_walk)
        #nodes are sorted by degree from highest to lowest
        list_walk_info = {unique_walk_nodes[nn]:(count_walk[unique_walk_nodes[nn]], degree_walk_nodes[nn]) for nn in rev_sort_idx}
        list_walk_info = str(list_walk_info).replace(" ", "")
        list_walk_info = list_walk_info.replace("),", "), ")

        initial_prompt += f"Walk {idx+1}: {str(list_walk_info)}\n"
        initial_prompt += f"Length of walk: {len(list_walk)}. Number of unique nodes in the walk: {len(unique_walk_nodes)}\n"

    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(initial_prompt))
    print(f"Number of tokens in the prompt: {num_tokens}")

    return initial_prompt, num_tokens



'''
dict_graph_params = {
    0: [1000, 100, 500, 3, 10, 400, 1.5],
    1: [2000, 150, 1000, 3, 10, 500, 1.5],
    2: [3000, 200, 1500, 3, 10, 1000, 1.5],
    3: [4000, 250, 2000, 3, 10, 2000, 1.5],
    4: [5000, 300, 2000, 4, 10, 2000, 1.5],
}
'''

dict_graph_params = {
    0: [1000, 100, 300, 4, 10, 400, 2],
    1: [2000, 150, 400, 4, 10, 500, 2],
    2: [3000, 200, 500, 4, 10, 1000, 2],
    3: [4000, 250, 600, 4, 10, 2000, 2],
    4: [5000, 300, 700, 4, 10, 2000, 2],
    5: [6000, 350, 800, 4, 10, 2500, 2],
    6: [7000, 400, 1000, 4, 10, 2500, 2],
    7: [8000, 400, 1200, 4, 10, 3000, 2],
    8: [9000, 500, 1500, 4, 10, 3000, 2],
    9: [10000, 500, 2000, 4, 10, 3500, 2],
}

list_graph = []

dict_prompts = defaultdict(dict)

for ii in range(15):
    print(f"==== Graph: {ii} ====")
    idx = random.randint(0,9)
    value = dict_graph_params[idx]
    value = value + [ii]  # Add a seed based on the iteration
    print(f"Parameters: {value}")
    graph, partition = generate_graph_nx(value)
    print(f"NetworkX graph info: {graph}")

    if graph == None:
        print(f"Failed to generate graph for parameters: {value}")
        continue

    # saving in case to plot degree distribution
    list_graph.append(nk.nxadapter.nx2nk(graph))

    start_nodes = random.sample(list(graph.nodes()), 5)
    graph_nk = nk.nxadapter.nx2nk(graph)
    print(f"Networkit graph info: {graph_nk}")

    walk_length = int(graph.number_of_nodes()*0.1)

    initial_prompt, num_tokens = get_prompt(graph_nk, start_nodes, walk_length)
    dict_prompts[ii]['graph_type'] = "LFR"
    dict_prompts[ii]['prompt'] = initial_prompt
    dict_prompts[ii]['graph_size'] = graph.number_of_nodes()
    dict_prompts[ii]['num_partitions'] = len(partition)
    dict_prompts[ii]['num_tokens'] = num_tokens


# Save the prompts to a file
with open("lfr_prompts.json", "w") as f:
    json.dump(dict_prompts, f, indent=4)

print(f"Generated {len(list_graph)} graphs with prompts.")
# Save the graphs to a file
with open("list_graph.pickle", "wb") as f:
    pickle.dump(list_graph, f)
